"""
RDM Sampler Wrapper for StyleGAN2 Training
Loads pretrained Unified Segmentation RDM and samples (G,S) embeddings.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path


class RDMSampler:
    """Wrapper to load pretrained RDM and sample (Global, Segment) embeddings."""
    
    def __init__(self, rdm_checkpoint_path, device='cuda', use_ema=True, ddim_steps=50):
        """
        Args:
            rdm_checkpoint_path: Path to pretrained RDM checkpoint (.pt or .ckpt)
            device: Device to load model on
            use_ema: Use EMA weights if available
            ddim_steps: Number of DDIM steps for fast sampling (default: 50)
        """
        self.device = device
        self.use_ema = use_ema
        self.ddim_steps = ddim_steps
        self.model = None
        self.ddim_sampler = None
        
        # Load checkpoint
        print(f"Loading RDM checkpoint from: {rdm_checkpoint_path}")
        self._load_checkpoint(rdm_checkpoint_path)
        
    def _load_checkpoint(self, checkpoint_path):
        """Load RDM model from checkpoint.

        Expected checkpoint keys (produced by SEG-RDM main_rdm.py):
            config   – full OmegaConf dict with model.target / model.params
            model    – raw model state_dict
            model_ema – EMA state_dict (or None)
            args, optimizer, epoch, scaler – training metadata
        """
        # Add SEG-RDM to path
        seg_rdm_path = Path(__file__).parent.parent.parent.parent / 'SEG-RDM'
        if str(seg_rdm_path) not in sys.path:
            sys.path.insert(0, str(seg_rdm_path))

        try:
            from rdm.models.diffusion.ddim import DDIMSampler
            from rdm.util import instantiate_from_config
            from omegaconf import OmegaConf
        except ImportError as e:
            raise ImportError(
                f"Failed to import SEG-RDM modules. Make sure {seg_rdm_path} exists. Error: {e}"
            )

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # --- Retrieve config ------------------------------------------------
        config = ckpt.get('config')
        if config is None or 'model' not in config:
            raise RuntimeError(
                "Checkpoint is missing the 'config' key (or config has no 'model' entry). "
                "Re-train the RDM with the updated save_model() that embeds the YAML config."
            )
        
        # Convert plain dict back to OmegaConf for attribute access
        # (checkpoint saves as plain dict via OmegaConf.to_container())
        config = OmegaConf.create(config)

        # --- Instantiate model from config -----------------------------------
        # Remove pretrained_enc_config so the 630M I-JEPA encoder is NOT
        # loaded at inference time (we only need the diffusion backbone).
        model_cfg = config['model']
        if 'params' in model_cfg and 'pretrained_enc_config' in model_cfg['params']:
            model_cfg['params']['pretrained_enc_config'] = {'params': {}}
            print("[RDM] Skipping pretrained encoder init (not needed for sampling)")

        self.model = instantiate_from_config(model_cfg)

        # --- Load weights ----------------------------------------------------
        if self.use_ema and ckpt.get('model_ema') is not None:
            print("[RDM] Loading EMA weights")
            self.model.load_state_dict(ckpt['model_ema'], strict=False)
        elif 'model' in ckpt:
            print("[RDM] Loading raw model weights")
            self.model.load_state_dict(ckpt['model'], strict=False)
        else:
            raise RuntimeError("Checkpoint contains neither 'model' nor 'model_ema' state_dict.")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Create DDIM sampler for fast sampling
        self.ddim_sampler = DDIMSampler(self.model)

        epoch = ckpt.get('epoch', '?')
        print(f"[RDM] Model loaded successfully (epoch={epoch}, EMA={self.use_ema})")
    
    @torch.no_grad()
    def sample(self, batch_size=1, num_segments=180, cond=None, use_ddim=True):
        """
        Sample (Global, Segment) embeddings from RDM.
        
        Args:
            batch_size: Number of samples to generate
            num_segments: Number of segment tokens per sample
            cond: Optional conditioning (e.g., class labels)
            use_ddim: Use DDIM for faster sampling (default: True)
            
        Returns:
            dict with keys:
                'global_vec': [batch_size, 256] global I-JEPA embeddings
                'seg_tokens': [batch_size, num_segments, 256] SAM segment embeddings
                'num_segments': [batch_size] actual segment counts (all = num_segments)
        """
        # Shape for unified tokens: [B, C, 1, N+1]
        # C=256, N+1 includes 1 global + num_segments
        shape = (batch_size, 256, 1, num_segments + 1)
        
        if use_ddim and self.ddim_sampler is not None:
            # Fast DDIM sampling
            samples, _ = self.ddim_sampler.sample(
                S=self.ddim_steps,
                batch_size=batch_size,
                shape=shape[1:],  # Exclude batch dim
                conditioning=cond,
                verbose=False,
            )
        else:
            # Standard DDPM sampling (slower)
            samples = self.model.sample(
                cond=cond,
                batch_size=batch_size,
                num_segments=num_segments,
                return_intermediates=False,
            )
        
        # Convert from diffusion format: [B, C, 1, N+1] -> [B, N+1, C]
        tokens = samples.squeeze(2).permute(0, 2, 1)  # [B, N+1, 256]
        
        # Split global and segment tokens
        global_vectors = tokens[:, 0, :]  # [B, 256]
        seg_tokens = tokens[:, 1:, :]  # [B, num_segments, 256]
        
        # Create num_segments array (all valid, no padding)
        num_segments_arr = torch.full((batch_size,), num_segments, dtype=torch.long)
        
        return {
            'global_vec': global_vectors,
            'seg_tokens': seg_tokens,
            'num_segments': num_segments_arr,
        }
    
    @torch.no_grad()
    def sample_cached(self, cache_size=100, num_segments=180):
        """
        Pre-generate a cache of samples for faster training.
        
        Args:
            cache_size: Number of samples to cache
            num_segments: Segments per sample
            
        Returns:
            dict with cached samples
        """
        print(f"Pre-generating {cache_size} RDM samples...")
        
        all_global = []
        all_seg = []
        
        # Generate in batches
        batch_size = 16
        for i in range(0, cache_size, batch_size):
            current_batch = min(batch_size, cache_size - i)
            samples = self.sample(
                batch_size=current_batch,
                num_segments=num_segments,
                use_ddim=True
            )
            all_global.append(samples['global_vec'])
            all_seg.append(samples['seg_tokens'])
        
        # Concatenate
        cache = {
            'global_vec': torch.cat(all_global, dim=0),  # [cache_size, 256]
            'seg_tokens': torch.cat(all_seg, dim=0),  # [cache_size, num_segments, 256]
        }
        
        print(f"Cache generated: {cache_size} samples")
        return cache


class CachedRDMSampler:
    """Efficient RDM sampler with pre-generated cache and random access."""
    
    def __init__(self, rdm_sampler, cache_size=1000, num_segments=180):
        """
        Args:
            rdm_sampler: RDMSampler instance
            cache_size: Size of pre-generated cache
            num_segments: Segments per sample
        """
        self.rdm_sampler = rdm_sampler
        self.num_segments = num_segments
        self.cache = None
        self.cache_size = cache_size
        self.current_idx = 0
        
        # Pre-generate cache
        self._refresh_cache()
    
    def _refresh_cache(self):
        """Refresh the cache with new samples."""
        self.cache = self.rdm_sampler.sample_cached(
            cache_size=self.cache_size,
            num_segments=self.num_segments
        )
        self.current_idx = 0
    
    def sample(self, batch_size=1):
        """
        Sample from cache (with automatic refresh when exhausted).
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            dict with sampled embeddings
        """
        # Check if we need to refresh
        if self.current_idx + batch_size > self.cache_size:
            self._refresh_cache()
        
        # Get batch from cache
        global_batch = self.cache['global_vec'][
            self.current_idx:self.current_idx + batch_size
        ]
        seg_batch = self.cache['seg_tokens'][
            self.current_idx:self.current_idx + batch_size
        ]
        
        self.current_idx += batch_size
        
        return {
            'global_vec': global_batch,
            'seg_tokens': seg_batch,
            'num_segments': torch.full((batch_size,), self.num_segments, dtype=torch.long),
        }


# Example usage
if __name__ == '__main__':
    # Initialize sampler
    sampler = RDMSampler(
        rdm_checkpoint_path='path/to/rdm_checkpoint.pt',
        device='cuda',
        use_ema=True,
        ddim_steps=50
    )
    
    # Sample embeddings
    samples = sampler.sample(batch_size=4, num_segments=180)
    
    print("Global vectors:", samples['global_vec'].shape)  # [4, 256]
    print("Segment tokens:", samples['seg_tokens'].shape)  # [4, 180, 256]
    
    # Use cached sampler for efficient training
    cached_sampler = CachedRDMSampler(sampler, cache_size=500)
    batch = cached_sampler.sample(batch_size=8)
    print("Cached batch:", batch['global_vec'].shape)
