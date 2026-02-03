#!/usr/bin/env python3
"""
Inference script for Unified Segmentation RDM.
Generates both global vectors and segmentation tokens from trained model.
Outputs can be used with StyleGAN2 for image synthesis.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdm.models.diffusion.ddpm import UnifiedSegRDM
from rdm.util import instantiate_from_config


class UnifiedSegRDMSampler:
    """Sampler for generating unified tokens from trained RDM."""
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        use_ema: bool = True,
    ):
        """
        Args:
            config_path: Path to config YAML file
            checkpoint_path: Path to model checkpoint
            use_ema: Whether to use EMA weights
        """
        # Load config
        self.config = OmegaConf.load(config_path)
        print(f"Loaded config from: {config_path}")
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        print("\n=== Loading Model ===")
        self.model = self.build_model()
        self.load_checkpoint(checkpoint_path, use_ema=use_ema)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model ready for sampling")
    
    def build_model(self) -> UnifiedSegRDM:
        """Build model from config."""
        model_config = self.config.model
        model = instantiate_from_config(model_config)
        return model
    
    def load_checkpoint(self, path: str, use_ema: bool = True):
        """Load model checkpoint."""
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")
        
        # Use EMA if requested
        if use_ema and self.model.use_ema:
            print("Using EMA weights")
            self.model.model_ema.copy_to(self.model)
        
        step = checkpoint.get('global_step', 'unknown')
        print(f"Loaded checkpoint from step: {step}")
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 16,
        num_segments: int = 180,
        batch_size: int = 8,
        ddpm_steps: int = 1000,
        verbose: bool = True,
    ) -> dict:
        """
        Generate unified token sequences.
        
        Args:
            num_samples: Total number of samples to generate
            num_segments: Number of segment tokens per sample
            batch_size: Batch size for generation
            ddpm_steps: Number of DDPM denoising steps
            verbose: Show progress bar
            
        Returns:
            dict with:
                - global_vectors: [N, 256] global token vectors
                - seg_tokens: [N, num_segments, 256] segmentation token sequences
        """
        all_global = []
        all_segments = []
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        pbar = tqdm(range(num_batches), desc="Sampling", disable=not verbose)
        
        for i in pbar:
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Sample from model
            # Shape: [B, 256, 1, num_segments+1]
            samples = self.model.sample(
                cond=None,
                batch_size=current_batch_size,
                shape=None,
                num_segments=num_segments,
                verbose=False,
                timesteps=ddpm_steps,
            )
            
            # Convert to [B, num_segments+1, 256]
            samples = self.model._from_diffusion_format(samples)
            
            # Split global and segments
            global_vec = samples[:, 0, :]  # [B, 256]
            seg_tokens = samples[:, 1:, :]  # [B, num_segments, 256]
            
            all_global.append(global_vec.cpu())
            all_segments.append(seg_tokens.cpu())
        
        pbar.close()
        
        # Concatenate all batches
        global_vectors = torch.cat(all_global, dim=0)  # [N, 256]
        seg_tokens = torch.cat(all_segments, dim=0)  # [N, num_segments, 256]
        
        return {
            'global_vectors': global_vectors,
            'seg_tokens': seg_tokens,
        }
    
    def save_samples(
        self,
        samples: dict,
        output_dir: str,
        prefix: str = "sample",
    ):
        """
        Save generated samples to disk.
        
        Args:
            samples: Dict from sample() containing global_vectors and seg_tokens
            output_dir: Directory to save samples
            prefix: Filename prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy arrays
        np.save(
            output_dir / f"{prefix}_global_vectors.npy",
            samples['global_vectors'].numpy()
        )
        np.save(
            output_dir / f"{prefix}_seg_tokens.npy",
            samples['seg_tokens'].numpy()
        )
        
        # Also save as PyTorch tensors for easier loading
        torch.save(
            samples,
            output_dir / f"{prefix}_tensors.pt"
        )
        
        print(f"Saved samples to: {output_dir}")
        print(f"  global_vectors: {samples['global_vectors'].shape}")
        print(f"  seg_tokens: {samples['seg_tokens'].shape}")


def main():
    parser = argparse.ArgumentParser(description="Sample from Unified Segmentation RDM")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='samples/unified_seg_rdm',
        help='Output directory for samples'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=16,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--num_segments',
        type=int,
        default=180,
        help='Number of segment tokens per sample'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for generation'
    )
    parser.add_argument(
        '--ddpm_steps',
        type=int,
        default=1000,
        help='Number of DDPM denoising steps'
    )
    parser.add_argument(
        '--use_ema',
        action='store_true',
        default=True,
        help='Use EMA weights'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='sample',
        help='Filename prefix for saved samples'
    )
    args = parser.parse_args()
    
    # Create sampler
    sampler = UnifiedSegRDMSampler(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        use_ema=args.use_ema,
    )
    
    # Generate samples
    print(f"\n=== Generating {args.num_samples} samples ===")
    samples = sampler.sample(
        num_samples=args.num_samples,
        num_segments=args.num_segments,
        batch_size=args.batch_size,
        ddpm_steps=args.ddpm_steps,
        verbose=True,
    )
    
    # Save samples
    sampler.save_samples(
        samples=samples,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
