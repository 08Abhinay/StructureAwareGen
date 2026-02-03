"""
SAM (Segment Anything Model) Extractor with Caching and Stochastic Conditioning

This module provides on-the-fly SAM extraction during StyleGAN2 training with:
- Lazy SAM initialization (only loads when needed)
- Cache-based system to avoid re-extracting images
- Validation of cached embeddings
- Async disk writes for non-blocking I/O
- Stochastic conditioning (dropout-like SAM usage)

Design:
- Cache key: Image file path (deterministic, not seed-dependent)
- Cache format: {class_folder}/{image_stem}.npz
- Cache contents: {'emb': (N, 256) float16, 'scores': (N,) float32}
- Coverage: Probabilistic extraction ensures eventual full coverage (0.75^N decay)

Usage:
    extractor = SAMExtractor(
        sam_checkpoint="sam_vit_b_01ec64.pth",
        cache_dir="./sam_cache",
        device="cuda",
        model_type="vit_b"
    )
    
    # Extract or load from cache
    embeddings = extractor.extract_or_load(image_paths, images)
    
    # Pad to batch format
    padded_emb, pad_mask = pad_embeddings_batch(embeddings)
"""

import os
import numpy as np
import torch
import torch.distributed as dist
import threading
import queue
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import PIL.Image
try:
    from filelock import FileLock
except ImportError:
    FileLock = None
    print("[SAMExtractor] Warning: filelock not installed. Install with: pip install filelock")
    print("[SAMExtractor] Falling back to non-locking mode (may cause issues with multi-GPU)")
try:
    from filelock import FileLock
except ImportError:
    FileLock = None
    print("[SAMExtractor] Warning: filelock not installed. Install with: pip install filelock")
    print("[SAMExtractor] Falling back to non-locking mode (may cause issues with multi-GPU)")


#----------------------------------------------------------------------------
# Distributed training utilities
#----------------------------------------------------------------------------

def is_distributed() -> bool:
    """Check if running in distributed mode"""
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    """Get current process rank (0 if not distributed)"""
    if is_distributed():
        return dist.get_rank()
    return 0

def get_world_size() -> int:
    """Get total number of processes (1 if not distributed)"""
    if is_distributed():
        return dist.get_world_size()
    return 1

def barrier():
    """Synchronize all processes (no-op if not distributed)"""
    if is_distributed():
        dist.barrier()

#----------------------------------------------------------------------------

class AsyncWriter:
    """Background thread for non-blocking file writes"""
    
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        """Worker thread that processes write requests"""
        while True:
            item = self.queue.get()
            if item is None:  # Sentinel to stop thread
                break
            
            filepath, data = item
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Use file locking for multi-GPU safety
                if FileLock is not None:
                    lock_path = filepath + '.lock'
                    with FileLock(lock_path, timeout=300):  # 5 min timeout
                        # Double-check if another process wrote it while we waited
                        if not os.path.exists(filepath):
                            np.savez_compressed(filepath, **data)
                    # Clean up lock file
                    try:
                        os.remove(lock_path)
                    except:
                        pass
                else:
                    # Fallback without locking (single-GPU or filelock not installed)
                    np.savez_compressed(filepath, **data)
            except Exception as e:
                print(f"[AsyncWriter] Failed to save {filepath}: {e}")
            finally:
                self.queue.task_done()
    
    def save(self, filepath: str, data: dict):
        """Queue a save operation"""
        self.queue.put((filepath, data))
    
    def shutdown(self):
        """Wait for all pending writes and stop thread"""
        self.queue.join()
        self.queue.put(None)
        self.thread.join()


class SAMExtractor:
    """
    SAM extractor with lazy initialization, caching, and validation.
    
    Lazy Init: SAM model only loaded when first extraction is needed
    Cache: Image path → .npz file mapping (check cache first)
    Validation: File size >1KB, required keys present, valid shapes
    Async Writes: Non-blocking disk I/O via background thread
    """
    
    def __init__(
        self,
        sam_checkpoint: str,
        cache_dir: str,
        device: str = "cuda",
        model_type: str = "vit_b",
        max_masks: int = 250,
        rank: Optional[int] = None,
        world_size: Optional[int] = None
    ):
        """
        Initialize SAM extractor.
        
        Args:
            sam_checkpoint: Path to SAM checkpoint file
            cache_dir: Directory to store cached embeddings
            device: Device to run SAM on ("cuda" or "cpu")
            model_type: SAM model type ("vit_b", "vit_l", "vit_h")
            max_masks: Maximum number of mask embeddings to keep per image
            rank: Process rank for distributed training (None = auto-detect)
            world_size: Total number of processes (None = auto-detect)
        """
        self.sam_checkpoint = sam_checkpoint
        self.cache_dir = cache_dir
        self.device = device
        self.model_type = model_type
        self.max_masks = max_masks
        
        # Distributed training support
        self.rank = rank if rank is not None else get_rank()
        self.world_size = world_size if world_size is not None else get_world_size()
        
        # Lazy initialization - SAM not loaded until needed
        self.sam = None
        self.mask_generator = None
        
        # Async writer for non-blocking saves
        self.async_writer = AsyncWriter()
        
        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.extractions = 0
    
    def _lazy_init_sam(self):
        """Lazy initialization of SAM model (only when first needed)"""
        if self.sam is not None:
            return
        
        print(f"[SAMExtractor] Loading SAM model from {self.sam_checkpoint}")
        
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        except ImportError:
            raise ImportError(
                "segment_anything not installed. Install with: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        
        # Load SAM model
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.sam.eval()
        
        # Create mask generator and predictor
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.predictor = SamPredictor(self.sam)  # For extracting image embeddings
        
        print(f"[SAMExtractor] SAM model loaded successfully")
    
    def _get_rank_subset(self, image_paths: List[str]) -> List[str]:
        """
        Get subset of images for this rank to process.
        Uses strided indexing for load balancing: rank 0 gets [0,4,8,...], rank 1 gets [1,5,9,...]
        
        Args:
            image_paths: Full list of image paths
            
        Returns:
            Subset for this rank
        """
        if self.world_size == 1:
            return image_paths
        
        # Strided indexing: [rank::world_size]
        return image_paths[self.rank::self.world_size]
    
    def _get_cache_path(self, image_path: str) -> str:
        """
        Get cache file path for an image.
        
        Cache structure: {cache_dir}/{parent_folder}/{image_stem}.npz
        Example: /sam_cache/n01440764/n01440764_10026.npz
        
        Args:
            image_path: Full path to image file
            
        Returns:
            Path to cache file
        """
        path_obj = Path(image_path)
        parent_folder = path_obj.parent.name
        image_stem = path_obj.stem
        cache_path = os.path.join(self.cache_dir, parent_folder, f"{image_stem}.npz")
        return cache_path
    
    def _validate_cache(self, cache_path: str) -> bool:
        """
        Validate cached embeddings.
        
        Checks:
        - File exists and size > 1KB
        - Has required keys: 'emb', 'scores'
        - Valid shapes
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            True if valid, False otherwise
        """
        if not os.path.exists(cache_path):
            return False
        
        # Check file size (should be at least 1KB)
        if os.path.getsize(cache_path) < 1024:
            return False
        
        try:
            data = np.load(cache_path)
            
            # Check required keys
            if 'emb' not in data or 'scores' not in data:
                return False
            
            emb = data['emb']
            scores = data['scores']
            
            # Check shapes
            if emb.ndim != 2 or emb.shape[1] != 256:
                return False
            
            if scores.ndim != 1 or len(scores) != emb.shape[0]:
                return False
            
            return True
            
        except Exception as e:
            print(f"[SAMExtractor] Cache validation failed for {cache_path}: {e}")
            return False
    
    def _load_from_cache(self, cache_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load embeddings from cache.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            Dict with 'emb' and 'scores' if valid, None otherwise
        """
        if not self._validate_cache(cache_path):
            return None
        
        try:
            data = np.load(cache_path)
            self.cache_hits += 1
            return {
                'emb': data['emb'],
                'scores': data['scores']
            }
        except Exception as e:
            print(f"[SAMExtractor] Failed to load cache {cache_path}: {e}")
            self.cache_misses += 1
            return None
    
    def _extract_single(self, image_path: str, image_tensor: Optional[torch.Tensor] = None) -> Dict[str, np.ndarray]:
        """
        Extract SAM embeddings for a single image.
        Uses production method from segmentation-play.ipynb.
        
        Args:
            image_path: Path to image file
            image_tensor: Optional pre-loaded image tensor (B, C, H, W) in [-1, 1]
            
        Returns:
            Dict with 'emb' (N, 256) and 'scores' (N,)
        """
        import torch.nn.functional as F
        
        # Lazy init SAM if not already loaded
        self._lazy_init_sam()
        
        # Load image
        if image_tensor is not None:
            # Convert from tensor [-1, 1] to numpy [0, 255]
            img_np = ((image_tensor[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            if img_np.shape[2] == 1:  # Grayscale
                img_np = np.repeat(img_np, 3, axis=2)
        else:
            # Load from disk
            image = PIL.Image.open(image_path).convert('RGB')
            img_np = np.array(image, dtype=np.uint8)
        
        # Generate masks using AMG
        masks = self.mask_generator.generate(img_np)
        
        # Get image embedding once (for all masks)
        self.predictor.set_image(img_np)
        feat = self.predictor.get_image_embedding()  # (1, C, H_feat, W_feat)
        hf, wf = feat.shape[-2], feat.shape[-1]
        
        # Extract embeddings and scores for each mask
        embeddings = []
        scores = []
        
        for mask in masks:
            seg = mask['segmentation'].astype(bool)
            
            # Compute score (same as production notebook)
            pred_iou = float(mask.get('predicted_iou', 0.0))
            stability_score = float(mask.get('stability_score', 0.0))
            score = pred_iou * stability_score
            
            # Extract per-mask embedding from SAM image encoder
            # This is the production method from segmentation-play.ipynb
            mask_t = torch.from_numpy(seg[None, None].astype(np.float32)).to(self.device)
            mask_small = F.interpolate(mask_t, size=(hf, wf), mode="nearest")
            denom = mask_small.sum(dim=(2, 3)) + 1e-6
            emb_t = (feat * mask_small).sum(dim=(2, 3)) / denom  # (1, C)
            emb = emb_t.squeeze(0).detach().cpu().to(torch.float16).numpy()  # (C,)
            
            embeddings.append(emb)
            scores.append(score)
        
        # Convert to numpy arrays
        if len(embeddings) == 0:
            # No masks found - create dummy embedding
            embeddings = np.zeros((1, 256), dtype=np.float16)
            scores = np.zeros((1,), dtype=np.float32)
        else:
            embeddings = np.stack(embeddings, axis=0)  # (N, 256)
            scores = np.array(scores, dtype=np.float32)
            
            # Sort by score (descending) and keep top max_masks
            if len(scores) > self.max_masks:
                top_indices = np.argsort(scores)[::-1][:self.max_masks]
                embeddings = embeddings[top_indices]
                scores = scores[top_indices]
        
        self.extractions += 1
        
        return {
            'emb': embeddings,
            'scores': scores
        }
    
    def extract_or_load(
        self,
        image_paths: List[str],
        image_tensors: Optional[torch.Tensor] = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Extract or load SAM embeddings for a batch of images.
        
        Checks cache first, extracts if not found, saves to cache asynchronously.
        
        Args:
            image_paths: List of image file paths
            image_tensors: Optional pre-loaded image tensors (B, C, H, W) in [-1, 1]
            
        Returns:
            List of dicts with 'emb' (N, 256) and 'scores' (N,) for each image
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            cache_path = self._get_cache_path(image_path)
            
            # Try loading from cache
            cached = self._load_from_cache(cache_path)
            
            if cached is not None:
                results.append(cached)
            else:
                # Use file locking to prevent concurrent extraction by multiple GPUs
                if FileLock is not None:
                    lock_path = cache_path + '.lock'
                    try:
                        with FileLock(lock_path, timeout=300):  # 5 min timeout
                            # Double-check cache after acquiring lock (another GPU may have written it)
                            cached = self._load_from_cache(cache_path)
                            if cached is not None:
                                results.append(cached)
                            else:
                                # Extract from image
                                img_tensor = image_tensors[i:i+1] if image_tensors is not None else None
                                extracted = self._extract_single(image_path, img_tensor)
                                results.append(extracted)
                                
                                # Save to cache asynchronously (will use locking internally)
                                self.async_writer.save(cache_path, extracted)
                        # Clean up lock file
                        try:
                            os.remove(lock_path)
                        except:
                            pass
                    except Exception as e:
                        print(f"[SAMExtractor] Lock timeout or error for {image_path}: {e}")
                        # Fallback: extract without saving to cache
                        img_tensor = image_tensors[i:i+1] if image_tensors is not None else None
                        extracted = self._extract_single(image_path, img_tensor)
                        results.append(extracted)
                else:
                    # No locking available - extract and save
                    img_tensor = image_tensors[i:i+1] if image_tensors is not None else None
                    extracted = self._extract_single(image_path, img_tensor)
                    results.append(extracted)
                    self.async_writer.save(cache_path, extracted)
        
        return results
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'extractions': self.extractions,
            'hit_rate': hit_rate
        }
    
    def __del__(self):
        """Cleanup async writer on deletion"""
        if hasattr(self, 'async_writer'):
            self.async_writer.shutdown()


def pad_embeddings_batch(
    embeddings_list: List[Dict[str, np.ndarray]],
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length embeddings to batch format.
    
    Args:
        embeddings_list: List of dicts with 'emb' (N_i, 256) and 'scores' (N_i,)
        device: Device to put tensors on
        
    Returns:
        Tuple of:
        - padded_emb: (B, max_N, 256) tensor
        - pad_mask: (B, max_N) bool tensor (True = padding, False = valid)
    """
    batch_size = len(embeddings_list)
    
    # Find max number of masks in batch
    max_masks = max(len(item['scores']) for item in embeddings_list)
    
    # Create padded tensors
    padded_emb = torch.zeros(batch_size, max_masks, 256, dtype=torch.float32, device=device)
    pad_mask = torch.ones(batch_size, max_masks, dtype=torch.bool, device=device)  # True = padding
    
    # Fill in actual embeddings
    for i, item in enumerate(embeddings_list):
        n_masks = len(item['scores'])
        if n_masks > 0:
            # Convert to float32 for training
            emb_tensor = torch.from_numpy(item['emb'].astype(np.float32))
            padded_emb[i, :n_masks] = emb_tensor.to(device)
            pad_mask[i, :n_masks] = False  # Mark as valid (not padding)
    
    return padded_emb, pad_mask
