"""
Aligned Segmentation Dataset for StyleGAN2.
Loads triplets: (image, I-JEPA global vector, SAM segment tokens).

Supports both conditional and unconditional training via use_labels flag.
"""

import os
import numpy as np
from pathlib import Path
from .dataset import ImageFolderDataset


class AlignedSegDataset(ImageFolderDataset):
    """
    Dataset that loads aligned triplets of (image, global_vec, seg_tokens).
    
    Args:
        path: Path to image directory
        sam_npz_dir: Path to SAM embeddings directory
        ijepa_npz_dir: Path to I-JEPA embeddings directory  
        max_segments: Maximum number of segments (for padding)
        use_labels: If True, use class labels (conditional). If False, all labels=0 (unconditional)
        **super_kwargs: Additional arguments for ImageFolderDataset base class
    """
    
    def __init__(
        self,
        path,
        sam_npz_dir=None,
        ijepa_npz_dir=None,
        max_segments=250,
        use_labels=False,
        **super_kwargs
    ):
        if sam_npz_dir is None:
            raise ValueError("sam_npz_dir must be specified")
        if ijepa_npz_dir is None:
            raise ValueError("ijepa_npz_dir must be specified")
        
        super().__init__(path=path, use_labels=use_labels, **super_kwargs)
        
        self.sam_npz_dir = Path(sam_npz_dir)
        self.ijepa_npz_dir = Path(ijepa_npz_dir)
        self.max_segments = max_segments
        self.image_dir = Path(path)
        
        if not self.sam_npz_dir.exists():
            raise ValueError(f"SAM directory not found: {sam_npz_dir}")
        if not self.ijepa_npz_dir.exists():
            raise ValueError(f"I-JEPA directory not found: {ijepa_npz_dir}")
        
        print(f"AlignedSegDataset initialized:")
        print(f"  Images: {path}")
        print(f"  SAM embeddings: {sam_npz_dir}")
        print(f"  I-JEPA embeddings: {ijepa_npz_dir}")
        print(f"  Max segments: {max_segments}")
        print(f"  Use labels (conditional): {use_labels}")
    
    def _get_corresponding_npz(self, image_fname, npz_dir):
        """Get corresponding .npz path for an image filename."""
        rel_path = Path(image_fname)
        npz_path = npz_dir / rel_path.parent / f"{rel_path.stem}.npz"
        return npz_path
    
    def __getitem__(self, idx):
        """
        Returns dict with:
            - image: [C, H, W] image tensor
            - label: [num_classes] one-hot label (or zeros if unconditional)
            - global_vec: [256] I-JEPA global vector
            - seg_tokens: [max_segments, 256] SAM segment tokens (padded)
            - seg_pad_mask: [max_segments] boolean mask (True = padding)
            - num_segments: scalar, actual number of segments
        """
        image, label = super().__getitem__(idx)
        
        fname = self._image_fnames[self._raw_idx[idx]]
        
        try:
            ijepa_path = self._get_corresponding_npz(fname, self.ijepa_npz_dir)
            ijepa_data = np.load(ijepa_path)
            global_vec = ijepa_data['emb'].astype(np.float32)
            
            assert global_vec.shape == (256,), f"I-JEPA embedding shape mismatch: {global_vec.shape}"
            
        except FileNotFoundError:
            print(f"Warning: I-JEPA embedding not found for {fname}, using zeros")
            global_vec = np.zeros(256, dtype=np.float32)
        except Exception as e:
            print(f"Error loading I-JEPA for {fname}: {e}, using zeros")
            global_vec = np.zeros(256, dtype=np.float32)
        
        try:
            sam_path = self._get_corresponding_npz(fname, self.sam_npz_dir)
            sam_data = np.load(sam_path)
            seg_tokens = sam_data['emb'].astype(np.float32)
            
            assert seg_tokens.ndim == 2 and seg_tokens.shape[1] == 256, \
                f"SAM embedding shape mismatch: {seg_tokens.shape}"
            
            num_segments = len(seg_tokens)
            
            if num_segments > self.max_segments:
                seg_tokens = seg_tokens[:self.max_segments]
                seg_pad_mask = np.zeros(self.max_segments, dtype=np.bool_)
                num_segments = self.max_segments
            else:
                pad_len = self.max_segments - num_segments
                seg_tokens = np.vstack([
                    seg_tokens,
                    np.zeros((pad_len, 256), dtype=np.float32)
                ])
                seg_pad_mask = np.concatenate([
                    np.zeros(num_segments, dtype=np.bool_),
                    np.ones(pad_len, dtype=np.bool_)
                ])
            
        except FileNotFoundError:
            print(f"Warning: SAM embedding not found for {fname}, using zeros")
            seg_tokens = np.zeros((self.max_segments, 256), dtype=np.float32)
            seg_pad_mask = np.ones(self.max_segments, dtype=np.bool_)
            num_segments = 0
        except Exception as e:
            print(f"Error loading SAM for {fname}: {e}, using zeros")
            seg_tokens = np.zeros((self.max_segments, 256), dtype=np.float32)
            seg_pad_mask = np.ones(self.max_segments, dtype=np.bool_)
            num_segments = 0
        
        if not self._use_labels:
            label = np.zeros_like(label)
        
        return {
            'image': image,
            'label': label,
            'global_vec': global_vec,
            'seg_tokens': seg_tokens,
            'seg_pad_mask': seg_pad_mask,
            'num_segments': num_segments
        }
