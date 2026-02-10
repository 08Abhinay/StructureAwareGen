"""
Aligned Segmentation Dataset for StyleGAN2.
Loads triplets: (image, I-JEPA global vector, SAM segment tokens).

Supports both conditional and unconditional training via use_labels flag.
"""

import os
import json
import numpy as np
from pathlib import Path
from .dataset import ImageFolderDataset


class AlignedSegDataset(ImageFolderDataset):
    """
    Dataset that loads aligned triplets of (image, global_vec, seg_tokens).
    
    Args:
        path: Path to image directory/zip
        sam_npz_dir: Path to SAM embeddings directory
        ijepa_npz_dir: Path to I-JEPA embeddings directory
        origin_map_json: Path to origin_map.json (maps zip filenames → original stems)
        max_segments: Maximum number of segments (for padding)
        use_labels: If True, use class labels (conditional). If False, all labels=0 (unconditional)
        **super_kwargs: Additional arguments for ImageFolderDataset base class
    """
    
    def __init__(
        self,
        path,
        sam_npz_dir=None,
        ijepa_npz_dir=None,
        origin_map_json=None,
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
        
        # --- Load origin map (zip filename → original class/stem) ---
        self._origin_map = {}
        if origin_map_json is not None:
            om_path = Path(origin_map_json)
            if om_path.exists():
                with open(om_path, 'r') as f:
                    self._origin_map = json.load(f)
            else:
                print(f"  WARNING: origin_map_json not found at {om_path}")
                print(f"           NPZ lookup will use zip filenames directly (likely won't find files)")
        
        if not self.sam_npz_dir.exists():
            raise ValueError(f"SAM directory not found: {sam_npz_dir}")
        if not self.ijepa_npz_dir.exists():
            raise ValueError(f"I-JEPA directory not found: {ijepa_npz_dir}")
        
        # Only print verbose init info once (first construction, typically rank 0)
        if not hasattr(AlignedSegDataset, '_init_logged'):
            AlignedSegDataset._init_logged = True
            if self._origin_map:
                print(f"  Loaded origin_map with {len(self._origin_map)} entries from {origin_map_json}")
                # Sanity check: verify first filename maps correctly
                if len(self._image_fnames) > 0:
                    test_fname = self._image_fnames[0]
                    if test_fname in self._origin_map:
                        orig = self._origin_map[test_fname]
                        print(f"  origin_map sanity check: '{test_fname}' -> '{orig}' OK")
                    else:
                        print(f"  WARNING: first image '{test_fname}' NOT in origin_map — check mapping!")
            elif origin_map_json is None:
                print(f"  INFO: No origin_map_json provided — assuming image filenames match NPZ filenames")
            print(f"AlignedSegDataset initialized:")
            print(f"  Images: {path}")
            print(f"  SAM embeddings: {sam_npz_dir}")
            print(f"  I-JEPA embeddings: {ijepa_npz_dir}")
            print(f"  Origin map entries: {len(self._origin_map)}")
            print(f"  Max segments: {max_segments}")
            print(f"  Use labels (conditional): {use_labels}")
    
    def _get_corresponding_npz(self, image_fname, npz_dir, subdir=None):
        """
        Get corresponding .npz path for an image filename.
        
        If origin_map is loaded, translates zip names like
          "00000/img00000005.png" -> original "0/980"
          -> npz_dir/0/{subdir}/980.npz  (if subdir given, e.g. "masks_npz" for SAM)
          -> npz_dir/0/980.npz           (if subdir is None, e.g. for I-JEPA)
        
        Falls back to using the zip filename directly if no mapping exists.
        """
        if self._origin_map and image_fname in self._origin_map:
            orig_key = self._origin_map[image_fname]   # e.g. "0/980"
            parts = orig_key.split("/")
            if len(parts) == 2:
                if subdir:
                    return npz_dir / parts[0] / subdir / f"{parts[1]}.npz"
                return npz_dir / parts[0] / f"{parts[1]}.npz"
            return npz_dir / f"{orig_key}.npz"
        
        # Fallback: use zip filename directly (works for non-zip or unmapped datasets)
        rel_path = Path(image_fname)
        if subdir:
            return npz_dir / rel_path.parent / subdir / f"{rel_path.stem}.npz"
        return npz_dir / rel_path.parent / f"{rel_path.stem}.npz"
    
    def __getitem__(self, idx):
        """
        Returns dict with:
            - image: [C, H, W] image tensor
            - label: [num_classes] one-hot label (or zeros if unconditional)
            - global_vec: [1280] I-JEPA global vector (ViT-H/14)
            - seg_tokens: [max_segments, 256] SAM segment tokens (padded)
            - seg_pad_mask: [max_segments] boolean mask (True = padding)
            - num_segments: scalar, actual number of segments
        """
        # Parent returns dict (with 'image', 'label', 'paths') or tuple
        result = super().__getitem__(idx)
        if isinstance(result, dict):
            image = result['image']
            label = result['label']
        else:
            image, label = result
        
        fname = self._image_fnames[self._raw_idx[idx]]
        
        try:
            ijepa_path = self._get_corresponding_npz(fname, self.ijepa_npz_dir)
            ijepa_data = np.load(ijepa_path)
            global_vec = ijepa_data['emb'].astype(np.float32)
            
            if global_vec.ndim > 1:
                global_vec = global_vec.squeeze()
            assert global_vec.shape == (1280,), f"I-JEPA embedding shape mismatch: {global_vec.shape}"
            
        except FileNotFoundError:
            print(f"Warning: I-JEPA embedding not found for {fname} (tried {ijepa_path}), using zeros")
            global_vec = np.zeros(1280, dtype=np.float32)
        except Exception as e:
            print(f"Error loading I-JEPA for {fname}: {e}, using zeros")
            global_vec = np.zeros(1280, dtype=np.float32)
        
        try:
            sam_path = self._get_corresponding_npz(fname, self.sam_npz_dir, subdir="masks_npz")
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
            print(f"Warning: SAM embedding not found for {fname} (tried {sam_path}), using zeros")
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
        
        # Get image path for SAM extraction (if needed)
        image_path = super().get_path(idx)
        
        return {
            'image': image,
            'label': label,
            'global_vec': global_vec,
            'seg_tokens': seg_tokens,
            'seg_pad_mask': seg_pad_mask,
            'num_segments': num_segments,
            'paths': image_path
        }
