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

try:
    import h5py
except ImportError:
    h5py = None


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
        sam_h5_path=None,
        ijepa_h5_path=None,
        ijepa_lookup_json=None,
        origin_map_json=None,
        max_segments=250,
        use_labels=False,
        **super_kwargs
    ):
        if sam_npz_dir is None and sam_h5_path is None:
            raise ValueError("At least one SAM source must be specified: sam_npz_dir or sam_h5_path")
        if ijepa_npz_dir is None and ijepa_h5_path is None:
            raise ValueError("At least one I-JEPA source must be specified: ijepa_npz_dir or ijepa_h5_path")
        if (sam_h5_path is not None or ijepa_h5_path is not None) and h5py is None:
            raise ValueError("h5py is required when using H5 sources. Install with: pip install h5py")
        if ijepa_h5_path is not None and ijepa_lookup_json is None:
            raise ValueError("ijepa_lookup_json is required when ijepa_h5_path is specified")

        super().__init__(path=path, use_labels=use_labels, **super_kwargs)

        self.sam_npz_dir = Path(sam_npz_dir) if sam_npz_dir is not None else None
        self.ijepa_npz_dir = Path(ijepa_npz_dir) if ijepa_npz_dir is not None else None
        self.sam_h5_path = Path(sam_h5_path) if sam_h5_path is not None else None
        self.ijepa_h5_path = Path(ijepa_h5_path) if ijepa_h5_path is not None else None
        self.ijepa_lookup_json = Path(ijepa_lookup_json) if ijepa_lookup_json is not None else None
        self.max_segments = max_segments
        self.image_dir = Path(path)
        self._sam_h5_file = None
        self._ijepa_h5_file = None
        self._sam_flat_index = {}
        self._ijepa_lookup = {}
        self._warn_counts = {}

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

        if self.sam_npz_dir is not None and not self.sam_npz_dir.exists():
            raise ValueError(f"SAM NPZ directory not found: {sam_npz_dir}")
        if self.ijepa_npz_dir is not None and not self.ijepa_npz_dir.exists():
            raise ValueError(f"I-JEPA NPZ directory not found: {ijepa_npz_dir}")
        if self.sam_h5_path is not None and not self.sam_h5_path.exists():
            raise ValueError(f"SAM H5 file not found: {sam_h5_path}")
        if self.ijepa_h5_path is not None and not self.ijepa_h5_path.exists():
            raise ValueError(f"I-JEPA H5 file not found: {ijepa_h5_path}")
        if self.ijepa_lookup_json is not None and not self.ijepa_lookup_json.exists():
            raise ValueError(f"I-JEPA lookup JSON not found: {ijepa_lookup_json}")

        # Load I-JEPA flat lookup: "class/name" -> row index.
        if self.ijepa_lookup_json is not None:
            with open(self.ijepa_lookup_json, 'r') as f:
                self._ijepa_lookup = json.load(f)

        # Build SAM flat index from class_ids and names in H5.
        if self.sam_h5_path is not None:
            with h5py.File(self.sam_h5_path, 'r') as f:
                required_keys = ['class_ids', 'names', 'offsets', 'n_segments', 'emb']
                missing = [k for k in required_keys if k not in f]
                if missing:
                    raise ValueError(f"SAM H5 missing required datasets: {missing}")
                class_ids = f['class_ids'][:]
                names = f['names'][:]
            for i, (cid, name) in enumerate(zip(class_ids, names)):
                name_str = name.decode() if isinstance(name, (bytes, np.bytes_)) else str(name)
                self._sam_flat_index[(str(cid), name_str)] = int(i)

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
            print(f"  SAM source: {'H5' if self.sam_h5_path is not None else 'NPZ'}")
            if self.sam_h5_path is not None:
                print(f"    SAM H5 path: {self.sam_h5_path}")
                print(f"    SAM H5 index entries: {len(self._sam_flat_index)}")
            if self.sam_npz_dir is not None:
                print(f"    SAM NPZ dir (fallback): {self.sam_npz_dir}")
            print(f"  I-JEPA source: {'H5' if self.ijepa_h5_path is not None else 'NPZ'}")
            if self.ijepa_h5_path is not None:
                print(f"    I-JEPA H5 path: {self.ijepa_h5_path}")
                print(f"    I-JEPA lookup entries: {len(self._ijepa_lookup)}")
            if self.ijepa_npz_dir is not None:
                print(f"    I-JEPA NPZ dir (fallback): {self.ijepa_npz_dir}")
            print(f"  Origin map entries: {len(self._origin_map)}")
            print(f"  Max segments: {max_segments}")
            print(f"  Use labels (conditional): {use_labels}")

    def close(self):
        super().close()
        if self._sam_h5_file is not None:
            try:
                self._sam_h5_file.close()
            finally:
                self._sam_h5_file = None
        if self._ijepa_h5_file is not None:
            try:
                self._ijepa_h5_file.close()
            finally:
                self._ijepa_h5_file = None

    def __getstate__(self):
        return dict(super().__getstate__(), _sam_h5_file=None, _ijepa_h5_file=None)

    def _warn(self, key, message, max_count=20):
        count = self._warn_counts.get(key, 0)
        if count < max_count:
            suffix = " (further warnings for this key will be suppressed)" if count + 1 == max_count else ""
            print(f"{message}{suffix}")
        self._warn_counts[key] = count + 1

    @staticmethod
    def _decode_name(value):
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode()
        return str(value)

    def _resolve_sample_key(self, image_fname):
        """
        Resolve canonical class/name key for H5 and lookup JSON access.
        Returns (class_id, name, canonical_key='class_id/name').
        """
        if self._origin_map and image_fname in self._origin_map:
            orig_key = self._origin_map[image_fname]
        else:
            rel = Path(image_fname)
            if str(rel.parent) in ('', '.'):
                orig_key = rel.stem
            else:
                orig_key = f"{rel.parent.as_posix()}/{rel.stem}"

        orig_key = str(orig_key).replace('\\', '/').lstrip('./')
        p = Path(orig_key)
        class_id = p.parent.name if str(p.parent) not in ('', '.') else ''
        name = p.name

        if class_id == '':
            # Fallback for paths without class prefix.
            rel = Path(image_fname)
            class_id = rel.parent.name if str(rel.parent) not in ('', '.') else ''
            name = rel.stem

        canonical = f"{class_id}/{name}" if class_id else name
        return class_id, name, canonical

    def _get_sam_h5(self):
        if self._sam_h5_file is None:
            self._sam_h5_file = h5py.File(self.sam_h5_path, 'r')
        return self._sam_h5_file

    def _get_ijepa_h5(self):
        if self._ijepa_h5_file is None:
            self._ijepa_h5_file = h5py.File(self.ijepa_h5_path, 'r')
        return self._ijepa_h5_file

    def _load_sam_from_h5(self, class_id, name):
        idx = self._sam_flat_index.get((class_id, name))
        if idx is None:
            raise KeyError(f"SAM H5 key not found: {class_id}/{name}")
        h5f = self._get_sam_h5()
        off = int(h5f['offsets'][idx])
        nseg = int(h5f['n_segments'][idx])
        seg_tokens = np.asarray(h5f['emb'][off:off + nseg], dtype=np.float32)
        return seg_tokens

    def _load_ijepa_from_h5(self, class_id, name, canonical_key):
        key = canonical_key if canonical_key in self._ijepa_lookup else f"{class_id}/{name}"
        row = self._ijepa_lookup.get(key)
        if row is None:
            raise KeyError(f"I-JEPA lookup key not found: {key}")
        h5f = self._get_ijepa_h5()
        global_vec = np.asarray(h5f['emb'][int(row)], dtype=np.float32)
        return global_vec

    def _pad_or_truncate_seg_tokens(self, seg_tokens):
        seg_tokens = np.asarray(seg_tokens, dtype=np.float32)
        if seg_tokens.ndim == 1 and seg_tokens.size == 0:
            seg_tokens = seg_tokens.reshape(0, 256)
        if not (seg_tokens.ndim == 2 and seg_tokens.shape[1] == 256):
            raise ValueError(f"SAM embedding shape mismatch: {seg_tokens.shape}")

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
        return seg_tokens, seg_pad_mask, num_segments

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
        
        class_id, name, canonical_key = self._resolve_sample_key(fname)

        # Load I-JEPA global embedding: prefer H5, fallback to NPZ.
        global_vec = None
        if self.ijepa_h5_path is not None:
            try:
                global_vec = self._load_ijepa_from_h5(class_id, name, canonical_key)
            except Exception as e:
                self._warn('ijepa_h5', f"Warning: failed I-JEPA H5 lookup for {canonical_key}: {e}")

        if global_vec is None and self.ijepa_npz_dir is not None:
            try:
                ijepa_path = self._get_corresponding_npz(fname, self.ijepa_npz_dir)
                ijepa_data = np.load(ijepa_path)
                global_vec = ijepa_data['emb'].astype(np.float32)
            except Exception as e:
                self._warn('ijepa_npz', f"Warning: failed I-JEPA NPZ lookup for {fname}: {e}")

        if global_vec is None:
            global_vec = np.zeros(1280, dtype=np.float32)
        else:
            if global_vec.ndim > 1:
                global_vec = global_vec.squeeze()
            if global_vec.shape != (1280,):
                self._warn('ijepa_shape', f"Warning: I-JEPA embedding shape mismatch for {canonical_key}: {global_vec.shape}")
                global_vec = np.zeros(1280, dtype=np.float32)

        # Load SAM segment embeddings: prefer H5, fallback to NPZ.
        raw_seg_tokens = None
        if self.sam_h5_path is not None:
            try:
                raw_seg_tokens = self._load_sam_from_h5(class_id, name)
            except Exception as e:
                self._warn('sam_h5', f"Warning: failed SAM H5 lookup for {canonical_key}: {e}")

        if raw_seg_tokens is None and self.sam_npz_dir is not None:
            try:
                sam_path = self._get_corresponding_npz(fname, self.sam_npz_dir, subdir="masks_npz")
                sam_data = np.load(sam_path)
                raw_seg_tokens = sam_data['emb'].astype(np.float32)
            except Exception as e:
                self._warn('sam_npz', f"Warning: failed SAM NPZ lookup for {fname}: {e}")

        if raw_seg_tokens is None:
            seg_tokens = np.zeros((self.max_segments, 256), dtype=np.float32)
            seg_pad_mask = np.ones(self.max_segments, dtype=np.bool_)
            num_segments = 0
        else:
            try:
                seg_tokens, seg_pad_mask, num_segments = self._pad_or_truncate_seg_tokens(raw_seg_tokens)
            except Exception as e:
                self._warn('sam_shape', f"Warning: invalid SAM embeddings for {canonical_key}: {e}")
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
