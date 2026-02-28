"""
Dataset loader for Unified Segmentation RDM.
Loads images and pre-computed SAM segmentation embeddings.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Optional, Dict, Tuple
import time
import json
import h5py


class SegmentationMaskDataset(Dataset):
    """
    Dataset for loading images with pre-computed SAM segmentation embeddings.
    
    Expected structure:
        image_dir/
            ├── 000000000139.jpg
            ├── 000000000285.jpg
            └── ...
        mask_npz_dir/
            ├── 000000000139.npz  (contains 'emb', 'scores', 'packed', 'shape')
            ├── 000000000285.npz
            └── ...
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_npz_dir: str,
        max_segments: int = 250,
        image_size: int = 256,
        file_ext: str = "*.jpg",
        normalize: bool = True,
        ijepa_cache_dir: Optional[str] = None,
        emb_source: str = "sam",
        h5_path: Optional[str] = None,
        h5_key_format: str = "{class_id}/{name}",
        ijepa_h5_path: Optional[str] = None,
        ijepa_h5_key_format: str = "{class_id}/{name}",
        ijepa_lookup_json: Optional[str] = None,
    ):
        """
        Args:
            image_dir: Directory containing original images
            mask_npz_dir: Directory containing SAM .npz files with embeddings
            max_segments: Maximum number of segments (for padding)
            image_size: Target image size for resizing
            file_ext: File extension pattern for images (e.g., "*.jpg", "*.png")
            normalize: Whether to normalize images to [-1, 1] range (DDPM expects this)
            ijepa_cache_dir: Optional directory with pre-cached IJEPA embeddings in NPZ format (speeds up training)
            emb_source: Embedding source type ("sam", "ijepa", "dinov2", "dino").
                        When not "sam", npz may contain 'emb_image_mean' for mean-subtracted embeddings.
            h5_path: Optional path to centralized .h5 file containing main embeddings (alternative to npz files)
            h5_key_format: Format string for h5 keys, e.g., "{class_id}/{name}" or "{name}"
            ijepa_h5_path: Optional path to .h5 file containing IJEPA embeddings (alternative to ijepa_cache_dir)
            ijepa_h5_key_format: Format string for IJEPA h5 keys
            ijepa_lookup_json: Optional path to JSON lookup for flat IJEPA h5 (maps 'class_id/name' -> row index)
        """
        self.image_dir = image_dir
        self.mask_npz_dir = mask_npz_dir
        self.max_segments = max_segments
        self.image_size = image_size
        self.normalize = normalize
        self.ijepa_cache_dir = ijepa_cache_dir
        self.emb_source = emb_source
        self.h5_path = h5_path
        self.h5_key_format = h5_key_format
        self.ijepa_h5_path = ijepa_h5_path
        self.ijepa_h5_key_format = ijepa_h5_key_format
        self.ijepa_lookup_json = ijepa_lookup_json
        self.h5_file = None
        self.ijepa_h5_file = None
        self._h5_is_flat = False
        self._flat_h5_index = None   # (class_id_str, name_str) → sample_index
        self._ijepa_h5_is_flat = False
        self._ijepa_flat_index = None  # "class_id/name" → row_index
        
        # Open h5 file if provided (lazy loading)
        if self.h5_path and os.path.exists(self.h5_path):
            # Auto-detect flat format (has top-level 'offsets' dataset)
            with h5py.File(self.h5_path, 'r') as _hf:
                if 'offsets' in _hf and 'names' in _hf and 'class_ids' in _hf:
                    self._h5_is_flat = True
                    print(f"Using flat .h5 file: {self.h5_path}")
                else:
                    print(f"Using nested .h5 file: {self.h5_path}")
            # h5 file will be opened lazily in __getitem__ to support multiprocessing
        
        # Detect flat IJEPA h5 format via lookup JSON
        if self.ijepa_lookup_json and os.path.exists(self.ijepa_lookup_json):
            t0 = time.time()
            with open(self.ijepa_lookup_json, 'r') as fp:
                self._ijepa_flat_index = json.load(fp)
            self._ijepa_h5_is_flat = True
            print(f"Using flat IJEPA h5 with lookup: {self.ijepa_lookup_json} "
                  f"({len(self._ijepa_flat_index)} entries, loaded in {time.time()-t0:.1f}s)")
        
        # Find all images with corresponding SAM embeddings
        # Scan mask_npz_dir subdirectories: {mask_npz_dir}/0/masks_npz/, /1/masks_npz/, etc.
        # Build image lookup table (fast: scan images once instead of checking each npz)
        print(f"Building image lookup table from {image_dir}...")
        image_lookup = {}  # {(class_id, basename): full_path}
        
        # Scan image directory structure
        try:
            class_dirs = [d for d in os.listdir(image_dir) 
                         if os.path.isdir(os.path.join(image_dir, d)) and d.isdigit()]
            
            for class_id in class_dirs:
                class_path = os.path.join(image_dir, class_id)
                for img_file in os.listdir(class_path):
                    if img_file.endswith(('.JPEG', '.jpg', '.jpeg', '.JPG', '.png', '.PNG')):
                        basename = os.path.splitext(img_file)[0]
                        image_lookup[(class_id, basename)] = os.path.join(class_path, img_file)
            
            print(f"  Found {len(image_lookup)} images")
        except Exception as e:
            print(f"  WARNING: Failed to build lookup: {e}")
        
        # Discover samples: prefer flat H5, then nested H5, then NPZ directory
        use_flat_h5_discovery = False
        use_h5_discovery = False
        
        if h5_path is not None and os.path.exists(h5_path):
            if self._h5_is_flat:
                use_flat_h5_discovery = True
            elif mask_npz_dir is None or not os.path.exists(mask_npz_dir):
                use_h5_discovery = True
        
        self.image_paths = []
        self.npz_paths = []  # Store corresponding npz paths for __getitem__
        missing_images = []
        
        if use_flat_h5_discovery:
            # ── Flat-H5 discovery (read two small arrays, instant) ──
            print(f"[Flat H5 discovery] Reading index from {h5_path} ...")
            t0 = time.time()
            with h5py.File(self.h5_path, 'r') as h5f:
                class_ids_arr = h5f['class_ids'][:]
                names_arr     = h5f['names'][:]
            # Build lookup: (class_id_str, name_str) → sample_index
            self._flat_h5_index = {}
            for i, (cid, nm) in enumerate(zip(class_ids_arr, names_arr)):
                nm_str = nm.decode() if isinstance(nm, bytes) else str(nm)
                self._flat_h5_index[(str(cid), nm_str)] = i
            elapsed_build = time.time() - t0
            print(f"  Indexed {len(self._flat_h5_index)} samples in {elapsed_build:.1f}s")

            # Intersect with image_lookup (pure RAM, O(1) per lookup)
            matched = 0
            for (class_id, basename), img_path in sorted(image_lookup.items()):
                if (class_id, basename) in self._flat_h5_index:
                    self.image_paths.append(img_path)
                    self.npz_paths.append("")  # placeholder; H5 used in __getitem__
                    matched += 1
                else:
                    missing_images.append(basename)
            elapsed_total = time.time() - t0
            print(f"  Matched {matched}/{len(image_lookup)} images in {elapsed_total:.1f}s "
                  f"({matched / max(len(image_lookup), 1) * 100:.1f}% coverage)")
        elif use_h5_discovery:
            # ── Nested-H5 discovery (slower, walks group tree) ──
            print(f"[Nested H5 discovery] Building key set from {h5_path} ...")
            t0 = time.time()
            h5_key_set = set()  # {(class_id_str, sample_name_str), ...}
            with h5py.File(self.h5_path, 'r') as h5f:
                for cls_key in h5f.keys():                    # '0', '1', …, '999'
                    cls_grp = h5f[cls_key]
                    if 'masks_npz' in cls_grp:
                        masks_grp = cls_grp['masks_npz']
                        for sample_name in masks_grp.keys():  # '1002209', …
                            h5_key_set.add((cls_key, sample_name))
                    else:
                        # Flat layout: class_id/sample_name/emb
                        for sample_name in cls_grp.keys():
                            h5_key_set.add((cls_key, sample_name))
            elapsed_build = time.time() - t0
            print(f"  Built key set: {len(h5_key_set)} entries in {elapsed_build:.1f}s")

            # 2. Intersect with image_lookup (pure RAM, O(1) per lookup)
            matched = 0
            for (class_id, basename), img_path in sorted(image_lookup.items()):
                if (class_id, basename) in h5_key_set:
                    self.image_paths.append(img_path)
                    self.npz_paths.append("")  # placeholder; H5 used in __getitem__
                    matched += 1
                else:
                    missing_images.append(basename)
            elapsed_total = time.time() - t0
            print(f"  Matched {matched}/{len(image_lookup)} images in {elapsed_total:.1f}s "
                  f"({matched / max(len(image_lookup), 1) * 100:.1f}% coverage)")
        else:
            # ── NPZ-based sample discovery (original path) ──
            print(f"Scanning for embeddings in {mask_npz_dir}...")
            
            # Check for numeric subdirectories at root (0/, 1/, 2/, ...)
            try:
                subdirs = [d for d in os.listdir(mask_npz_dir) 
                          if os.path.isdir(os.path.join(mask_npz_dir, d)) and d.isdigit()]
            except FileNotFoundError:
                subdirs = []
            
            if subdirs:
                # Structure: mask_npz_dir/{class_id}/masks_npz/*.npz
                print(f"  Found {len(subdirs)} numeric subdirectories at root")
                npz_files = []
                for subdir in sorted(subdirs, key=int):
                    masks_npz_path = os.path.join(mask_npz_dir, subdir, "masks_npz")
                    if os.path.exists(masks_npz_path):
                        npz_files.extend(glob.glob(os.path.join(masks_npz_path, "*.npz")))
            else:
                # Fallback: try various structures
                masks_npz_path = os.path.join(mask_npz_dir, "masks_npz")
                if os.path.exists(masks_npz_path):
                    subdirs = [d for d in os.listdir(masks_npz_path) 
                              if os.path.isdir(os.path.join(masks_npz_path, d)) and d.isdigit()]
                    if subdirs:
                        npz_files = []
                        for subdir in sorted(subdirs, key=int):
                            npz_files.extend(glob.glob(os.path.join(masks_npz_path, subdir, "*.npz")))
                    else:
                        npz_files = glob.glob(os.path.join(masks_npz_path, "*.npz"))
                else:
                    npz_files = glob.glob(os.path.join(mask_npz_dir, "**", "*.npz"), recursive=True)
            
            npz_files = sorted(npz_files)
            print(f"Found {len(npz_files)} SAM .npz files")
            
            print(f"Matching {len(npz_files)} npz files to images...")
            for npz_path in npz_files:
                npz_name = self._get_basename(npz_path)
                
                path_parts = npz_path.split(os.sep)
                class_folder = None
                for i, part in enumerate(path_parts):
                    if part == "masks_npz" and i > 0:
                        class_folder = path_parts[i-1]
                        break
                
                if class_folder and (class_folder, npz_name) in image_lookup:
                    img_path = image_lookup[(class_folder, npz_name)]
                    self.image_paths.append(img_path)
                    self.npz_paths.append(npz_path)
                else:
                    missing_images.append(npz_name)
        
        if missing_images and len(missing_images) < 10:
            print(f"WARNING: {len(missing_images)} samples have no matching images: {missing_images[:5]}")
        elif missing_images:
            print(f"WARNING: {len(missing_images)} samples have no matching images")
        
        print(f"SegmentationMaskDataset: Loaded {len(self.image_paths)} images with {emb_source} embeddings")
        if not use_h5_discovery and not use_flat_h5_discovery:
            print(f"  (filtered from {len(npz_files)} npz files)")
        print(f"  image_dir: {image_dir}")
        print(f"  mask_npz_dir: {mask_npz_dir}")
        if h5_path:
            print(f"  h5_path: {h5_path}")
        print(f"  max_segments: {max_segments}")
        print(f"  emb_source: {emb_source}")
        
        # Verify at least one image has corresponding data source
        if len(self.image_paths) > 0 and mask_npz_dir and not use_h5_discovery and not use_flat_h5_discovery:
            sample_name = self._get_basename(self.image_paths[0])
            sample_npz = os.path.join(mask_npz_dir, f"{sample_name}.npz")
            if not os.path.exists(sample_npz):
                print(f"WARNING: Sample npz not found: {sample_npz}")
                print(f"  Make sure npz files have same basename as images")
        
        # Image transforms
        if normalize:
            # Normalize to [-1, 1] for DDPM
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            # Keep in [0, 1] range
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
    
    def _get_basename(self, path: str) -> str:
        """Get filename without extension."""
        return os.path.splitext(os.path.basename(path))[0]
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load image and corresponding SAM embeddings.
        
        Returns:
            dict with keys:
                - image: [3, H, W] RGB image tensor
                - seg_embs: [max_segments, 256] segmentation embeddings (padded)
                - num_segments: int, actual number of segments before padding
                - scores: [N] confidence scores for each segment
                - seg_masks: [N, H, W] binary masks (optional, for visualization)
        """
        max_retries = min(5, len(self.image_paths))
        loaded = False
        last_error = None
        img_path = self.image_paths[idx]
        name = self._get_basename(img_path)

        for retry in range(max_retries):
            current_idx = (idx + retry) % len(self.image_paths)
            img_path = self.image_paths[current_idx]
            npz_path = self.npz_paths[current_idx]
            name = self._get_basename(img_path)
            try:
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)  # [3, H, W]
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image = torch.zeros(3, self.image_size, self.image_size)

            try:
                # Try loading from h5 file first if available
                if self.h5_path and os.path.exists(self.h5_path):
                    data = self._load_from_h5(img_path, name)
                else:
                    # Fall back to npz
                    if not os.path.exists(npz_path):
                        raise FileNotFoundError(f"npz not found: {npz_path}")
                    data = np.load(npz_path, allow_pickle=True)

                # Get embeddings (already 256-dim from SAM)
                if 'emb' not in data or data['emb'] is None:
                    raise ValueError("missing 'emb' in data")

                embs = data['emb']  # [N, 256] float16/float32
                if isinstance(embs, np.ndarray) and embs.dtype == np.object_:
                    raise ValueError("data has object dtype, likely corrupt")
                
                # Handle both numpy arrays and tensors
                if isinstance(embs, np.ndarray):
                    embs = torch.from_numpy(embs).float()
                elif not isinstance(embs, torch.Tensor):
                    embs = torch.tensor(embs).float()
                else:
                    embs = embs.float()

                # Get scores
                if 'scores' in data:
                    scores_data = data['scores']
                    if isinstance(scores_data, np.ndarray):
                        scores = torch.from_numpy(scores_data).float()
                    else:
                        scores = torch.tensor(scores_data).float()
                else:
                    scores = torch.ones(len(embs))

                # Optionally load binary masks (for visualization/debugging)
                seg_masks = None
                if 'packed' in data and 'shape' in data:
                    packed = data['packed']
                    shape = data['shape']
                    if len(shape) == 3 and packed.size > 0:
                        N, H, W = shape
                        HW = H * W
                        masks_flat = np.unpackbits(packed, axis=1)[:, :HW]
                        seg_masks = torch.from_numpy(masks_flat.reshape(N, H, W)).bool()

                N_actual = embs.shape[0]
                emb_dim = embs.shape[1] if embs.dim() == 2 else 256

                # Load per-image mean if present (from region embedding extraction)
                emb_image_mean = None
                if 'emb_image_mean' in data:
                    emb_mean_data = data['emb_image_mean']
                    if isinstance(emb_mean_data, np.ndarray):
                        emb_image_mean = torch.from_numpy(np.array(emb_mean_data)).float()
                    else:
                        emb_image_mean = torch.tensor(emb_mean_data).float()

                # Pad or truncate to max_segments
                if N_actual < self.max_segments:
                    pad_size = self.max_segments - N_actual
                    seg_embs = torch.cat([embs, torch.zeros(pad_size, emb_dim)], dim=0)
                    num_segments = N_actual
                else:
                    seg_embs = embs[:self.max_segments]
                    num_segments = self.max_segments
                    scores = scores[:self.max_segments]
                    if seg_masks is not None:
                        seg_masks = seg_masks[:self.max_segments]

                loaded = True
                break
            except Exception as e:
                last_error = e
                print(
                    f"WARNING: skipping SAM sample idx={current_idx} "
                    f"(attempt {retry + 1}/{max_retries}) at {npz_path}: {e}"
                )

        if not loaded:
            # Final fallback if all nearby samples are invalid.
            img_path = self.image_paths[idx]
            name = self._get_basename(img_path)
            try:
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)
            except Exception:
                image = torch.zeros(3, self.image_size, self.image_size)
            print(
                f"ERROR: failed to load valid embeddings after {max_retries} attempts "
                f"(start idx={idx}). Last error: {last_error}. Using dummy embeddings."
            )
            seg_embs = torch.zeros(self.max_segments, 256)
            num_segments = 0
            scores = torch.zeros(0)
            seg_masks = None
            emb_image_mean = None
        
        # Load pre-cached IJEPA embedding if available
        ijepa_emb = None
        
        # Try H5 file first (faster for large datasets)
        if self.ijepa_h5_path is not None:
            try:
                parent_dir = os.path.basename(os.path.dirname(img_path))
                ijepa_data = self._load_from_ijepa_h5(parent_dir, name)
                if 'emb' in ijepa_data:
                    emb_array = ijepa_data['emb']
                    if isinstance(emb_array, np.ndarray):
                        ijepa_emb = torch.from_numpy(emb_array).float()
                    else:
                        ijepa_emb = torch.tensor(emb_array).float()
            except Exception as e:
                # Fall back to NPZ or runtime extraction
                if self.ijepa_cache_dir is None:
                    pass  # Will use runtime extraction
        
        # Fall back to NPZ cache if H5 didn't work
        if ijepa_emb is None and self.ijepa_cache_dir is not None:
            # Extract class ID from image path (ImageNet structure: class_folder/image.JPEG)
            # Try to find the class folder (parent directory of the image)
            try:
                # Get parent directory name (class ID for ImageNet)
                parent_dir = os.path.basename(os.path.dirname(img_path))
                # Try to find npz file in ijepa_cache_dir/{class_id}/{name}.npz
                ijepa_npz_path = os.path.join(self.ijepa_cache_dir, parent_dir, f"{name}.npz")
                
                if os.path.exists(ijepa_npz_path):
                    ijepa_data = np.load(ijepa_npz_path, allow_pickle=True)
                    if 'emb' in ijepa_data:
                        ijepa_emb = torch.from_numpy(ijepa_data['emb']).float()  # [1280] from ViT-H/14
                    else:
                        print(f"WARNING: 'emb' key not found in {ijepa_npz_path}")
                # If not found, will fall back to runtime extraction in get_input()
            except Exception as e:
                print(f"Error loading IJEPA cache for {name}: {e}")
                # Fall back to runtime extraction
        
        # Build output dict
        output = {
            'image': image,
            'seg_embs': seg_embs,  # [max_segments, 256]
            'num_segments': num_segments,  # actual count
            'scores': scores,  # [N]
            'filename': name,
            'emb_source': self.emb_source,
        }
        
        # Add per-image mean (for mean-subtracted region embeddings)
        if emb_image_mean is not None:
            output['emb_image_mean'] = emb_image_mean  # [256]
        
        # Add IJEPA embedding if loaded from cache
        if ijepa_emb is not None:
            output['ijepa_emb'] = ijepa_emb  # [1280] raw ViT-H/14 dimension
        
        if seg_masks is not None:
            output['seg_masks'] = seg_masks
        
        return output
    
    def _load_from_h5(self, img_path: str, name: str) -> Dict:
        """
        Load embeddings from centralized h5 file.
        Auto-dispatches between flat and nested formats.
        
        Args:
            img_path: Path to the image file
            name: Base filename without extension
            
        Returns:
            Dictionary with 'emb', 'scores', and optionally other keys
        """
        # Dispatch to flat loader if applicable
        if self._h5_is_flat:
            return self._load_from_flat_h5(img_path, name)
        
        # ── Nested format ──
        # Lazy open h5 file (supports multiprocessing)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        
        # Build key directly (O(1), no fallback probing)
        class_id = os.path.basename(os.path.dirname(img_path))
        key = f"{class_id}/masks_npz/{name}"
        
        if key not in self.h5_file:
            raise KeyError(f"H5 key not found: {key}")
        h5_data = self.h5_file[key]
        
        # Convert h5 group/dataset to dict
        result = {}
        
        # Handle different h5 structures
        if isinstance(h5_data, h5py.Group):
            # h5 group with multiple datasets
            for key in h5_data.keys():
                result[key] = h5_data[key][:]
        elif isinstance(h5_data, h5py.Dataset):
            # Single dataset - assume it's the embeddings
            result['emb'] = h5_data[:]
        
        # Ensure 'emb' key exists
        if 'emb' not in result:
            # Maybe the dataset itself is stored under a different key
            if 'embeddings' in result:
                result['emb'] = result['embeddings']
            elif 'embedding' in result:
                result['emb'] = result['embedding']
        
        return result
    
    def _load_from_flat_h5(self, img_path: str, name: str) -> Dict:
        """
        Load embeddings from flat H5 file using offset-based indexing.
        
        Flat H5 layout:
            emb          [N_total, 256]   float32  – all segment embeddings concatenated
            scores       [N_total]        float32
            offsets      [n_samples]      int64    – start offset into emb/scores
            n_segments   [n_samples]      int32
            class_ids    [n_samples]      int32
            names        [n_samples]      string
            mask_shapes  [n_samples, 3]   int32    – (N, H, W) per image
        
        Args:
            img_path: Path to the image file (used to extract class_id)
            name: Base filename without extension
            
        Returns:
            Dictionary with 'emb', 'scores', 'shape'
        """
        # Lazy open (supports multiprocessing / DataLoader workers)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        
        class_id = os.path.basename(os.path.dirname(img_path))
        idx = self._flat_h5_index.get((class_id, name))
        if idx is None:
            raise KeyError(f"Flat H5: sample not found: class={class_id} name={name}")
        
        off = int(self.h5_file['offsets'][idx])
        ns  = int(self.h5_file['n_segments'][idx])
        
        result = {
            'emb':    self.h5_file['emb'][off:off + ns],       # (ns, 256) float32
            'scores': self.h5_file['scores'][off:off + ns],    # (ns,)     float32
            'shape':  self.h5_file['mask_shapes'][idx],        # (3,)      int32
        }
        return result
    
    def _load_from_flat_ijepa_h5(self, class_id: str, name: str) -> Dict:
        """
        Load IJEPA embedding from flat h5 using JSON lookup index.
        
        Flat layout:
            emb        (N, 1280)  float32 — one row per image
            class_ids  (N,)       int32
            names      (N,)       string
        Lookup: ijepa_lookup.json  {"class_id/name": row_index, ...}
        """
        if self.ijepa_h5_file is None:
            self.ijepa_h5_file = h5py.File(self.ijepa_h5_path, 'r')
        
        key = f"{class_id}/{name}"
        row = self._ijepa_flat_index.get(key)
        if row is None:
            raise KeyError(f"Flat IJEPA H5: key not found in lookup: {key}")
        
        return {'emb': self.ijepa_h5_file['emb'][row]}  # (1280,) float32
    
    def _load_from_ijepa_h5(self, class_id: str, name: str) -> Dict:
        """
        Load IJEPA embeddings from centralized h5 file.
        Auto-dispatches to flat loader if lookup index is available.
        
        Args:
            class_id: Class directory name
            name: Base filename without extension
            
        Returns:
            Dictionary with 'emb' and optionally other keys
        """
        # Dispatch to flat loader if applicable
        if self._ijepa_h5_is_flat:
            return self._load_from_flat_ijepa_h5(class_id, name)
        
        # ── Nested format (legacy) ──
        # Lazy open h5 file (supports multiprocessing)
        if self.ijepa_h5_file is None:
            self.ijepa_h5_file = h5py.File(self.ijepa_h5_path, 'r')
        
        # Build key directly (O(1), no fallback probing)
        key = self.ijepa_h5_key_format.format(class_id=class_id, name=name)
        
        if key not in self.ijepa_h5_file:
            raise KeyError(f"IJEPA H5 key not found: {key}")
        h5_data = self.ijepa_h5_file[key]
        
        # Convert h5 group/dataset to dict
        result = {}
        
        # Handle different h5 structures
        if isinstance(h5_data, h5py.Group):
            # h5 group with multiple datasets
            for key in h5_data.keys():
                result[key] = h5_data[key][:]
        elif isinstance(h5_data, h5py.Dataset):
            # Single dataset - assume it's the embeddings
            result['emb'] = h5_data[:]
        
        # Ensure 'emb' key exists
        if 'emb' not in result:
            # Maybe the dataset itself is stored under a different key
            if 'embeddings' in result:
                result['emb'] = result['embeddings']
            elif 'embedding' in result:
                result['emb'] = result['embedding']
            elif 'ijepa_emb' in result:
                result['emb'] = result['ijepa_emb']
        
        return result
    
    def __del__(self):
        """Close h5 files on cleanup."""
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass
        if self.ijepa_h5_file is not None:
            try:
                self.ijepa_h5_file.close()
            except:
                pass


class ConditionalSegmentationDataset(SegmentationMaskDataset):
    """
    Extended dataset that includes class labels for conditional generation.
    
    Expects a labels file mapping image names to class indices:
        labels.txt:
            000000000139 5
            000000000285 12
            ...
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_npz_dir: str,
        labels_file: str,
        num_classes: int,
        max_segments: int = 250,
        image_size: int = 256,
        file_ext: str = "*.jpg",
        normalize: bool = True,
    ):
        """
        Args:
            labels_file: Path to text file with "filename class_id" per line
            num_classes: Total number of classes
        """
        super().__init__(
            image_dir=image_dir,
            mask_npz_dir=mask_npz_dir,
            max_segments=max_segments,
            image_size=image_size,
            file_ext=file_ext,
            normalize=normalize,
        )
        
        self.num_classes = num_classes
        
        # Load labels
        self.labels = {}
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        name, label = parts[0], int(parts[1])
                        self.labels[name] = label
            print(f"  Loaded {len(self.labels)} class labels from {labels_file}")
        else:
            print(f"  WARNING: labels file not found: {labels_file}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item with class label."""
        output = super().__getitem__(idx)
        
        # Add class label
        name = output['filename']
        if name in self.labels:
            class_label = self.labels[name]
        else:
            class_label = 0  # Default to class 0 if not found
        
        output['class_label'] = torch.tensor(class_label, dtype=torch.long)
        
        return output


# Utility functions
def collate_seg_batch(batch):
    """
    Custom collate function for variable-length segments.
    Handles batches where different samples have different numbers of segments.
    """
    images = torch.stack([item['image'] for item in batch])
    seg_embs = torch.stack([item['seg_embs'] for item in batch])
    num_segments = torch.tensor([item['num_segments'] for item in batch], dtype=torch.long)
    
    # Pad scores to same length
    max_scores = max(len(item['scores']) for item in batch)
    scores = []
    for item in batch:
        s = item['scores']
        if len(s) < max_scores:
            s = torch.cat([s, torch.zeros(max_scores - len(s))])
        scores.append(s)
    scores = torch.stack(scores)
    
    output = {
        'image': images,
        'seg_embs': seg_embs,
        'num_segments': num_segments,
        'scores': scores,
        'filename': [item['filename'] for item in batch],
    }
    
    # Add emb_source tag if present
    if 'emb_source' in batch[0]:
        output['emb_source'] = batch[0]['emb_source']
    
    # Add per-image mean if present (for mean-subtracted region embeddings)
    if 'emb_image_mean' in batch[0]:
        output['emb_image_mean'] = torch.stack([item['emb_image_mean'] for item in batch])
    
    # Add pre-cached IJEPA embeddings if present
    if 'ijepa_emb' in batch[0]:
        output['ijepa_emb'] = torch.stack([item['ijepa_emb'] for item in batch])
    
    # Add class labels if present
    if 'class_label' in batch[0]:
        output['class_label'] = torch.stack([item['class_label'] for item in batch])
    
    # Add masks if present (check all items have the key to avoid KeyError)
    if all('seg_masks' in item for item in batch):
        # Note: masks may have different sizes, so we keep them as list
        output['seg_masks'] = [item['seg_masks'] for item in batch]
    
    return output


# Test function
def test_dataset():
    """Test the dataset loader."""
    print("Testing SegmentationMaskDataset...")
    
    # These paths should be adjusted to your actual data
    image_dir = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/val2017"
    mask_npz_dir = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/out/masks_npz"
    
    if not os.path.exists(image_dir):
        print(f"WARNING: image_dir not found: {image_dir}")
        print("Skipping test")
        return
    
    dataset = SegmentationMaskDataset(
        image_dir=image_dir,
        mask_npz_dir=mask_npz_dir,
        max_segments=250,
        image_size=256,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test first sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  image shape: {sample['image'].shape}")
        print(f"  seg_embs shape: {sample['seg_embs'].shape}")
        print(f"  num_segments: {sample['num_segments']}")
        print(f"  scores shape: {sample['scores'].shape}")
        print(f"  filename: {sample['filename']}")
        
        # Test dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_seg_batch
        )
        
        batch = next(iter(dataloader))
        print(f"\nBatch:")
        print(f"  images: {batch['image'].shape}")
        print(f"  seg_embs: {batch['seg_embs'].shape}")
        print(f"  num_segments: {batch['num_segments']}")
        print(f"  scores: {batch['scores'].shape}")
        
        print("\n✓ Dataset test passed!")
    else:
        print("WARNING: Dataset is empty")


if __name__ == "__main__":
    test_dataset()
