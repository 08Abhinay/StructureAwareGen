"""
SAM (Segment Anything Model) Extractor with Unified Cache and Stochastic Conditioning

This module provides on-the-fly SAM extraction during StyleGAN2 training with:
- Lazy SAM initialization (only loads when needed)
- Unified cache system compatible with precompute_sam_embeddings.py format
- Validation of cached embeddings (all 5 keys: packed, shape, scores, label_map, emb)
- Async disk writes for non-blocking I/O (both NPZ and metadata JSON)
- Stochastic conditioning (dropout-like SAM usage)
- Multi-GPU safe: each GPU extracts its own batch images independently
  (natural load balancing via DistributedSampler in training_loop.py)

Design:
- Cache key: Image file path -> O(1) deterministic path mapping
- Cache format: {cache_dir}/masks_npz/{class_folder}/{image_stem}.npz
- Cache contents: {'packed': (N, ceil(H*W/8)) uint8, 'shape': (3,) int32,
                   'scores': (N,) float32, 'label_map': (H,W) int32,
                   'emb': (N, 256) float16}
- Metadata: {cache_dir}/meta/{class_folder}/{image_stem}.json
- Compatible with precompute_sam_embeddings.py output

Usage:
    extractor = SAMExtractor(
        sam_checkpoint="sam_vit_b_01ec64.pth",
        cache_dir="/path/to/sam_cache_unified",
        device="cuda",
        model_type="vit_b"
    )

    # Extract or load from cache
    embeddings = extractor.extract_or_load(image_paths, images)

    # Pad to batch format
    padded_emb, pad_mask = pad_embeddings_batch(embeddings)
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
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
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

try:
    import h5py
except ImportError:
    h5py = None


# ---------------------------------------------------------------------------
# Distributed training utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Mask utility functions (matching precompute_sam_embeddings.py)
# ---------------------------------------------------------------------------

def mask_stats(mask_bool: np.ndarray) -> Optional[dict]:
    """Compute statistics for a binary mask (same as precompute_sam_embeddings.py)."""
    mask_bool = np.asarray(mask_bool, dtype=bool)
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return None

    h, w = mask_bool.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    area_frac = float(mask_bool.mean())
    cx = float(xs.mean() / w)
    cy = float(ys.mean() / h)

    bw = float((x1 - x0 + 1) / w)
    bh = float((y1 - y0 + 1) / h)
    bbox_area_frac = float(((x1 - x0 + 1) * (y1 - y0 + 1)) / (h * w))

    bbox_area_px = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    fill_frac = float(mask_bool.sum() / bbox_area_px)

    return {
        "area_frac": area_frac,
        "cx": cx, "cy": cy,
        "bbox_w": bw, "bbox_h": bh,
        "bbox_area_frac": bbox_area_frac,
        "bbox_xyxy": [x0, y0, x1, y1],
        "fill_frac": fill_frac,
    }


def bbox_xywh_from_mask(mask_bool: np.ndarray) -> Optional[list]:
    """Extract bounding box in xywh format from mask (same as precompute_sam_embeddings.py)."""
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / max(1, union))


def _remove_small_islands(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0 or ndi is None:
        return mask
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_area
    return keep[lab]


def _fill_small_holes(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0 or ndi is None:
        return mask
    inv = ~mask
    lab, n = ndi.label(inv)
    if n == 0:
        return mask

    border = np.zeros_like(inv, dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    border_labels = np.unique(lab[border])
    hole_labels = np.setdiff1d(np.arange(1, n + 1), border_labels, assume_unique=False)
    if hole_labels.size == 0:
        return mask

    hole_sizes = ndi.sum(inv, lab, index=hole_labels)
    small_holes = hole_labels[hole_sizes < min_area]
    if small_holes.size == 0:
        return mask

    filled = mask.copy()
    for hole_idx in small_holes:
        filled[lab == hole_idx] = True
    return filled


def filter_mask_like_sam(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = mask.astype(bool)
    if min_area <= 0:
        return mask
    mask = _fill_small_holes(mask, min_area)
    mask = _remove_small_islands(mask, min_area)
    return mask


# ---------------------------------------------------------------------------

class AsyncWriter:
    """Background thread for non-blocking file writes (NPZ + JSON metadata)"""

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

            try:
                job_type = item.get('type', 'npz')

                if job_type == 'npz':
                    filepath = item['filepath']
                    data = item['data']
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)

                    if FileLock is not None:
                        lock_path = filepath + '.lock'
                        with FileLock(lock_path, timeout=300):
                            if not os.path.exists(filepath):
                                np.savez_compressed(filepath, **data)
                        try:
                            os.remove(lock_path)
                        except OSError:
                            pass
                    else:
                        if not os.path.exists(filepath):
                            np.savez_compressed(filepath, **data)

                elif job_type == 'json':
                    filepath = item['filepath']
                    data = item['data']
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)

                    if FileLock is not None:
                        lock_path = filepath + '.lock'
                        with FileLock(lock_path, timeout=300):
                            if not os.path.exists(filepath):
                                with open(filepath, 'w') as f:
                                    json.dump(data, f, indent=2)
                        try:
                            os.remove(lock_path)
                        except OSError:
                            pass
                    else:
                        if not os.path.exists(filepath):
                            with open(filepath, 'w') as f:
                                json.dump(data, f, indent=2)

            except Exception as e:
                print(f"[AsyncWriter] Failed to save {item.get('filepath', 'unknown')}: {e}")
            finally:
                self.queue.task_done()

    def save_npz(self, filepath: str, data: dict):
        """Queue a NPZ save operation"""
        self.queue.put({'type': 'npz', 'filepath': filepath, 'data': data})

    def save_json(self, filepath: str, data: dict):
        """Queue a JSON save operation"""
        self.queue.put({'type': 'json', 'filepath': filepath, 'data': data})

    def shutdown(self):
        """Wait for all pending writes and stop thread"""
        self.queue.join()
        self.queue.put(None)
        self.thread.join()


class SAMExtractor:
    """
    SAM extractor with lazy initialization, unified caching, and validation.

    Unified Cache: Uses same directory structure as precompute_sam_embeddings.py
      - NPZ: {cache_dir}/masks_npz/{class}/{stem}.npz  (5 keys)
      - Meta: {cache_dir}/meta/{class}/{stem}.json
    Lazy Init: SAM model only loaded when first extraction is needed
    Validation: File size >1KB, all 5 required keys present, valid shapes
    Async Writes: Non-blocking disk I/O via background thread
    Multi-GPU: Each GPU extracts its own assigned batch images (natural parallelism
               via DistributedSampler — GPUs never see the same image in a batch)
    """

    def __init__(
        self,
        sam_checkpoint: str,
        cache_dir: str,
        device: str = "cuda",
        model_type: str = "vit_b",
        max_masks: int = 250,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        origin_map: Optional[Dict[str, str]] = None,
        # AMG parameters (matching precompute_sam_embeddings.py defaults)
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.82,
        stability_score_thresh: float = 0.85,
        box_nms_thresh: float = 0.70,
        crop_overlap_ratio: float = 0.35,
        crop_n_points_downscale: int = 2,
        crop_n_layers: int = 0,
        dedup_iou_thresh: float = 0.95,
        min_mask_region_area: int = 100,
        # Embedding mode.
        embedding_mode: str = "sam_encoder",
        # Region mode options (parity with precompute_region_embeddings.py).
        region_backbone: str = "ijepa_vit_h14",
        region_ijepa_checkpoint: Optional[str] = None,
        region_proj_path: Optional[str] = None,
        region_max_keep: int = 100,
        region_dedup_iou_thresh: float = 0.65,
        region_min_quality_score: float = 0.75,
        region_min_area_frac: float = 0.001,
        region_max_area_frac: float = 0.85,
        region_mean_subtract: bool = False,
        # Optional per-rank H5 delta output.
        h5_delta_dir: Optional[str] = None,
    ):
        """
        Initialize SAM extractor.

        Args:
            sam_checkpoint: Path to SAM checkpoint file
            cache_dir: Unified cache directory (same as precompute_sam_embeddings.py output_dir)
            device: Device to run SAM on ("cuda" or "cpu")
            model_type: SAM model type ("vit_b", "vit_l", "vit_h")
            max_masks: Maximum number of mask embeddings to keep per image
            rank: Process rank for distributed training (None = auto-detect)
            world_size: Total number of processes (None = auto-detect)
            origin_map: Dict mapping zip filenames to original names (e.g. "00004/img00004572.png" -> "921/499656")
            points_per_side: AMG points per side for mask generation
            pred_iou_thresh: AMG predicted IoU threshold
            stability_score_thresh: AMG stability score threshold
            box_nms_thresh: AMG box NMS threshold
            crop_n_layers: AMG crop layers
            dedup_iou_thresh: IoU threshold for greedy deduplication
            min_mask_region_area: Minimum mask region area (post-processing)
        """
        self.sam_checkpoint = sam_checkpoint
        self.cache_dir = cache_dir
        self.device = device
        self.model_type = model_type
        self.max_masks = max_masks

        # AMG parameters
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.box_nms_thresh = box_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale = crop_n_points_downscale
        self.crop_n_layers = crop_n_layers
        self.dedup_iou_thresh = dedup_iou_thresh
        self.min_mask_region_area = min_mask_region_area
        self.embedding_mode = embedding_mode

        # Region mode parameters.
        self.region_backbone = region_backbone
        self.region_ijepa_checkpoint = region_ijepa_checkpoint
        self.region_proj_path = region_proj_path
        self.region_max_keep = region_max_keep
        self.region_dedup_iou_thresh = region_dedup_iou_thresh
        self.region_min_quality_score = region_min_quality_score
        self.region_min_area_frac = region_min_area_frac
        self.region_max_area_frac = region_max_area_frac
        self.region_mean_subtract = region_mean_subtract
        self.h5_delta_dir = h5_delta_dir

        # Origin map for translating zip names to original names
        # so cache paths match AlignedSegDataset._get_corresponding_npz()
        self.origin_map = origin_map or {}

        # Distributed training support
        self.rank = rank if rank is not None else get_rank()
        self.world_size = world_size if world_size is not None else get_world_size()

        if self.origin_map and self.rank == 0:
            print(f"[SAMExtractor] origin_map loaded with {len(self.origin_map)} entries")

        # Lazy initialization - SAM not loaded until needed
        self.sam = None
        self.mask_generator = None
        self.predictor = None
        self.region_backbone_model = None
        self.region_projection = None
        self.region_embed_dim = None
        self.region_patch_size = None
        self.region_input_size = None

        # Async writer for non-blocking saves
        self.async_writer = AsyncWriter()

        # Optional per-rank H5 delta writer.
        self._delta_h5 = None
        self._delta_h5_path = None
        self._delta_h5_keys = set()
        self._delta_h5_appends = 0
        self._delta_h5_flush_every = 64
        if self.h5_delta_dir is not None:
            if h5py is None:
                raise ImportError("h5py is required when h5_delta_dir is set")
            os.makedirs(self.h5_delta_dir, exist_ok=True)
            self._delta_h5_path = os.path.join(self.h5_delta_dir, f"region_delta_rank{self.rank}.h5")

        if self.embedding_mode not in ["sam_encoder", "region_vit"]:
            raise ValueError(f"Invalid embedding_mode={self.embedding_mode}")
        if self.embedding_mode == "region_vit":
            if ndi is None and self.min_mask_region_area > 0:
                raise ImportError("scipy is required for region_vit mode with mask cleanup enabled")
            if self.region_backbone == "ijepa_vit_h14" and self.region_ijepa_checkpoint is None:
                raise ValueError("region_ijepa_checkpoint is required for region_backbone=ijepa_vit_h14")
            if self.region_proj_path is None:
                raise ValueError("region_proj_path is required for embedding_mode=region_vit")
            if not os.path.exists(self.region_proj_path):
                raise ValueError(f"region projection file not found: {self.region_proj_path}")

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.extractions = 0
        self.lock_wait_time = 0.0  # cumulative seconds spent waiting on locks

        if self.rank == 0:
            print(
                f"[SAMExtractor] mode={self.embedding_mode}, model_type={self.model_type}, "
                f"cache_dir={self.cache_dir}"
            )

    def _lazy_init_sam(self):
        """Lazy initialization of SAM model (only when first needed)"""
        if self.sam is not None:
            return

        print(f"[SAMExtractor][Rank {self.rank}] Loading SAM model from {self.sam_checkpoint}")

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

        # Create mask generator with AMG parameters matching precompute_sam_embeddings.py
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            box_nms_thresh=self.box_nms_thresh,
            crop_n_layers=self.crop_n_layers,
            crop_overlap_ratio=self.crop_overlap_ratio,
            crop_n_points_downscale_factor=self.crop_n_points_downscale,
            min_mask_region_area=0,  # We do post-processing manually (same as precompute)
        )
        if self.embedding_mode == "sam_encoder":
            self.predictor = SamPredictor(self.sam)

        print(f"[SAMExtractor][Rank {self.rank}] SAM model loaded successfully "
              f"(points_per_side={self.points_per_side}, pred_iou_thresh={self.pred_iou_thresh})")

    # ------------------------------------------------------------------
    # Cache path helpers (O(1) lookup)
    # ------------------------------------------------------------------

    def _resolve_image_key(self, image_path: str) -> str:
        """Strip zip prefix and return the bare relative path."""
        if "::" in image_path:
            image_path = image_path.split("::", 1)[1]
        return image_path

    def _get_npz_cache_path(self, image_path: str) -> str:
        """
        Get NPZ cache file path for an image. O(1) path computation.

        If origin_map is available, translates zip names to original names
        so the cache path matches precompute_sam_embeddings.py output:
          "00004/img00004572.png" -> origin_map -> "921/499656"
          -> {cache_dir}/921/masks_npz/499656.npz

        Without origin_map, falls back to:
          {cache_dir}/{class_folder}/masks_npz/{image_stem}.npz
        """
        image_path = self._resolve_image_key(image_path)

        # Use origin_map if available (matches precompute_sam_embeddings.py layout)
        if self.origin_map and image_path in self.origin_map:
            orig_key = self.origin_map[image_path]  # e.g. "921/499656"
            parts = orig_key.split("/")
            if len(parts) == 2:
                return os.path.join(self.cache_dir, parts[0], "masks_npz", f"{parts[1]}.npz")
            return os.path.join(self.cache_dir, "masks_npz", f"{orig_key}.npz")

        # Fallback: use zip filename structure
        path_obj = Path(image_path)
        class_folder = path_obj.parent.name      # e.g. "00003"
        image_stem = path_obj.stem                # e.g. "img00003170"
        return os.path.join(self.cache_dir, class_folder, "masks_npz", f"{image_stem}.npz")

    def _get_meta_cache_path(self, image_path: str) -> str:
        """
        Get metadata JSON cache file path for an image. O(1) path computation.

        Uses origin_map if available to stay consistent with _get_npz_cache_path.
        """
        image_path = self._resolve_image_key(image_path)

        # Use origin_map if available
        if self.origin_map and image_path in self.origin_map:
            orig_key = self.origin_map[image_path]  # e.g. "921/499656"
            parts = orig_key.split("/")
            if len(parts) == 2:
                return os.path.join(self.cache_dir, parts[0], "meta", f"{parts[1]}.json")
            return os.path.join(self.cache_dir, "meta", f"{orig_key.replace('/', '_')}.json")

        # Fallback
        path_obj = Path(image_path)
        class_folder = path_obj.parent.name
        image_stem = path_obj.stem
        return os.path.join(self.cache_dir, class_folder, "meta", f"{image_stem}.json")

    def _sample_identity(self, image_path: str) -> Tuple[str, str]:
        image_key = self._resolve_image_key(image_path)
        if self.origin_map and image_key in self.origin_map:
            mapped = str(self.origin_map[image_key]).strip().replace("\\", "/")
            parts = [p for p in mapped.split("/") if p]
            if len(parts) >= 2:
                return parts[-2], parts[-1]
            if len(parts) == 1:
                return "", Path(parts[0]).stem
            return "", ""

        path_obj = Path(image_key)
        return path_obj.parent.name, path_obj.stem

    @staticmethod
    def _class_id_to_int(class_id: str) -> int:
        return int(class_id) if str(class_id).isdigit() else -1

    @staticmethod
    def _class_id_to_key(class_id: str) -> str:
        return str(int(class_id)) if str(class_id).isdigit() else str(class_id)

    @staticmethod
    def _build_label_map(masks: np.ndarray, scores: np.ndarray, H: int, W: int) -> np.ndarray:
        label_map = -np.ones((H, W), dtype=np.int32)
        occupied = np.zeros((H, W), dtype=bool)
        order = np.argsort(-scores)
        for new_id in order:
            pix = masks[new_id] & (~occupied)
            if pix.sum() == 0:
                continue
            label_map[pix] = int(new_id)
            occupied[pix] = True
        return label_map

    def _lazy_init_region_components(self):
        if self.region_backbone_model is not None and self.region_projection is not None:
            return

        print(f"[SAMExtractor][Rank {self.rank}] Loading region backbone={self.region_backbone}")

        if self.region_backbone == "ijepa_vit_h14":
            scripts_dir = Path(__file__).resolve().parents[3]
            seg_rdm_dir = scripts_dir / "SEG-RDM"
            if str(seg_rdm_dir) not in sys.path:
                sys.path.insert(0, str(seg_rdm_dir))
            from rdm.pretrained_enc.ijepa import vision_transformer as ijepa_vits
            from rdm.pretrained_enc.models_pretrained_enc import load_pretrained_ijepa

            model = ijepa_vits.vit_huge(patch_size=14)
            model = load_pretrained_ijepa(model, self.region_ijepa_checkpoint)
            self.region_embed_dim = model.embed_dim
            self.region_patch_size = 14
            self.region_input_size = 224
        elif self.region_backbone == "dinov2_vitl14":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
            self.region_embed_dim = 1024
            self.region_patch_size = 14
            self.region_input_size = 224
        elif self.region_backbone == "dino_vitb8":
            model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
            self.region_embed_dim = 768
            self.region_patch_size = 8
            self.region_input_size = 224
        else:
            raise ValueError(f"Unknown region_backbone: {self.region_backbone}")

        model = model.to(self.device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.region_backbone_model = model

        proj = nn.Linear(self.region_embed_dim, 256, bias=False).to(self.device)
        state = torch.load(self.region_proj_path, map_location=self.device)
        proj.load_state_dict(state, strict=True)
        proj.eval()
        for param in proj.parameters():
            param.requires_grad_(False)
        self.region_projection = proj

        print(
            f"[SAMExtractor][Rank {self.rank}] Region backbone loaded "
            f"(embed_dim={self.region_embed_dim}, patch={self.region_patch_size}, input={self.region_input_size})"
        )

    def _extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.region_backbone == "ijepa_vit_h14":
                tokens = self.region_backbone_model.forward_features(images)
            elif self.region_backbone.startswith("dinov2"):
                out = self.region_backbone_model.forward_features(images)
                tokens = out["x_norm_patchtokens"] if isinstance(out, dict) else out[:, 1:, :]
            elif self.region_backbone.startswith("dino"):
                out = self.region_backbone_model.get_intermediate_layers(images, n=1)[0]
                tokens = out[:, 1:, :]
            else:
                raise ValueError(f"Unknown region_backbone: {self.region_backbone}")
        return tokens

    def _preprocess_for_vit(self, img_np: np.ndarray) -> torch.Tensor:
        img_t = torch.from_numpy(img_np).to(self.device).float() / 255.0
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)
        img_t = F.interpolate(
            img_t,
            size=(self.region_input_size, self.region_input_size),
            mode="bicubic",
            align_corners=False,
        )
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (img_t - mean) / std

    def _pool_tokens_per_mask(self, patch_tokens: torch.Tensor, masks: np.ndarray) -> torch.Tensor:
        n_patches_side = self.region_input_size // self.region_patch_size
        D = patch_tokens.shape[-1]
        if masks.shape[0] == 0:
            return torch.zeros(0, D, device=self.device)

        features = patch_tokens.reshape(n_patches_side, n_patches_side, D).permute(2, 0, 1)
        masks_t = torch.from_numpy(masks.astype(np.float32)).to(self.device)
        masks_small = F.interpolate(
            masks_t.unsqueeze(1),
            size=(n_patches_side, n_patches_side),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        masks_small = (masks_small > 0.1).float()
        region_embs = torch.einsum("rhw,chw->rc", masks_small, features)
        denom = masks_small.sum(dim=(1, 2)).clamp(min=1.0).unsqueeze(1)
        return region_embs / denom

    def _get_delta_h5(self):
        if self._delta_h5_path is None:
            return None
        if self._delta_h5 is not None:
            return self._delta_h5

        self._delta_h5 = h5py.File(self._delta_h5_path, "a")
        if "class_ids" not in self._delta_h5:
            str_dt = h5py.string_dtype(encoding="utf-8")
            self._delta_h5.create_dataset("class_ids", shape=(0,), maxshape=(None,), dtype=np.int32)
            self._delta_h5.create_dataset("names", shape=(0,), maxshape=(None,), dtype=str_dt)
            self._delta_h5.create_dataset("offsets", shape=(0,), maxshape=(None,), dtype=np.int64)
            self._delta_h5.create_dataset("n_segments", shape=(0,), maxshape=(None,), dtype=np.int32)
            self._delta_h5.create_dataset("mask_shapes", shape=(0, 3), maxshape=(None, 3), dtype=np.int32)
            self._delta_h5.create_dataset("emb", shape=(0, 256), maxshape=(None, 256), dtype=np.float32)
            self._delta_h5.create_dataset("scores", shape=(0,), maxshape=(None,), dtype=np.float32)
            self._delta_h5.attrs["source"] = "region_vit_on_the_fly_delta"
            self._delta_h5.attrs["emb_dim"] = 256
            self._delta_h5.attrs["emb_dtype"] = "float32"
        else:
            class_ids = self._delta_h5["class_ids"][:]
            names = self._delta_h5["names"][:]
            for cid, name in zip(class_ids, names):
                name_str = name.decode() if isinstance(name, (bytes, np.bytes_)) else str(name)
                self._delta_h5_keys.add((str(int(cid)), name_str))
        return self._delta_h5

    def _append_delta_h5(self, image_path: str, npz_data: Dict[str, np.ndarray]):
        h5f = self._get_delta_h5()
        if h5f is None:
            return
        class_id, name = self._sample_identity(image_path)
        key = (self._class_id_to_key(class_id), str(name))
        if key in self._delta_h5_keys:
            return

        emb = np.asarray(npz_data["emb"], dtype=np.float32)
        scores = np.asarray(npz_data["scores"], dtype=np.float32)
        shape = np.asarray(npz_data["shape"], dtype=np.int32)
        ns = int(shape[0]) if shape.size >= 1 else int(emb.shape[0])

        sample_n = h5f["class_ids"].shape[0]
        seg_n = h5f["emb"].shape[0]

        h5f["class_ids"].resize((sample_n + 1,))
        h5f["names"].resize((sample_n + 1,))
        h5f["offsets"].resize((sample_n + 1,))
        h5f["n_segments"].resize((sample_n + 1,))
        h5f["mask_shapes"].resize((sample_n + 1, 3))

        h5f["class_ids"][sample_n] = self._class_id_to_int(class_id)
        h5f["names"][sample_n] = str(name)
        h5f["offsets"][sample_n] = int(seg_n)
        h5f["n_segments"][sample_n] = int(ns)
        h5f["mask_shapes"][sample_n] = np.array(shape[:3], dtype=np.int32)

        if ns > 0:
            h5f["emb"].resize((seg_n + ns, 256))
            h5f["scores"].resize((seg_n + ns,))
            h5f["emb"][seg_n:seg_n + ns] = emb[:ns]
            h5f["scores"][seg_n:seg_n + ns] = scores[:ns]

        self._delta_h5_keys.add(key)
        self._delta_h5_appends += 1
        if self._delta_h5_appends % self._delta_h5_flush_every == 0:
            h5f.flush()

    # ------------------------------------------------------------------
    # Cache validation (all 5 keys)
    # ------------------------------------------------------------------

    def _validate_cache(self, cache_path: str) -> bool:
        """
        Validate cached embeddings. Checks all 5 keys matching
        precompute_sam_embeddings.py format.

        Checks:
        - File exists and size > 1KB
        - Has required keys: 'packed', 'shape', 'scores', 'label_map', 'emb'
        - Valid shapes: shape=(3,), N=shape[0], scores=(N,), emb=(N,256), label_map 2D
        - N <= 1000 (sanity bound)
        """
        if not os.path.exists(cache_path):
            return False

        # Check file size (should be at least 1KB)
        try:
            if os.path.getsize(cache_path) < 1024:
                return False
        except OSError:
            return False

        try:
            data = np.load(cache_path)

            # Check all 5 required keys
            required_keys = ['packed', 'shape', 'scores', 'label_map', 'emb']
            for key in required_keys:
                if key not in data:
                    return False

            shape = data['shape']
            scores = data['scores']
            emb = data['emb']
            label_map = data['label_map']

            # Validate shape array
            if len(shape) != 3:
                return False
            N = int(shape[0])

            # Validate N is reasonable
            if N > 1000:
                return False

            # Validate scores
            if scores.ndim != 1 or scores.shape[0] != N:
                return False

            # Validate embeddings
            if emb.ndim != 2 or emb.shape[0] != N or emb.shape[1] != 256:
                return False

            # Validate label_map
            if label_map.ndim != 2:
                return False

            return True

        except Exception as e:
            print(f"[SAMExtractor] Cache validation failed for {cache_path}: {e}")
            return False

    def _load_from_cache(self, cache_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load embeddings from cache.

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

    # ------------------------------------------------------------------
    # Extraction (matching precompute_sam_embeddings.py output format)
    # ------------------------------------------------------------------

    def _load_image_np(self, image_path: str, image_tensor: Optional[torch.Tensor]) -> np.ndarray:
        if image_tensor is not None:
            img_np = ((image_tensor[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            return img_np
        image = PIL.Image.open(image_path).convert('RGB')
        return np.array(image, dtype=np.uint8)

    def _extract_single_sam_encoder(self, image_path: str, image_tensor: Optional[torch.Tensor]) -> Dict[str, object]:
        self._lazy_init_sam()
        img_np = self._load_image_np(image_path, image_tensor)
        H, W, _ = img_np.shape

        amg = self.mask_generator.generate(img_np)
        self.predictor.set_image(img_np)
        feat = self.predictor.get_image_embedding()
        hf, wf = feat.shape[-2], feat.shape[-1]

        candidates = []
        for orig_i, m in enumerate(amg):
            seg = m['segmentation'].astype(bool)
            area_px = int(seg.sum())
            if area_px == 0:
                continue
            pred_iou = float(m.get('predicted_iou', 0.0))
            stab = float(m.get('stability_score', 0.0))
            score = pred_iou * stab
            st = mask_stats(seg)
            if st is None:
                continue

            mask_t = torch.from_numpy(seg[None, None].astype(np.float32)).to(self.device)
            mask_small = F.interpolate(mask_t, size=(hf, wf), mode="nearest")
            denom = mask_small.sum(dim=(2, 3)) + 1e-6
            emb_t = (feat * mask_small).sum(dim=(2, 3)) / denom
            emb = emb_t.squeeze(0).detach().cpu().to(torch.float16).numpy()

            candidates.append({
                "orig_amg_index": int(orig_i),
                "seg": seg,
                "score": float(score),
                "predicted_iou": pred_iou,
                "stability_score": stab,
                "area_px": area_px,
                "bbox_xywh": bbox_xywh_from_mask(seg),
                "stats": st,
                "emb": emb,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        kept = []
        for cand in candidates:
            if len(kept) >= self.max_masks:
                break
            ok = True
            for prev in kept:
                if iou(cand["seg"], prev["seg"]) >= self.dedup_iou_thresh:
                    ok = False
                    break
            if ok:
                kept.append(cand)

        N = len(kept)
        if N == 0:
            masks = np.zeros((0, H, W), dtype=np.bool_)
            scores = np.zeros((0,), dtype=np.float32)
            embs = np.zeros((0, 256), dtype=np.float16)
            label_map = -np.ones((H, W), dtype=np.int32)
        else:
            masks = np.stack([k["seg"] for k in kept], axis=0).astype(np.bool_)
            scores = np.asarray([k["score"] for k in kept], dtype=np.float32)
            embs = np.stack([k["emb"] for k in kept], axis=0)
            label_map = self._build_label_map(masks, scores, H, W)

        packed = (
            np.packbits(masks.reshape(masks.shape[0], -1), axis=1)
            if masks.shape[0] > 0 else np.zeros((0, 0), dtype=np.uint8)
        )
        npz_data = {
            'packed': packed,
            'shape': np.array(masks.shape, dtype=np.int32),
            'scores': scores,
            'label_map': label_map,
            'emb': embs,
        }

        amg_params = {
            "points_per_side": self.points_per_side,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_score_thresh": self.stability_score_thresh,
            "box_nms_thresh": self.box_nms_thresh,
            "crop_n_layers": self.crop_n_layers,
            "crop_overlap_ratio": self.crop_overlap_ratio,
            "crop_n_points_downscale": self.crop_n_points_downscale,
            "dedup_iou_thresh": self.dedup_iou_thresh,
            "max_keep": self.max_masks,
        }
        meta_masks = []
        for new_id, k in enumerate(kept):
            st = dict(k["stats"])
            st.update({
                "mask_index": int(new_id),
                "orig_amg_index": int(k["orig_amg_index"]),
                "score": float(k["score"]),
                "predicted_iou": float(k["predicted_iou"]),
                "stability_score": float(k["stability_score"]),
                "area_px": int(k["area_px"]),
                "bbox_xywh": k["bbox_xywh"],
            })
            meta_masks.append(st)

        metadata = {
            "image": str(image_path),
            "device": str(self.device),
            "num_masks": int(N),
            "source": "on-the-fly (SAMExtractor sam_encoder)",
            "amg_params": amg_params,
            "masks": meta_masks,
        }

        return {'emb': embs, 'scores': scores, 'npz_data': npz_data, 'metadata': metadata}

    def _extract_single_region_vit(self, image_path: str, image_tensor: Optional[torch.Tensor]) -> Dict[str, object]:
        self._lazy_init_sam()
        self._lazy_init_region_components()
        img_np = self._load_image_np(image_path, image_tensor)
        H, W, _ = img_np.shape

        amg = self.mask_generator.generate(img_np)
        candidates = []
        for orig_i, m in enumerate(amg):
            seg = m['segmentation'].astype(bool)
            seg = filter_mask_like_sam(seg, self.min_mask_region_area)
            area_px = int(seg.sum())
            if area_px == 0:
                continue
            pred_iou = float(m.get('predicted_iou', 0.0))
            stab = float(m.get('stability_score', 0.0))
            score = pred_iou * stab
            st = mask_stats(seg)
            if st is None:
                continue

            if score < self.region_min_quality_score:
                continue
            area_frac = st["area_frac"]
            if area_frac < self.region_min_area_frac or area_frac > self.region_max_area_frac:
                continue

            candidates.append({
                "orig_amg_index": int(orig_i),
                "seg": seg,
                "score": float(score),
                "predicted_iou": pred_iou,
                "stability_score": stab,
                "area_px": area_px,
                "bbox_xywh": bbox_xywh_from_mask(seg),
                "stats": st,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        kept = []
        for cand in candidates:
            if len(kept) >= self.region_max_keep:
                break
            ok = True
            for prev in kept:
                if iou(cand["seg"], prev["seg"]) >= self.region_dedup_iou_thresh:
                    ok = False
                    break
            if ok:
                kept.append(cand)

        N = len(kept)
        if N == 0:
            masks = np.zeros((0, H, W), dtype=np.bool_)
            scores = np.zeros((0,), dtype=np.float32)
            embs = np.zeros((0, 256), dtype=np.float32)
            emb_image_mean = None
            label_map = -np.ones((H, W), dtype=np.int32)
        else:
            masks = np.stack([k["seg"] for k in kept], axis=0).astype(np.bool_)
            scores = np.asarray([k["score"] for k in kept], dtype=np.float32)

            vit_img = self._preprocess_for_vit(img_np)
            patch_tokens = self._extract_patch_tokens(vit_img).squeeze(0)
            region_embs = self._pool_tokens_per_mask(patch_tokens, masks)
            with torch.no_grad():
                embs_t = self.region_projection(region_embs)

            if self.region_mean_subtract:
                emb_mean_t = embs_t.mean(dim=0)
                embs_t = embs_t - emb_mean_t.unsqueeze(0)
                emb_image_mean = emb_mean_t.detach().cpu().numpy().astype(np.float32)
            else:
                emb_image_mean = None

            embs = embs_t.detach().cpu().numpy().astype(np.float32)
            label_map = self._build_label_map(masks, scores, H, W)

        packed = (
            np.packbits(masks.reshape(masks.shape[0], -1), axis=1)
            if masks.shape[0] > 0 else np.zeros((0, 0), dtype=np.uint8)
        )
        npz_data = {
            'packed': packed,
            'shape': np.array(masks.shape, dtype=np.int32),
            'scores': scores,
            'label_map': label_map,
            'emb': embs,
        }
        if self.region_mean_subtract and emb_image_mean is not None:
            npz_data['emb_image_mean'] = emb_image_mean

        amg_params = {
            "points_per_side": self.points_per_side,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_score_thresh": self.stability_score_thresh,
            "box_nms_thresh": self.box_nms_thresh,
            "crop_n_layers": self.crop_n_layers,
            "crop_overlap_ratio": self.crop_overlap_ratio,
            "crop_n_points_downscale": self.crop_n_points_downscale,
            "dedup_iou_thresh": self.region_dedup_iou_thresh,
            "max_keep": self.region_max_keep,
            "backbone": self.region_backbone,
            "min_quality_score": self.region_min_quality_score,
            "min_area_frac": self.region_min_area_frac,
            "max_area_frac": self.region_max_area_frac,
        }
        meta_masks = []
        for new_id, k in enumerate(kept):
            st = dict(k["stats"])
            st.update({
                "mask_index": int(new_id),
                "orig_amg_index": int(k["orig_amg_index"]),
                "score": float(k["score"]),
                "predicted_iou": float(k["predicted_iou"]),
                "stability_score": float(k["stability_score"]),
                "area_px": int(k["area_px"]),
                "bbox_xywh": k["bbox_xywh"],
            })
            meta_masks.append(st)

        metadata = {
            "image": str(image_path),
            "device": str(self.device),
            "num_masks": int(N),
            "source": "on-the-fly (SAMExtractor region_vit)",
            "backbone": self.region_backbone,
            "embed_dim": int(self.region_embed_dim),
            "proj_dim": 256,
            "mean_subtracted": bool(self.region_mean_subtract),
            "dedup_iou_thresh": float(self.region_dedup_iou_thresh),
            "amg_params": amg_params,
            "masks": meta_masks,
        }

        return {'emb': embs, 'scores': scores, 'npz_data': npz_data, 'metadata': metadata}

    def _extract_single(
        self,
        image_path: str,
        image_tensor: Optional[torch.Tensor] = None
    ) -> Dict[str, object]:
        if self.embedding_mode == "region_vit":
            extracted = self._extract_single_region_vit(image_path, image_tensor)
        else:
            extracted = self._extract_single_sam_encoder(image_path, image_tensor)
        self.extractions += 1
        return extracted

    # ------------------------------------------------------------------
    # Main entry point: extract or load from cache
    # ------------------------------------------------------------------

    def extract_or_load(
        self,
        image_paths: List[str],
        image_tensors: Optional[torch.Tensor] = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Extract or load SAM embeddings for a batch of images.
        Processes images sequentially (consistent with precompute_sam_embeddings.py).
        Each GPU processes its own batch images independently (natural parallelism
        via DistributedSampler in training_loop.py).

        Checks cache first, extracts if not found, saves to cache asynchronously
        (both NPZ and metadata JSON).

        Args:
            image_paths: List of image file paths
            image_tensors: Optional pre-loaded image tensors (B, C, H, W) in [-1, 1]

        Returns:
            List of dicts with 'emb' (N, 256) and 'scores' (N,) for each image
        """
        results = []

        for i, image_path in enumerate(image_paths):
            npz_path = self._get_npz_cache_path(image_path)

            # Try loading from cache (O(1) path lookup + file read)
            cached = self._load_from_cache(npz_path)

            if cached is not None:
                results.append(cached)
            else:
                self.cache_misses += 1

                # Use file locking to prevent concurrent extraction by multiple GPUs
                if FileLock is not None:
                    lock_path = npz_path + '.lock'
                    try:
                        lock_start = time.monotonic()
                        with FileLock(lock_path, timeout=300):  # 5 min timeout
                            lock_elapsed = time.monotonic() - lock_start
                            self.lock_wait_time += lock_elapsed

                            if lock_elapsed > 30.0:
                                print(f"[SAMExtractor][Rank {self.rank}] WARNING: "
                                      f"Lock wait {lock_elapsed:.1f}s for {Path(image_path).name}")

                            # Double-check cache (another GPU may have written it)
                            cached = self._load_from_cache(npz_path)
                            if cached is not None:
                                results.append(cached)
                            else:
                                # Extract from image
                                img_tensor = image_tensors[i:i+1] if image_tensors is not None else None
                                extracted = self._extract_single(image_path, img_tensor)
                                results.append({'emb': extracted['emb'], 'scores': extracted['scores']})

                                # Save NPZ and metadata JSON asynchronously
                                self.async_writer.save_npz(npz_path, extracted['npz_data'])
                                meta_path = self._get_meta_cache_path(image_path)
                                self.async_writer.save_json(meta_path, extracted['metadata'])
                                self._append_delta_h5(image_path, extracted['npz_data'])
                        # Clean up lock file
                        try:
                            os.remove(lock_path)
                        except OSError:
                            pass
                    except Exception as e:
                        print(f"[SAMExtractor][Rank {self.rank}] Lock error for {image_path}: {e}")
                        # Fallback: extract without saving to cache
                        img_tensor = image_tensors[i:i+1] if image_tensors is not None else None
                        extracted = self._extract_single(image_path, img_tensor)
                        results.append({'emb': extracted['emb'], 'scores': extracted['scores']})
                        self._append_delta_h5(image_path, extracted['npz_data'])
                else:
                    # No locking available - extract and save
                    img_tensor = image_tensors[i:i+1] if image_tensors is not None else None
                    extracted = self._extract_single(image_path, img_tensor)
                    results.append({'emb': extracted['emb'], 'scores': extracted['scores']})

                    self.async_writer.save_npz(npz_path, extracted['npz_data'])
                    meta_path = self._get_meta_cache_path(image_path)
                    self.async_writer.save_json(meta_path, extracted['metadata'])
                    self._append_delta_h5(image_path, extracted['npz_data'])

        return results

    def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'extractions': self.extractions,
            'hit_rate': hit_rate,
            'lock_wait_time_s': self.lock_wait_time,
            'delta_h5_appends': self._delta_h5_appends,
        }

    def __del__(self):
        """Cleanup async writer on deletion"""
        if hasattr(self, 'async_writer'):
            self.async_writer.shutdown()
        if hasattr(self, '_delta_h5') and self._delta_h5 is not None:
            try:
                self._delta_h5.flush()
                self._delta_h5.close()
            except Exception:
                pass


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
