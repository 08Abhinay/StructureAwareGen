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
import json
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
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
        # AMG parameters (matching precompute_sam_embeddings.py defaults)
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.82,
        stability_score_thresh: float = 0.85,
        box_nms_thresh: float = 0.70,
        crop_n_layers: int = 0,
        dedup_iou_thresh: float = 0.95,
        min_mask_region_area: int = 100,
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
        self.crop_n_layers = crop_n_layers
        self.dedup_iou_thresh = dedup_iou_thresh
        self.min_mask_region_area = min_mask_region_area

        # Distributed training support
        self.rank = rank if rank is not None else get_rank()
        self.world_size = world_size if world_size is not None else get_world_size()

        # Lazy initialization - SAM not loaded until needed
        self.sam = None
        self.mask_generator = None
        self.predictor = None

        # Async writer for non-blocking saves
        self.async_writer = AsyncWriter()

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.extractions = 0
        self.lock_wait_time = 0.0  # cumulative seconds spent waiting on locks

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
            min_mask_region_area=0,  # We do post-processing manually (same as precompute)
        )
        self.predictor = SamPredictor(self.sam)

        print(f"[SAMExtractor][Rank {self.rank}] SAM model loaded successfully "
              f"(points_per_side={self.points_per_side}, pred_iou_thresh={self.pred_iou_thresh})")

    # ------------------------------------------------------------------
    # Cache path helpers (O(1) lookup)
    # ------------------------------------------------------------------

    def _get_npz_cache_path(self, image_path: str) -> str:
        """
        Get NPZ cache file path for an image. O(1) path computation.

        Cache structure: {cache_dir}/masks_npz/{class_folder}/{image_stem}.npz
        Example: /sam_cache_unified/masks_npz/n01440764/n01440764_10026.npz

        Matches precompute_sam_embeddings.py output structure exactly.
        """
        path_obj = Path(image_path)
        class_folder = path_obj.parent.name      # e.g. "n01440764"
        image_stem = path_obj.stem                # e.g. "n01440764_10026"
        return os.path.join(self.cache_dir, "masks_npz", class_folder, f"{image_stem}.npz")

    def _get_meta_cache_path(self, image_path: str) -> str:
        """
        Get metadata JSON cache file path for an image. O(1) path computation.

        Meta structure: {cache_dir}/meta/{class_folder}/{image_stem}.json
        Example: /sam_cache_unified/meta/n01440764/n01440764_10026.json
        """
        path_obj = Path(image_path)
        class_folder = path_obj.parent.name
        image_stem = path_obj.stem
        return os.path.join(self.cache_dir, "meta", class_folder, f"{image_stem}.json")

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

    def _extract_single(
        self,
        image_path: str,
        image_tensor: Optional[torch.Tensor] = None
    ) -> Dict[str, object]:
        """
        Extract SAM embeddings for a single image.
        Produces output matching precompute_sam_embeddings.py format exactly:
        - NPZ with 5 keys: packed, shape, scores, label_map, emb
        - Metadata dict for JSON

        Args:
            image_path: Path to image file
            image_tensor: Optional pre-loaded image tensor (1, C, H, W) in [-1, 1]

        Returns:
            Dict with:
                'emb': (N, 256) float16
                'scores': (N,) float32
                'npz_data': dict with all 5 keys for saving
                'metadata': dict for JSON saving
        """
        # Lazy init SAM if not already loaded
        self._lazy_init_sam()

        # Load image
        if image_tensor is not None:
            # Convert from tensor [-1, 1] to numpy [0, 255]
            img_np = ((image_tensor[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            if img_np.shape[2] == 1:  # Grayscale
                img_np = np.repeat(img_np, 3, axis=2)
        else:
            image = PIL.Image.open(image_path).convert('RGB')
            img_np = np.array(image, dtype=np.uint8)

        H, W, _ = img_np.shape

        # 1) Generate proposals via AMG
        amg = self.mask_generator.generate(img_np)

        # 2) Compute image embedding once (for all masks)
        self.predictor.set_image(img_np)
        feat = self.predictor.get_image_embedding()  # (1, C, H_feat, W_feat)
        hf, wf = feat.shape[-2], feat.shape[-1]

        # 3) Process candidates (matching precompute_sam_embeddings.py logic)
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

            # Per-mask embedding from SAM image encoder
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

        # 4) Sort by score and deduplicate (greedy IoU, same as precompute)
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

        # 5) Build final arrays (matching precompute format)
        N = len(kept)
        if N == 0:
            masks = np.zeros((0, H, W), dtype=np.bool_)
            scores = np.zeros((0,), dtype=np.float32)
            embs = np.zeros((0, 256), dtype=np.float16)
            label_map = -np.ones((H, W), dtype=np.int32)
        else:
            masks = np.stack([k["seg"] for k in kept], axis=0).astype(np.bool_)
            scores = np.asarray([k["score"] for k in kept], dtype=np.float32)
            embs = np.stack([k["emb"] for k in kept], axis=0)  # (N, 256) float16

            # Non-overlapping label map (same as precompute)
            label_map = -np.ones((H, W), dtype=np.int32)
            occupied = np.zeros((H, W), dtype=bool)
            order = np.argsort(-scores)
            for new_id in order:
                pix = masks[new_id] & (~occupied)
                if pix.sum() == 0:
                    continue
                label_map[pix] = int(new_id)
                occupied[pix] = True

        # 6) Pack masks for compressed storage
        packed = (np.packbits(masks.reshape(masks.shape[0], -1), axis=1)
                  if masks.shape[0] > 0
                  else np.zeros((0, 0), dtype=np.uint8))

        # 7) Build NPZ data dict (all 5 keys)
        npz_data = {
            'packed': packed,
            'shape': np.array(masks.shape, dtype=np.int32),
            'scores': scores,
            'label_map': label_map,
            'emb': embs,
        }

        # 8) Build metadata for JSON (matching precompute format)
        amg_params = {
            "points_per_side": self.points_per_side,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_score_thresh": self.stability_score_thresh,
            "box_nms_thresh": self.box_nms_thresh,
            "crop_n_layers": self.crop_n_layers,
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
            "source": "on-the-fly (SAMExtractor)",
            "amg_params": amg_params,
            "masks": meta_masks,
        }

        self.extractions += 1

        return {
            'emb': embs,
            'scores': scores,
            'npz_data': npz_data,
            'metadata': metadata,
        }

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
                else:
                    # No locking available - extract and save
                    img_tensor = image_tensors[i:i+1] if image_tensors is not None else None
                    extracted = self._extract_single(image_path, img_tensor)
                    results.append({'emb': extracted['emb'], 'scores': extracted['scores']})

                    self.async_writer.save_npz(npz_path, extracted['npz_data'])
                    meta_path = self._get_meta_cache_path(image_path)
                    self.async_writer.save_json(meta_path, extracted['metadata'])

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
