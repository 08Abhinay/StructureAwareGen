#!/usr/bin/env python3
"""
Pre-compute region + CLS embeddings using MoCo v3 ViT pooled per SAM mask.
Writes directly to per-shard H5 files (no intermediate NPZ).

This script:
  1. Generates SAM masks per image.
  2. Extracts MoCo v3 ViT backbone patch tokens + CLS token (1024-dim).
  3. Pools patch tokens per SAM mask (REN-style masked avg pooling).
  4. Saves backbone features directly (no projection head).
  5. Accumulates results in memory, writes one H5 shard per SLURM job.

Note: We intentionally skip the MoCo v3 3-layer MLP projection head
  (1024→4096→4096→256). That head is trained for contrastive invariance
  and collapses spatial/instance information (eff_rank ~5/256). The raw
  backbone features (1024-dim) retain much richer representations.

H5 shard schema (flat, matches h5_convert.py):
    emb:         (N_total_segments, 1024) float32 — region embeddings (backbone)
    scores:      (N_total_segments,)      float32 — mask quality scores
    offsets:     (n_images,)              int64   — start offset per image
    n_segments:  (n_images,)              int32   — segment count per image
    class_ids:   (n_images,)              int32   — class label per image
    names:       (n_images,)              string  — sample basename
    mask_shapes: (n_images, 3)            int32   — [N, H, W] per image
    cls_emb:     (n_images, 1024)         float32 — MoCo CLS token per image

Usage:
    python3 precompute_region_embeddings_h5.py \\
        --image_dir /path/to/imagenet/train \\
        --output_dir /path/to/shards/ \\
        --sam_checkpoint /path/to/sam_vit_b.pth \\
        --backbone mocov3_vit_large \\
        --moco_checkpoint /path/to/vitl.pth.tar \\
        --start_index 0 --end_index 12811 \\
        --max_keep 100
"""

import gc
import os
import sys
import glob
import json
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from scipy import ndimage as ndi
from pathlib import Path
from collections import defaultdict

# Add SEG-RDM to path for MoCo model loading
SCRIPT_DIR = Path(__file__).resolve().parent
SEG_RDM_DIR = SCRIPT_DIR.parent / "SEG-RDM"
if str(SEG_RDM_DIR) not in sys.path:
    sys.path.insert(0, str(SEG_RDM_DIR))


# ---------------------------------------------------------------------------
# DDP helpers (consistent with precompute_region_embeddings.py)
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialize DDP for multi-GPU processing."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"[Pre-init] Rank {rank}/{world_size}, Local rank {local_rank}, "
              f"Hostname: {os.uname().nodename}")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Post-init] Rank {rank} successfully initialized on device {device}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Single GPU mode] Using device {device}")
    return rank, world_size, local_rank, device


# ---------------------------------------------------------------------------
# ViT backbone loading (MoCo v3)
# ---------------------------------------------------------------------------

def load_backbone(backbone: str, checkpoint: str, device: torch.device):
    """
    Load a MoCo v3 ViT backbone (projection head loaded but NOT used).

    The MoCo v3 checkpoint includes the contrastive projection head but we
    skip it entirely. We only use forward_features() to get the backbone's
    embed_dim-dimensional CLS + patch tokens.

    Returns:
        model: ViT model (eval mode, on device). .head exists but is unused.
        embed_dim: Per-patch feature dimension (1024 for ViT-L, 768 for ViT-B).
        patch_size: Patch size in pixels (16).
        input_size: Expected input resolution (224).
        embed_dim: Same as embed_dim (returned as 'proj_dim' for API compat).
    """
    from rdm.pretrained_enc.models_pretrained_enc import (
        mocov3_vit_base, mocov3_vit_large, load_pretrained_moco,
    )

    # Build model WITH projection head so checkpoint loads cleanly (strict=True).
    # The head is never called — we use backbone features only.
    ckpt_proj_dim = 256  # Must match the checkpoint's head output dim for loading

    if backbone == "mocov3_vit_base":
        model = mocov3_vit_base(proj_dim=ckpt_proj_dim)
        embed_dim = 768
        patch_size = 16
    elif backbone == "mocov3_vit_large":
        model = mocov3_vit_large(proj_dim=ckpt_proj_dim)
        embed_dim = 1024
        patch_size = 16
    else:
        raise ValueError(
            f"Unknown backbone: {backbone}. "
            f"Choose from: mocov3_vit_base, mocov3_vit_large"
        )

    # Load pretrained weights (strips 'module.base_encoder.' prefix, strict=True)
    model = load_pretrained_moco(model, checkpoint)
    input_size = 224

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Output dim = backbone embed_dim (NOT the 256-dim projection)
    proj_dim = embed_dim

    n_patches_side = input_size // patch_size
    print(f"  Backbone: {backbone}")
    print(f"  embed_dim={embed_dim}, patch_size={patch_size}, "
          f"spatial_res={n_patches_side}x{n_patches_side}, "
          f"output_dim={proj_dim} (backbone only, MLP head skipped)")
    return model, embed_dim, patch_size, input_size, proj_dim


def extract_tokens(model, images: torch.Tensor):
    """
    Extract CLS token + patch tokens from MoCo v3 ViT.

    MoCo v3 uses timm's VisionTransformer, whose forward_features()
    returns [B, 1+N, D] where position 0 is CLS.

    Args:
        images: [B, 3, H, W] ImageNet-normalised tensor.

    Returns:
        cls_token:    [B, embed_dim]
        patch_tokens: [B, N_patches, embed_dim]
    """
    with torch.no_grad():
        out = model.forward_features(images)  # [B, 1+N, D]
        cls_token = out[:, 0, :]     # [B, D]
        patch_tokens = out[:, 1:, :] # [B, N, D]
    return cls_token, patch_tokens


# ---------------------------------------------------------------------------
# REN-style masked average pooling
# ---------------------------------------------------------------------------

def pool_tokens_per_mask(
    patch_tokens: torch.Tensor,
    masks: np.ndarray,
    patch_size: int,
    input_size: int,
):
    """
    Pool ViT patch tokens per SAM binary mask.

    Args:
        patch_tokens: [N_patches, embed_dim] for a single image.
        masks: [N_masks, H_img, W_img] boolean mask array.
        patch_size: ViT patch size in pixels.
        input_size: ViT input resolution (e.g. 224).

    Returns:
        region_embs: [N_masks, embed_dim] float32 tensor on same device.
    """
    device = patch_tokens.device
    n_patches_side = input_size // patch_size
    D = patch_tokens.shape[-1]
    N_masks = masks.shape[0]

    if N_masks == 0:
        return torch.zeros(0, D, device=device)

    # Reshape patch tokens to spatial grid: [D, h, w]
    features = patch_tokens.reshape(n_patches_side, n_patches_side, D).permute(2, 0, 1)

    # Resize masks to patch grid resolution
    masks_t = torch.from_numpy(masks.astype(np.float32)).to(device)  # [N, H, W]
    masks_small = F.interpolate(
        masks_t.unsqueeze(1),  # [N, 1, H, W]
        size=(n_patches_side, n_patches_side),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)  # [N, h, w]

    # Threshold: any patch with >0.1 mask overlap counts
    masks_small = (masks_small > 0.1).float()

    # REN-style pooling
    region_embs = torch.einsum("rhw,chw->rc", masks_small, features)
    denom = masks_small.sum(dim=(1, 2)).clamp(min=1.0).unsqueeze(1)
    region_embs = region_embs / denom  # [N, D]

    return region_embs


# ---------------------------------------------------------------------------
# Mask helper functions
# ---------------------------------------------------------------------------

def load_image_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / max(1, union))


def _remove_small_islands(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_area
    return keep[lab]


def _fill_small_holes(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    inv = ~mask
    lab, n = ndi.label(inv)
    if n == 0:
        return mask
    border = np.zeros_like(inv, dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    border_labels = np.unique(lab[border])
    hole_labels = np.setdiff1d(np.arange(1, n + 1), border_labels,
                               assume_unique=False)
    if hole_labels.size == 0:
        return mask
    hole_sizes = ndi.sum(inv, lab, index=hole_labels)
    small_holes = hole_labels[hole_sizes < min_area]
    if small_holes.size == 0:
        return mask
    filled = mask.copy()
    for hl in small_holes:
        filled[lab == hl] = True
    return filled


def filter_mask_like_sam(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = mask.astype(bool)
    if min_area <= 0:
        return mask
    mask = _fill_small_holes(mask, min_area)
    mask = _remove_small_islands(mask, min_area)
    return mask


# ---------------------------------------------------------------------------
# Per-image processing (returns data instead of writing files)
# ---------------------------------------------------------------------------

def process_image(
    image_path: str,
    mask_generator,
    backbone_model,
    device: torch.device,
    embed_dim: int,
    patch_size: int,
    input_size: int,
    min_mask_region_area: int,
    max_keep: int,
    dedup_iou_thresh: float,
    min_quality_score: float,
    min_area_frac: float,
    max_area_frac: float,
    mean_subtract: bool,
    proj_dim: int = 1024,
):
    """
    Process a single image. Returns dict of arrays or None on failure.

    Returns:
        dict with keys:
            emb:        (N, embed_dim) float32  — backbone region embeddings
            scores:     (N,) float32             — mask quality scores
            cls_emb:    (embed_dim,) float32     — z-score normalized CLS token
            n_segments: int                      — number of segments
            mask_shape: (3,) int32               — [N, H, W]
            emb_image_mean: (embed_dim,) float32 or None
    """
    img_rgb = load_image_rgb(image_path)
    H, W, _ = img_rgb.shape

    # 1. Generate SAM masks
    amg = mask_generator.generate(img_rgb)

    candidates = []
    for orig_i, m in enumerate(amg):
        seg = m["segmentation"].astype(bool)
        if min_mask_region_area > 0:
            seg = filter_mask_like_sam(seg, min_mask_region_area)
        area_px = int(seg.sum())
        if area_px == 0:
            continue
        pred_iou = float(m.get("predicted_iou", 0.0))
        stab = float(m.get("stability_score", 0.0))
        score = pred_iou * stab
        area_frac = area_px / (H * W)
        if score < min_quality_score:
            continue
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue
        candidates.append({
            "seg": seg,
            "score": score,
            "area_px": area_px,
            "pred_iou": pred_iou,
            "stability": stab,
            "bbox": list(m.get("bbox", [0, 0, 0, 0])),  # XYWH from SAM
        })

    # 2. Sort and deduplicate (greedy IoU)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    kept = []
    for cand in candidates:
        if len(kept) >= max_keep:
            break
        ok = True
        for prev in kept:
            if iou(cand["seg"], prev["seg"]) >= dedup_iou_thresh:
                ok = False
                break
        if ok:
            kept.append(cand)

    # 3. Prepare image for ViT
    from torchvision import transforms
    vit_transform = transforms.Compose([
        transforms.Resize(input_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_pil = Image.fromarray(img_rgb)
    img_tensor = vit_transform(img_pil).unsqueeze(0).to(device)  # [1,3,224,224]

    # 4. Extract CLS + patch tokens
    cls_token_raw, patch_tokens = extract_tokens(backbone_model, img_tensor)
    # cls_token_raw: [1, D], patch_tokens: [1, N_patches, D]
    cls_token_raw = cls_token_raw.squeeze(0)  # [D]
    patch_tokens = patch_tokens.squeeze(0)    # [N_patches, D]

    # 5. Z-score normalize CLS token (backbone features, no projection head)
    with torch.no_grad():
        cls_proj = cls_token_raw  # [embed_dim] — raw backbone CLS, no MLP head
        cls_mean = cls_proj.mean()
        cls_std = cls_proj.std().clamp(min=1e-6)
        cls_proj = (cls_proj - cls_mean) / cls_std

    N = len(kept)
    if N == 0:
        return {
            "emb": np.zeros((0, proj_dim), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "areas": np.zeros((0,), dtype=np.float32),
            "bboxes": np.zeros((0, 4), dtype=np.float32),
            "pred_ious": np.zeros((0,), dtype=np.float32),
            "stability_scores": np.zeros((0,), dtype=np.float32),
            "cls_emb": cls_proj.cpu().numpy().astype(np.float32),
            "n_segments": 0,
            "mask_shape": np.array([0, H, W], dtype=np.int32),
            "emb_image_mean": None,
        }

    masks = np.stack([k["seg"] for k in kept], axis=0).astype(np.bool_)
    scores = np.asarray([k["score"] for k in kept], dtype=np.float32)
    areas = np.asarray([k["area_px"] for k in kept], dtype=np.float32)
    bboxes = np.asarray([k["bbox"] for k in kept], dtype=np.float32)  # [N, 4] XYWH
    pred_ious = np.asarray([k["pred_iou"] for k in kept], dtype=np.float32)
    stability_scores = np.asarray([k["stability"] for k in kept], dtype=np.float32)

    # 6. Pool per mask (REN-style)
    region_embs = pool_tokens_per_mask(
        patch_tokens, masks, patch_size, input_size
    )  # [N, D]

    # 7. Use backbone region embeddings directly (no projection head)
    embs_proj_t = region_embs  # [N, embed_dim] — raw backbone features

    # 8. Per-image mean subtraction (optional)
    if mean_subtract:
        emb_image_mean_t = embs_proj_t.mean(dim=0)
        embs_proj_t = embs_proj_t - emb_image_mean_t.unsqueeze(0)
        emb_image_mean = emb_image_mean_t.cpu().numpy().astype(np.float32)
    else:
        emb_image_mean = None

    return {
        "emb": embs_proj_t.cpu().numpy().astype(np.float32),
        "scores": scores,
        "areas": areas,
        "bboxes": bboxes,
        "pred_ious": pred_ious,
        "stability_scores": stability_scores,
        "cls_emb": cls_proj.cpu().numpy().astype(np.float32),
        "n_segments": N,
        "mask_shape": np.array([N, H, W], dtype=np.int32),
        "emb_image_mean": emb_image_mean,
    }


# ---------------------------------------------------------------------------
# Incremental H5 shard writer  (write-on-process, crash-safe)
# ---------------------------------------------------------------------------

class H5ShardWriter:
    """
    Append one image's results at a time to a resizable H5 shard.

    Datasets are created with ``maxshape=(None, ...)`` so they can be
    extended after every image.  If the file already exists (e.g. from a
    previous killed run), the writer reopens it in append mode and
    resumes from where it left off.
    """

    def __init__(self, output_path: str, proj_dim: int = 256,
                 store_mean: bool = False):
        self.output_path = output_path
        self.proj_dim = proj_dim
        self.store_mean = store_mean

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            # ---- Resume from existing shard ----
            self.f = h5py.File(output_path, "a")
            self.n_samples = int(self.f.attrs.get("total_samples", 0))
            self.seg_cursor = int(self.f.attrs.get("total_segments", 0))
            # Build set of already-written image names for skip logic
            if "names" in self.f and self.n_samples > 0:
                self.existing_names = set(
                    n.decode() if isinstance(n, bytes) else n
                    for n in self.f["names"][:self.n_samples]
                )
            else:
                self.existing_names = set()
            print(f"  Resuming H5 shard: {self.n_samples} samples, "
                  f"{self.seg_cursor} segments already on disk")
        else:
            # ---- Fresh shard ----
            self.f = h5py.File(output_path, "w")
            self.n_samples = 0
            self.seg_cursor = 0
            self.existing_names = set()
            self._create_datasets()
            # Write initial attrs
            self.f.attrs["total_samples"] = 0
            self.f.attrs["total_segments"] = 0
            self.f.attrs["emb_dim"] = proj_dim
            self.f.attrs["emb_dtype"] = "float32"
            self.f.attrs["source"] = "mocov3_region_extraction"
            self.f.flush()

    # ---- dataset creation (resizable) ----

    def _create_datasets(self):
        D = self.proj_dim
        f = self.f
        # Flat segment arrays (resizable along axis 0)
        f.create_dataset("emb",    shape=(0, D), maxshape=(None, D),
                         dtype="float32", chunks=(512, D))
        f.create_dataset("scores", shape=(0,),    maxshape=(None,),
                         dtype="float32", chunks=(4096,))
        f.create_dataset("areas",  shape=(0,),    maxshape=(None,),
                         dtype="float32", chunks=(4096,))
        f.create_dataset("bboxes", shape=(0, 4),  maxshape=(None, 4),
                         dtype="float32", chunks=(512, 4))
        f.create_dataset("pred_ious",        shape=(0,), maxshape=(None,),
                         dtype="float32", chunks=(4096,))
        f.create_dataset("stability_scores", shape=(0,), maxshape=(None,),
                         dtype="float32", chunks=(4096,))
        # Per-sample arrays
        f.create_dataset("offsets",    shape=(0,),    maxshape=(None,),
                         dtype="int64",  chunks=(4096,))
        f.create_dataset("n_segments", shape=(0,),    maxshape=(None,),
                         dtype="int32",  chunks=(4096,))
        f.create_dataset("class_ids",  shape=(0,),    maxshape=(None,),
                         dtype="int32",  chunks=(4096,))
        f.create_dataset("names",      shape=(0,),    maxshape=(None,),
                         dtype=h5py.string_dtype())
        f.create_dataset("mask_shapes", shape=(0, 3), maxshape=(None, 3),
                         dtype="int32",  chunks=(4096, 3))
        f.create_dataset("cls_emb",    shape=(0, D),  maxshape=(None, D),
                         dtype="float32", chunks=(512, D))
        if self.store_mean:
            f.create_dataset("emb_image_mean", shape=(0, D), maxshape=(None, D),
                             dtype="float32", chunks=(512, D))

    # ---- public API ----

    def has_image(self, name: str) -> bool:
        """Return True if *name* was already written to this shard."""
        return name in self.existing_names

    def append(self, result: dict):
        """
        Append a single image result and flush to disk immediately.

        ``result`` must contain the keys produced by ``process_image()``
        plus ``class_id`` (int) and ``name`` (str).
        """
        f = self.f
        n = result["n_segments"]
        i = self.n_samples
        c = self.seg_cursor

        # Extend flat segment datasets
        if n > 0:
            new_seg = c + n
            f["emb"].resize(new_seg, axis=0)
            f["scores"].resize(new_seg, axis=0)
            f["areas"].resize(new_seg, axis=0)
            f["bboxes"].resize(new_seg, axis=0)
            f["pred_ious"].resize(new_seg, axis=0)
            f["stability_scores"].resize(new_seg, axis=0)

            f["emb"][c:new_seg]              = result["emb"]
            f["scores"][c:new_seg]           = result["scores"]
            f["areas"][c:new_seg]            = result["areas"]
            f["bboxes"][c:new_seg]           = result["bboxes"]
            f["pred_ious"][c:new_seg]        = result["pred_ious"]
            f["stability_scores"][c:new_seg] = result["stability_scores"]

        # Extend per-sample datasets
        f["offsets"].resize(i + 1, axis=0)
        f["n_segments"].resize(i + 1, axis=0)
        f["class_ids"].resize(i + 1, axis=0)
        f["names"].resize(i + 1, axis=0)
        f["mask_shapes"].resize(i + 1, axis=0)
        f["cls_emb"].resize(i + 1, axis=0)

        f["offsets"][i]    = c
        f["n_segments"][i] = n
        f["class_ids"][i]  = result["class_id"]
        f["names"][i]      = result["name"]
        f["mask_shapes"][i] = result["mask_shape"]
        f["cls_emb"][i]    = result["cls_emb"]

        if self.store_mean:
            f["emb_image_mean"].resize(i + 1, axis=0)
            if result.get("emb_image_mean") is not None:
                f["emb_image_mean"][i] = result["emb_image_mean"]
            else:
                f["emb_image_mean"][i] = np.zeros(self.proj_dim, dtype=np.float32)

        # Update bookkeeping
        self.seg_cursor = c + n
        self.n_samples = i + 1
        self.existing_names.add(result["name"])

        # Update attrs and flush so data is durable on disk
        f.attrs["total_samples"]  = self.n_samples
        f.attrs["total_segments"] = self.seg_cursor
        f.flush()

    def close(self):
        """Final flush and close."""
        if self.f:
            self.f.attrs["total_samples"]  = self.n_samples
            self.f.attrs["total_segments"] = self.seg_cursor
            self.f.flush()
            self.f.close()
            self.f = None
        sz = os.path.getsize(self.output_path)
        print(f"  Shard closed: {self.n_samples} samples, "
              f"{self.seg_cursor} segments, {sz / (1024**2):.1f} MB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute MoCo v3 region + CLS embeddings → H5 shard"
    )

    # Paths
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for H5 shard files")
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--sam_model_type", type=str, default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"])

    # Backbone
    parser.add_argument("--backbone", type=str, default="mocov3_vit_large",
                        choices=["mocov3_vit_base", "mocov3_vit_large"])
    parser.add_argument("--moco_checkpoint", type=str, required=True,
                        help="Path to MoCo v3 checkpoint (.pth.tar)")

    # SAM AMG parameters
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.82)
    parser.add_argument("--stability_score_thresh", type=float, default=0.85)
    parser.add_argument("--box_nms_thresh", type=float, default=0.70)
    parser.add_argument("--crop_n_layers", type=int, default=0)
    parser.add_argument("--crop_overlap_ratio", type=float, default=0.35)
    parser.add_argument("--crop_n_points_downscale", type=int, default=2)

    # Post-processing
    parser.add_argument("--min_mask_region_area", type=int, default=300)
    parser.add_argument("--max_keep", type=int, default=100)
    parser.add_argument("--dedup_iou_thresh", type=float, default=0.65)
    parser.add_argument("--min_quality_score", type=float, default=0.75)
    parser.add_argument("--min_area_frac", type=float, default=0.001)
    parser.add_argument("--max_area_frac", type=float, default=0.85)

    # Embedding options
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--mean_subtract", action="store_true",
                        help="Enable per-image mean subtraction")

    # Runtime
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip images whose shard already contains them")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start index (inclusive) in globally sorted list")
    parser.add_argument("--end_index", type=int, default=-1,
                        help="End index (exclusive), -1 = all")
    parser.add_argument("--max_images", type=int, default=-1,
                        help="Max images per rank (-1 = all)")
    parser.add_argument("--subset_fraction", type=float, default=1.0,
                        help="Fraction of images per class to process (stratified)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ---- DDP setup (consistent with working extraction script) ----
    rank, world_size, local_rank, device = setup_ddp()

    if rank == 0:
        print(f"Using {world_size} GPUs for parallel processing")
        print(f"Backbone: {args.backbone}")

    # ---- Load SAM ----
    print("Loading SAM model...")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry[args.sam_model_type](
        checkpoint=args.sam_checkpoint
    ).to(device)
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_overlap_ratio=args.crop_overlap_ratio,
        crop_n_points_downscale_factor=args.crop_n_points_downscale,
        min_mask_region_area=0,
    )

    # ---- Load MoCo v3 backbone ----
    print(f"Loading backbone: {args.backbone}...")
    backbone_model, embed_dim, patch_size, input_size, proj_dim = load_backbone(
        args.backbone, args.moco_checkpoint, device
    )
    # The model's .head exists (loaded from checkpoint) but is NOT called.
    # We use backbone features directly (embed_dim=1024 for ViT-L).

    # ---- Free fragmented CUDA memory after model loading ----
    gc.collect()
    torch.cuda.empty_cache()
    if rank == 0:
        free_mem = torch.cuda.mem_get_info(device)[0] / (1024**3)
        print(f"GPU memory after model loading + cache clear: {free_mem:.1f} GiB free")

    # ---- Gather image paths ----
    if rank == 0:
        print(f"Scanning for images in {args.image_dir}")
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(
            glob.glob(os.path.join(args.image_dir, "**", ext), recursive=True)
        )
    image_paths = sorted(image_paths)

    if rank == 0:
        print(f"Total images found: {len(image_paths)}")
    if len(image_paths) == 0:
        if rank == 0:
            print("No images found!")
        return

    # ---- Optional index slicing ----
    if args.start_index > 0 or args.end_index >= 0:
        start = max(0, int(args.start_index))
        end = len(image_paths) if args.end_index < 0 else min(
            len(image_paths), int(args.end_index)
        )
        if end < start:
            end = start
        image_paths = image_paths[start:end]
        if rank == 0:
            print(f"Applied index slice: [{start}:{end}] -> {len(image_paths)} images")

    # ---- Stratified subset selection (matches precompute_region_embeddings.py) ----
    if args.subset_fraction < 1.0:
        if rank == 0:
            print(f"Stratified sampling: {args.subset_fraction:.1%} per class...")
        class_images = defaultdict(list)
        for idx, img_path in enumerate(image_paths):
            class_id = Path(img_path).parent.name
            class_images[class_id].append(idx)

        rng = np.random.RandomState(args.seed)
        selected_indices = []
        for class_id in sorted(class_images.keys()):
            indices = class_images[class_id]
            n_target = max(1, int(len(indices) * args.subset_fraction))
            sampled = rng.choice(indices, size=min(n_target, len(indices)),
                                 replace=False)
            selected_indices.extend(sampled)

        selected_indices = sorted(selected_indices)
        image_paths = [image_paths[i] for i in selected_indices]
        if rank == 0:
            print(f"Total images to extract: {len(image_paths)}")

    # ---- DDP shard (consistent with working script) ----
    image_paths = image_paths[rank::world_size]
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    print(f"[Rank {rank}] Assigned {len(image_paths)} images")

    if len(image_paths) == 0:
        if rank == 0:
            print("No images to process!")
        return

    # ---- Open H5 shard for incremental writing ----
    shard_name = f"region_moco_shard_r{rank}.h5"
    shard_path = os.path.join(args.output_dir, shard_name)
    writer = H5ShardWriter(
        shard_path, proj_dim=proj_dim, store_mean=args.mean_subtract,
    )

    # ---- Process images and write to disk immediately ----
    n_processed = 0
    n_skipped = 0
    total_segs_written = 0
    errors = []

    pbar = tqdm(image_paths, desc=f"Rank {rank}", position=rank,
                leave=True, dynamic_ncols=True)
    for img_path in pbar:
        try:
            # Extract class_id and name from path:
            #   .../train/{class_id}/{filename}.JPEG
            rel = Path(img_path).relative_to(args.image_dir)
            class_id_str = rel.parts[0]
            name = rel.stem  # e.g. "ILSVRC2012_val_00000001"

            # Skip if already in shard (resume support)
            if args.skip_existing and writer.has_image(name):
                n_skipped += 1
                continue

            # Try to parse class_id as int; if class dirs are synset names,
            # use a hash or index
            try:
                class_id = int(class_id_str)
            except ValueError:
                # For synset-style dirs (n01440764), use sorted index
                class_id = hash(class_id_str) % (2**31)

            result = process_image(
                image_path=img_path,
                mask_generator=mask_generator,
                backbone_model=backbone_model,
                device=device,
                embed_dim=embed_dim,
                patch_size=patch_size,
                input_size=input_size,
                min_mask_region_area=args.min_mask_region_area,
                max_keep=args.max_keep,
                dedup_iou_thresh=args.dedup_iou_thresh,
                min_quality_score=args.min_quality_score,
                min_area_frac=args.min_area_frac,
                max_area_frac=args.max_area_frac,
                mean_subtract=args.mean_subtract,
                proj_dim=proj_dim,
            )

            result["class_id"] = class_id
            result["name"] = name

            # Write to H5 immediately (durable on disk)
            writer.append(result)
            n_processed += 1
            total_segs_written += result["n_segments"]

            pbar.set_postfix(
                segs=result["n_segments"],
                total=total_segs_written,
                saved=n_processed,
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # OOM: clear CUDA cache and continue
                gc.collect()
                torch.cuda.empty_cache()
                errors.append((img_path, f"OOM (cleared cache): {e}"))
                print(f"\n  OOM (cleared cache): {img_path}")
            else:
                errors.append((img_path, str(e)))
                print(f"\n  Error: {img_path}: {e}")
            continue
        except Exception as e:
            errors.append((img_path, str(e)))
            print(f"\n  Error: {img_path}: {e}")
            continue

        # Periodic cache clear to reduce fragmentation
        if n_processed % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ---- Close H5 shard ----
    writer.close()
    if n_skipped > 0:
        print(f"[Rank {rank}] Skipped {n_skipped} already-processed images")

    # ---- Aggregate stats across ranks (consistent with working script) ----
    total_segs = total_segs_written

    if world_size > 1:
        dist.barrier()
        processed_t = torch.tensor([n_processed], device=device)
        masks_t = torch.tensor([total_segs], device=device)
        dist.all_reduce(processed_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(masks_t, op=dist.ReduceOp.SUM)
        total_processed = processed_t.item()
        total_masks_all = masks_t.item()
    else:
        total_processed = n_processed
        total_masks_all = total_segs

    if rank == 0:
        avg = total_masks_all / max(1, total_processed)
        print(f"\nDone! Processed {total_processed} images, "
              f"{total_masks_all} segments ({avg:.1f}/img)")
        print(f"Backbone: {args.backbone}")
        print(f"Shards saved to {args.output_dir}/")

    if errors:
        print(f"[Rank {rank}] Errors: {len(errors)}")
        for p, e in errors[:5]:
            print(f"  {p}: {e}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
