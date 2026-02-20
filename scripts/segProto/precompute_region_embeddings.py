#!/usr/bin/env python3
"""
Pre-compute region embeddings using ViT patch tokens pooled per SAM mask.
Replaces SAM encoder features with I-JEPA/DINO/DINOv2 patch features for
spatially-discriminative per-region embeddings (REN-style masked avg pooling).

This script:
  1. Uses existing SAM masks (from precompute_sam_embeddings.py) or generates new ones.
  2. Extracts ViT patch tokens from a chosen backbone (I-JEPA, DINOv2, DINO).
  3. Pools patch tokens per SAM mask via einsum('rhw,chw->rc', mask, features).
  4. Projects to 256-dim, applies per-image mean subtraction, saves as .npz.

Usage (multi-GPU with DDP):
    torchrun --nproc_per_node=4 scripts/segProto/precompute_region_embeddings.py \
        --image_dir /path/to/images \
        --output_dir /path/to/output \
        --sam_checkpoint /path/to/sam_checkpoint.pth \
        --backbone ijepa_vit_h14 \
        --ijepa_checkpoint /path/to/ijepa_checkpoint.pth.tar \
        --max_keep 100 \
        --subset_fraction 0.4

Supported backbones:
    ijepa_vit_h14: I-JEPA ViT-H/14 (1280-dim, 256 patches for 224px) [default]
    dinov2_vitl14: DINOv2 ViT-L/14 (1024-dim) via torch.hub
    dino_vitb8:    DINO ViT-B/8 (768-dim) via torch.hub
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from scipy import ndimage as ndi
from pathlib import Path

# Add SEG-RDM to path for I-JEPA model loading
SCRIPT_DIR = Path(__file__).resolve().parent
SEG_RDM_DIR = SCRIPT_DIR.parent / "SEG-RDM"
if str(SEG_RDM_DIR) not in sys.path:
    sys.path.insert(0, str(SEG_RDM_DIR))


# ---------------------------------------------------------------------------
# DDP helpers (reused from precompute_sam_embeddings.py)
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
# ViT backbone loading
# ---------------------------------------------------------------------------

def load_backbone(backbone: str, checkpoint: str, device: torch.device):
    """
    Load a ViT backbone and return (model, embed_dim, patch_size, spatial_res).

    Returns:
        model: ViT model (eval mode, on device).
        embed_dim: Per-patch feature dimension (e.g. 1280 for ViT-H/14).
        patch_size: Patch size in pixels.
        input_size: Expected input resolution (224 for all current backbones).
    """
    if backbone == "ijepa_vit_h14":
        from rdm.pretrained_enc.ijepa import vision_transformer as ijepa_vits
        from rdm.pretrained_enc.models_pretrained_enc import load_pretrained_ijepa
        model = ijepa_vits.vit_huge(patch_size=14)
        model = load_pretrained_ijepa(model, checkpoint)
        embed_dim = model.embed_dim  # 1280
        patch_size = 14
        input_size = 224

    elif backbone == "dinov2_vitl14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        embed_dim = 1024
        patch_size = 14
        input_size = 224

    elif backbone == "dino_vitb8":
        model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
        embed_dim = 768
        patch_size = 8
        input_size = 224

    else:
        raise ValueError(f"Unknown backbone: {backbone}. "
                         f"Choose from: ijepa_vit_h14, dinov2_vitl14, dino_vitb8")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_patches_side = input_size // patch_size
    print(f"  Backbone: {backbone}")
    print(f"  embed_dim={embed_dim}, patch_size={patch_size}, "
          f"spatial_res={n_patches_side}x{n_patches_side}")
    return model, embed_dim, patch_size, input_size


def extract_patch_tokens(model, backbone: str, images: torch.Tensor):
    """
    Extract all patch tokens from a ViT backbone.

    Args:
        model: Loaded ViT.
        backbone: Backbone name string.
        images: [B, 3, H, W] ImageNet-normalised tensor.

    Returns:
        tokens: [B, N_patches, embed_dim]  (no CLS token).
    """
    with torch.no_grad():
        if backbone == "ijepa_vit_h14":
            # I-JEPA ViT returns [B, N, D] from forward_features() (no CLS)
            tokens = model.forward_features(images)  # [B, 256, 1280]

        elif backbone.startswith("dinov2"):
            # DINOv2 forward_features returns dict with 'x_norm_patchtokens'
            # or we can override to get patch tokens
            out = model.forward_features(images)
            if isinstance(out, dict):
                tokens = out["x_norm_patchtokens"]
            else:
                # Older API: returns [B, 1+N, D] with CLS at position 0
                tokens = out[:, 1:, :]

        elif backbone.startswith("dino"):
            # DINO ViT: forward returns [B, 1+N, D] (CLS + patches)
            # Use get_intermediate_layers for cleaner access
            out = model.get_intermediate_layers(images, n=1)[0]
            # out is [B, 1+N, D] — remove CLS
            tokens = out[:, 1:, :]

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    return tokens  # [B, N, D]


# ---------------------------------------------------------------------------
# REN-style masked average pooling
# ---------------------------------------------------------------------------

def pool_tokens_per_mask(
    patch_tokens: torch.Tensor,
    masks: np.ndarray,
    patch_size: int,
    input_size: int,
    image_hw: tuple,
):
    """
    Pool ViT patch tokens per SAM binary mask using REN's einsum approach.

    Args:
        patch_tokens: [N_patches, embed_dim] for a single image.
        masks: [N_masks, H_img, W_img] boolean mask array.
        patch_size: ViT patch size in pixels.
        input_size: ViT input resolution (e.g. 224).
        image_hw: (H_img, W_img) original image size.

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
    # features: [D, h, w] where h=w=n_patches_side

    # Resize masks to patch grid resolution
    # masks: [N, H_img, W_img] bool -> [N, h, w] float
    masks_t = torch.from_numpy(masks.astype(np.float32)).to(device)  # [N, H, W]
    masks_small = F.interpolate(
        masks_t.unsqueeze(1),  # [N, 1, H, W]
        size=(n_patches_side, n_patches_side),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)  # [N, h, w]

    # Threshold: any patch with >0.1 mask overlap counts
    masks_small = (masks_small > 0.1).float()

    # REN-style pooling: einsum('rhw,chw->rc', mask, features) / mask.sum()
    # region_embs[r, c] = sum_{h,w} masks_small[r,h,w] * features[c,h,w]
    region_embs = torch.einsum("rhw,chw->rc", masks_small, features)
    denom = masks_small.sum(dim=(1, 2)).clamp(min=1.0).unsqueeze(1)  # [N, 1]
    region_embs = region_embs / denom  # [N, D]

    return region_embs


# ---------------------------------------------------------------------------
# Projection layer
# ---------------------------------------------------------------------------

def get_or_create_projection(embed_dim: int, proj_dim: int, proj_path: str, device: torch.device):
    """
    Load or create a fixed linear projection: embed_dim -> proj_dim.
    Saved/loaded for reproducibility across runs.
    """
    proj = nn.Linear(embed_dim, proj_dim, bias=False).to(device)

    if os.path.exists(proj_path):
        state = torch.load(proj_path, map_location=device)
        proj.load_state_dict(state)
        print(f"  Loaded projection from {proj_path}")
    else:
        # Xavier init for stable projection
        nn.init.xavier_uniform_(proj.weight)
        os.makedirs(os.path.dirname(proj_path), exist_ok=True)
        torch.save(proj.state_dict(), proj_path)
        print(f"  Created and saved projection to {proj_path}")

    proj.eval()
    for p in proj.parameters():
        p.requires_grad_(False)
    return proj


# ---------------------------------------------------------------------------
# Mask helper functions (reused from precompute_sam_embeddings.py)
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
    hole_labels = np.setdiff1d(np.arange(1, n + 1), border_labels, assume_unique=False)
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


def mask_stats(mask_bool: np.ndarray):
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
        "area_frac": area_frac, "cx": cx, "cy": cy,
        "bbox_w": bw, "bbox_h": bh,
        "bbox_area_frac": bbox_area_frac,
        "bbox_xyxy": [x0, y0, x1, y1],
        "fill_frac": fill_frac,
    }


def bbox_xywh_from_mask(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)]


def get_output_path(image_path, image_dir, output_dir):
    return Path(output_dir) / Path(image_path).relative_to(image_dir).parent / Path(image_path).stem


def validate_npz_file(npz_path, min_size_kb=1):
    try:
        file_size = npz_path.stat().st_size
        if file_size < min_size_kb * 1024:
            return False
        data = np.load(npz_path)
        required_keys = ['packed', 'shape', 'scores', 'label_map', 'emb']
        for key in required_keys:
            if key not in data:
                data.close()
                return False
        data.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main per-image processing
# ---------------------------------------------------------------------------

def process_image(
    image_path: str,
    mask_generator,
    backbone_model,
    backbone_name: str,
    projection: nn.Linear,
    device: torch.device,
    embed_dim: int,
    patch_size: int,
    input_size: int,
    min_mask_region_area: int,
    max_keep: int,
    dedup_iou_thresh: float,
    output_base: Path,
    skip_existing: bool = True,
    mean_subtract: bool = True,
    amg_params: dict = None,
):
    """
    Process a single image: generate SAM masks, extract ViT patch tokens,
    pool per mask, project to 256-dim, optionally mean-subtract, save .npz.
    """
    npz_path = output_base.parent / "masks_npz" / f"{output_base.name}.npz"

    if skip_existing and npz_path.exists() and validate_npz_file(npz_path):
        return None

    img_rgb = load_image_rgb(image_path)
    H, W, _ = img_rgb.shape

    # ---- 1. Generate SAM masks ----
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
        st = mask_stats(seg)
        if st is None:
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

    # ---- 2. Quality and size filtering ----
    # Apply thresholds from args (passed via process_image kwargs)
    min_qual = amg_params.get('min_quality_score', 0.75) if amg_params else 0.75
    min_area = amg_params.get('min_area_frac', 0.001) if amg_params else 0.001
    max_area = amg_params.get('max_area_frac', 0.85) if amg_params else 0.85
    
    filtered = []
    for cand in candidates:
        # Quality check
        if cand["score"] < min_qual:
            continue
        # Size check
        area_f = cand["stats"]["area_frac"]
        if area_f < min_area or area_f > max_area:
            continue
        filtered.append(cand)
    
    # ---- 3. Sort and deduplicate (greedy IoU) ----
    filtered.sort(key=lambda x: x["score"], reverse=True)
    kept = []
    for cand in filtered:
        if len(kept) >= max_keep:
            break
        ok = True
        for prev in kept:
            if iou(cand["seg"], prev["seg"]) >= dedup_iou_thresh:
                ok = False
                break
        if ok:
            kept.append(cand)

    N = len(kept)
    if N == 0:
        masks = np.zeros((0, H, W), dtype=np.bool_)
        scores = np.zeros((0,), dtype=np.float32)
        embs_proj = np.zeros((0, 256), dtype=np.float32)
        emb_image_mean = None  # No mean for empty images
        label_map = -np.ones((H, W), dtype=np.int32)
    else:
        masks = np.stack([k["seg"] for k in kept], axis=0).astype(np.bool_)
        scores = np.asarray([k["score"] for k in kept], dtype=np.float32)

        # ---- 4. Extract ViT patch tokens ----
        # Prepare image for ViT: resize to input_size, ImageNet normalise
        img_pil = Image.fromarray(img_rgb)
        from torchvision import transforms
        vit_transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = vit_transform(img_pil).unsqueeze(0).to(device)  # [1, 3, 224, 224]

        patch_tokens = extract_patch_tokens(
            backbone_model, backbone_name, img_tensor
        )  # [1, N_patches, D]
        patch_tokens = patch_tokens.squeeze(0)  # [N_patches, D]

        # ---- 5. Pool per mask (REN-style) ----
        region_embs = pool_tokens_per_mask(
            patch_tokens, masks, patch_size, input_size, (H, W)
        )  # [N, D]

        # ---- 6. Project to 256-dim ----
        with torch.no_grad():
            embs_proj_t = projection(region_embs)  # [N, 256]

        # ---- 7. Per-image mean subtraction (optional) ----
        if mean_subtract:
            emb_image_mean_t = embs_proj_t.mean(dim=0)  # [256]
            embs_proj_t = embs_proj_t - emb_image_mean_t.unsqueeze(0)
            emb_image_mean = emb_image_mean_t.cpu().numpy().astype(np.float32)
        else:
            emb_image_mean = None
        
        embs_proj = embs_proj_t.cpu().numpy().astype(np.float32)

        # ---- 8. Non-overlapping label map ----
        label_map = -np.ones((H, W), dtype=np.int32)
        occupied = np.zeros((H, W), dtype=bool)
        order = np.argsort(-scores)
        for new_id in order:
            pix = masks[new_id] & (~occupied)
            if pix.sum() == 0:
                continue
            label_map[pix] = int(new_id)
            occupied[pix] = True

    # ---- 9. Save .npz ----
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    packed = (np.packbits(masks.reshape(masks.shape[0], -1), axis=1)
              if masks.shape[0] > 0 else np.zeros((0, 0), dtype=np.uint8))
    save_dict = {
        'packed': packed,
        'shape': np.array(masks.shape, dtype=np.int32),
        'scores': scores,
        'label_map': label_map,
        'emb': embs_proj,  # [N, 256] float32 (projected, optionally mean-subtracted)
    }
    if mean_subtract and emb_image_mean is not None:
        save_dict['emb_image_mean'] = emb_image_mean  # [256] float32 (only if mean_subtract=True)
    np.savez_compressed(npz_path, **save_dict)

    # ---- 10. Save metadata JSON ----
    json_path = output_base.parent / "meta" / f"{output_base.name}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
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
        "device": str(device),
        "num_masks": int(N),
        "backbone": backbone_name,
        "embed_dim": embed_dim,
        "proj_dim": 256,
        "mean_subtracted": mean_subtract,
        "dedup_iou_thresh": dedup_iou_thresh,
        "amg_params": amg_params or {},
        "masks": meta_masks,
    }
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return N


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute region embeddings with ViT backbone + SAM masks')

    # Paths
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--sam_checkpoint', type=str, required=True,
                        help='Path to SAM checkpoint (.pth)')
    parser.add_argument('--sam_model_type', type=str, default='vit_b',
                        choices=['vit_b', 'vit_l', 'vit_h'])

    # Backbone
    parser.add_argument('--backbone', type=str, default='ijepa_vit_h14',
                        choices=['ijepa_vit_h14', 'dinov2_vitl14', 'dino_vitb8'],
                        help='ViT backbone for patch token extraction')
    parser.add_argument('--ijepa_checkpoint', type=str, default=None,
                        help='Path to I-JEPA checkpoint (required if backbone=ijepa_vit_h14)')

    # SAM AMG parameters
    parser.add_argument('--points_per_side', type=int, default=32)
    parser.add_argument('--pred_iou_thresh', type=float, default=0.82)
    parser.add_argument('--stability_score_thresh', type=float, default=0.85)
    parser.add_argument('--box_nms_thresh', type=float, default=0.70)
    parser.add_argument('--crop_n_layers', type=int, default=0)
    parser.add_argument('--crop_overlap_ratio', type=float, default=0.35)
    parser.add_argument('--crop_n_points_downscale', type=int, default=2)

    # Post-processing
    parser.add_argument('--min_mask_region_area', type=int, default=300)
    parser.add_argument('--max_keep', type=int, default=64,
                        help='Max regions to keep (lowered from 100 to match REN region counts)')
    parser.add_argument('--dedup_iou_thresh', type=float, default=0.50,
                        help='IoU threshold for dedup (lowered from 0.90 for more diversity)')
    parser.add_argument('--min_quality_score', type=float, default=0.75,
                        help='Minimum score (pred_iou * stability) to keep candidate')
    parser.add_argument('--min_area_frac', type=float, default=0.001,
                        help='Minimum mask area as fraction of image (reject tiny masks)')
    parser.add_argument('--max_area_frac', type=float, default=0.85,
                        help='Maximum mask area as fraction of image (reject near-full-image masks)')

    # Embedding options
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--mean_subtract', action='store_true',
                        help='Enable per-image mean subtraction (default: False)')

    # Runtime
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--max_images', type=int, default=-1)
    parser.add_argument('--subset_fraction', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Validate
    if args.backbone == "ijepa_vit_h14" and args.ijepa_checkpoint is None:
        parser.error("--ijepa_checkpoint is required when --backbone=ijepa_vit_h14")

    # Setup DDP
    rank, world_size, local_rank, device = setup_ddp()

    if rank == 0:
        print(f"Using {world_size} GPUs for parallel processing")
        print(f"Backbone: {args.backbone}")
        print(f"IoU dedup threshold: {args.dedup_iou_thresh}")
        print(f"Mean subtraction: {args.mean_subtract}")

    # ---- Load SAM for mask generation ----
    if rank == 0:
        print(f"Loading SAM model ({args.sam_model_type})...")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint).to(device)
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

    # ---- Load ViT backbone ----
    if rank == 0:
        print(f"Loading backbone: {args.backbone}...")
    backbone_model, embed_dim, patch_size, input_size = load_backbone(
        args.backbone, args.ijepa_checkpoint or "", device
    )

    # ---- Load or create projection ----
    proj_path = os.path.join(args.output_dir, f"projection_{args.backbone}.pt")
    projection = get_or_create_projection(embed_dim, args.proj_dim, proj_path, device)

    # ---- Gather image paths ----
    if rank == 0:
        print(f"Scanning for images in {args.image_dir}")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, '**', ext), recursive=True))
    image_paths = sorted(image_paths)

    if rank == 0:
        print(f"Total images found: {len(image_paths)}")
    if len(image_paths) == 0:
        if rank == 0:
            print("No images found!")
        return

    # ---- Stratified subset selection ----
    if args.subset_fraction < 1.0:
        from collections import defaultdict
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
            class_output_dir = Path(args.output_dir) / class_id / "masks_npz"
            n_existing = len(list(class_output_dir.glob("*.npz"))) if class_output_dir.exists() else 0
            n_needed = max(0, n_target - n_existing)
            if n_needed > 0:
                sampled = rng.choice(indices, size=min(n_needed, len(indices)), replace=False)
                selected_indices.extend(sampled)

        selected_indices = sorted(selected_indices)
        image_paths = [image_paths[i] for i in selected_indices]
        if rank == 0:
            print(f"Total images to extract: {len(image_paths)}")

    # ---- DDP shard ----
    image_paths = image_paths[rank::world_size]
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    print(f"[Rank {rank}] Assigned {len(image_paths)} images")

    # ---- Filter existing ----
    if args.skip_existing:
        filtered = []
        skipped = 0
        for img_path in image_paths:
            output_base = get_output_path(img_path, args.image_dir, args.output_dir)
            npz_path = output_base.parent / "masks_npz" / f"{output_base.name}.npz"
            if npz_path.exists() and validate_npz_file(npz_path):
                skipped += 1
            else:
                filtered.append(img_path)
        if rank == 0:
            print(f"Skipped {skipped} valid existing embeddings")
        image_paths = filtered

    if len(image_paths) == 0:
        if rank == 0:
            print("All embeddings already extracted!")
        return

    # ---- Process images ----
    print(f"[Rank {rank}] Processing {len(image_paths)} images...")
    total_masks = 0

    amg_params = {
        "points_per_side": args.points_per_side,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "dedup_iou_thresh": args.dedup_iou_thresh,
        "max_keep": args.max_keep,
        "backbone": args.backbone,
    }

    pbar = tqdm(image_paths, desc=f"Rank {rank}", position=rank, leave=True, dynamic_ncols=True)
    for img_path in pbar:
        output_base = get_output_path(img_path, args.image_dir, args.output_dir)
        try:
            n_masks = process_image(
                image_path=img_path,
                mask_generator=mask_generator,
                backbone_model=backbone_model,
                backbone_name=args.backbone,
                projection=projection,
                device=device,
                embed_dim=embed_dim,
                patch_size=patch_size,
                input_size=input_size,
                min_mask_region_area=args.min_mask_region_area,
                max_keep=args.max_keep,
                dedup_iou_thresh=args.dedup_iou_thresh,
                output_base=output_base,
                skip_existing=args.skip_existing,
                mean_subtract=args.mean_subtract,
                amg_params=amg_params,
            )
            if n_masks is not None:
                total_masks += n_masks
        except Exception as e:
            if rank == 0:
                print(f"\nError processing {img_path}: {e}")
            continue

    # ---- Aggregate ----
    if world_size > 1:
        dist.barrier()
        processed_t = torch.tensor([len(image_paths)], device=device)
        masks_t = torch.tensor([total_masks], device=device)
        dist.all_reduce(processed_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(masks_t, op=dist.ReduceOp.SUM)
        total_processed = processed_t.item()
        total_masks_all = masks_t.item()
    else:
        total_processed = len(image_paths)
        total_masks_all = total_masks

    if rank == 0:
        avg = total_masks_all / max(1, total_processed)
        print(f"\nDone! Processed {total_processed} images, {total_masks_all} masks ({avg:.1f}/img)")
        print(f"Backbone: {args.backbone}, IoU thresh: {args.dedup_iou_thresh}")
        print(f"Saved to {args.output_dir}/")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
