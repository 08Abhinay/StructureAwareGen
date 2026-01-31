#!/usr/bin/env python3
"""
Pre-compute SAM (Segment Anything Model) segmentation masks and embeddings.
Saves masks, embeddings, and metadata as .npz and .json files.

Usage:
    python scripts/segProto/precompute_sam_embeddings.py \
        --image_dir /path/to/images \
        --output_dir /path/to/output \
        --checkpoint /path/to/sam_checkpoint.pth \
        --model_type vit_b \
        --max_keep 250
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy import ndimage as ndi
from pathlib import Path

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def load_image_rgb(path: str) -> np.ndarray:
    """Load image and convert to RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def mask_stats(mask_bool: np.ndarray):
    """Compute statistics for a binary mask."""
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


def bbox_xywh_from_mask(mask_bool: np.ndarray):
    """Extract bounding box in xywh format from mask."""
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)]


def _remove_small_islands(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove small disconnected regions from mask."""
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
    """Fill small holes in mask."""
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
    """Apply SAM-like filtering to mask."""
    mask = mask.astype(bool)
    if min_area <= 0:
        return mask
    mask = _fill_small_holes(mask, min_area)
    mask = _remove_small_islands(mask, min_area)
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / max(1, union))


def get_output_path(image_path, image_dir, output_dir):
    """Get corresponding output path maintaining directory structure."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    image_path = Path(image_path)
    
    rel_path = image_path.relative_to(image_dir)
    output_base = output_dir / rel_path.parent / rel_path.stem
    
    return output_base


def process_image(
    image_path,
    mask_generator,
    predictor,
    device,
    min_mask_region_area,
    max_keep,
    dedup_iou_thresh,
    save_mask_emb,
    output_base,
    skip_existing=True
):
    """Process a single image and save SAM embeddings."""
    
    npz_path = output_base.parent / "masks_npz" / f"{output_base.name}.npz"
    
    if skip_existing and npz_path.exists():
        return None
    
    img_rgb = load_image_rgb(image_path)
    H, W, _ = img_rgb.shape

    # 1) Generate proposals
    amg = mask_generator.generate(img_rgb)

    # 2) Compute image embedding once for mask embeddings
    if save_mask_emb:
        predictor.set_image(img_rgb)
        feat = predictor.get_image_embedding()
        hf, wf = feat.shape[-2], feat.shape[-1]
    else:
        feat, hf, wf = None, None, None

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

        # per-mask embedding from SAM image encoder
        emb = None
        if save_mask_emb:
            mask_t = torch.from_numpy(seg[None, None].astype(np.float32)).to(device)
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

    # 3) Sort and deduplicate (greedy IoU)
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

    # 4) Build final arrays
    N = len(kept)
    if N == 0:
        masks = np.zeros((0, H, W), dtype=np.bool_)
        scores = np.zeros((0,), dtype=np.float32)
        embs = None
        label_map = -np.ones((H, W), dtype=np.int32)
    else:
        masks = np.stack([k["seg"] for k in kept], axis=0).astype(np.bool_)
        scores = np.asarray([k["score"] for k in kept], dtype=np.float32)

        if save_mask_emb:
            embs = np.stack([k["emb"] for k in kept], axis=0)
        else:
            embs = None

        # Non-overlapping label map
        label_map = -np.ones((H, W), dtype=np.int32)
        occupied = np.zeros((H, W), dtype=bool)
        order = np.argsort(-scores)
        for new_id in order:
            pix = masks[new_id] & (~occupied)
            if pix.sum() == 0:
                continue
            label_map[pix] = int(new_id)
            occupied[pix] = True

    # 5) Save compressed masks and embeddings
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(masks.reshape(masks.shape[0], -1), axis=1) if masks.shape[0] > 0 else np.zeros((0, 0), dtype=np.uint8)
    np.savez_compressed(
        npz_path,
        packed=packed,
        shape=np.array(masks.shape, dtype=np.int32),
        scores=scores,
        label_map=label_map,
        emb=embs,
    )
    
    return N


def main():
    parser = argparse.ArgumentParser(description='Pre-compute SAM embeddings')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save embeddings')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SAM checkpoint (.pth file)')
    parser.add_argument('--model_type', type=str, default='vit_b',
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        help='SAM model type')
    
    # AMG parameters
    parser.add_argument('--points_per_side', type=int, default=64,
                        help='Number of points per side for mask generation')
    parser.add_argument('--pred_iou_thresh', type=float, default=0.80,
                        help='Predicted IoU threshold')
    parser.add_argument('--stability_score_thresh', type=float, default=0.85,
                        help='Stability score threshold')
    parser.add_argument('--box_nms_thresh', type=float, default=0.70,
                        help='Box NMS threshold')
    parser.add_argument('--crop_n_layers', type=int, default=1,
                        help='Number of crop layers')
    parser.add_argument('--crop_overlap_ratio', type=float, default=0.35,
                        help='Crop overlap ratio')
    parser.add_argument('--crop_n_points_downscale', type=int, default=2,
                        help='Crop points downscale factor')
    
    # Post-processing parameters
    parser.add_argument('--min_mask_region_area', type=int, default=300,
                        help='Minimum mask region area (0 to disable)')
    parser.add_argument('--max_keep', type=int, default=250,
                        help='Maximum number of masks to keep per image')
    parser.add_argument('--dedup_iou_thresh', type=float, default=0.90,
                        help='IoU threshold for deduplication')
    
    # Options
    parser.add_argument('--save_mask_emb', action='store_true', default=True,
                        help='Save per-mask SAM embeddings')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--batch_process', action='store_true', default=False,
                        help='Process images in batch (not implemented yet)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip images with existing embeddings')
    parser.add_argument('--max_images', type=int, default=-1,
                        help='Maximum number of images to process (-1 for all)')
    
    # Chunking support for parallel jobs
    parser.add_argument('--start_index', type=int, default=0,
                        help='Start index for processing (for parallel jobs)')
    parser.add_argument('--end_index', type=int, default=-1,
                        help='End index for processing (for parallel jobs, -1 for all)')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available() and 'cuda' in args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
        print("Warning: CUDA not available, using CPU (this will be slow)")
    
    print(f"Using device: {device}")
    
    # Check checkpoint
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Load SAM model
    print(f"Loading SAM model ({args.model_type})...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
    sam.eval()
    
    # Create predictor and mask generator
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_overlap_ratio=args.crop_overlap_ratio,
        crop_n_points_downscale_factor=args.crop_n_points_downscale,
        min_mask_region_area=0,  # We do post-processing manually
    )
    
    # Get image paths
    print(f"Scanning for images in {args.image_dir}")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, '**', ext), recursive=True))
    
    image_paths = sorted(image_paths)
    
    # Apply chunking for parallel jobs
    total_images = len(image_paths)
    start_idx = max(0, args.start_index)
    end_idx = args.end_index if args.end_index > 0 else total_images
    end_idx = min(end_idx, total_images)
    
    if start_idx > 0 or end_idx < total_images:
        print(f"Chunking: processing images [{start_idx}:{end_idx}] out of {total_images}")
        image_paths = image_paths[start_idx:end_idx]
    
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    
    print(f"Found {len(image_paths)} images to process")
    
    if len(image_paths) == 0:
        print("No images found! Check your image_dir path.")
        return
    
    # Filter existing if skip_existing
    if args.skip_existing:
        filtered_paths = []
        for img_path in image_paths:
            output_base = get_output_path(img_path, args.image_dir, args.output_dir)
            npz_path = output_base.parent / "masks_npz" / f"{output_base.name}.npz"
            if not npz_path.exists():
                filtered_paths.append(img_path)
        
        print(f"Skipping {len(image_paths) - len(filtered_paths)} existing embeddings")
        image_paths = filtered_paths
        
        if len(image_paths) == 0:
            print("All embeddings already exist!")
            return
    
    # Process images
    print(f"\nProcessing {len(image_paths)} images...")
    total_masks = 0
    
    for img_path in tqdm(image_paths, desc="Generating SAM masks"):
        output_base = get_output_path(img_path, args.image_dir, args.output_dir)
        
        try:
            n_masks = process_image(
                img_path,
                mask_generator,
                predictor,
                device,
                args.min_mask_region_area,
                args.max_keep,
                args.dedup_iou_thresh,
                args.save_mask_emb,
                output_base,
                args.skip_existing
            )
            
            if n_masks is not None:
                total_masks += n_masks
                
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue
    
    print(f"\n✓ Successfully processed {len(image_paths)} images")
    print(f"✓ Total masks generated: {total_masks}")
    print(f"✓ Average masks per image: {total_masks / len(image_paths):.1f}")
    print(f"✓ Saved to {args.output_dir}/masks_npz/")


if __name__ == '__main__':
    main()
