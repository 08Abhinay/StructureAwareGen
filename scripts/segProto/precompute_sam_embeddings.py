#!/usr/bin/env python3
"""
Pre-compute SAM (Segment Anything Model) segmentation masks and embeddings.
Saves masks, embeddings, and metadata as .npz and .json files.

Usage (multi-GPU with DDP):
    torchrun --nproc_per_node=4 scripts/segProto/precompute_sam_embeddings.py \
        --image_dir /path/to/images \
        --output_dir /path/to/output \
        --checkpoint /path/to/sam_checkpoint.pth \
        --model_type vit_b \
        --max_keep 250 \
        --subset_fraction 0.4
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from scipy import ndimage as ndi
from pathlib import Path

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def setup_ddp():
    """Initialize DDP for multi-GPU processing."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # Debug: Print from all ranks before init
        print(f"[Pre-init] Rank {rank}/{world_size}, Local rank {local_rank}, Hostname: {os.uname().nodename}")
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        # Debug: Confirm initialization
        print(f"[Post-init] Rank {rank} successfully initialized on device {device}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Single GPU mode] Using device {device}")
    
    return rank, world_size, local_rank, device


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


def validate_npz_file(npz_path, min_size_kb=1):
    """
    Validate that a .npz file is complete and not corrupted.
    
    Returns:
        True if valid, False if corrupted/incomplete
    """
    try:
        # Check file size (corrupted files are often 0 bytes or tiny)
        file_size = npz_path.stat().st_size
        if file_size < min_size_kb * 1024:
            return False
        
        # Try to load and check required keys
        data = np.load(npz_path)
        required_keys = ['packed', 'shape', 'scores', 'label_map']
        
        for key in required_keys:
            if key not in data:
                data.close()
                return False
        
        # Validate shapes are reasonable
        shape = data['shape']
        if len(shape) != 3 or shape[0] > 1000:  # More than 1000 masks is suspicious
            data.close()
            return False
        
        data.close()
        return True
        
    except Exception:
        # Any error during loading = corrupted file
        return False


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
    skip_existing=True,
    amg_params=None
):
    """Process a single image and save SAM embeddings and metadata."""
    
    npz_path = output_base.parent / "masks_npz" / f"{output_base.name}.npz"
    
    # Skip if exists AND is valid
    if skip_existing and npz_path.exists():
        if validate_npz_file(npz_path):
            return None
        else:
            # File exists but is corrupted - reprocess
            print(f"Warning: Corrupted file detected, reprocessing: {npz_path.name}")
    
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
    
    # 6) Save metadata JSON
    json_path = output_base.parent / "meta" / f"{output_base.name}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build metadata for each mask
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
    
    # Save JSON
    metadata = {
        "image": str(image_path),
        "device": str(device),
        "num_masks": int(N),
        "amg_params": amg_params or {},
        "masks": meta_masks,
    }
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
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
    
    # DDP and subset support
    parser.add_argument('--subset_fraction', type=float, default=1.0,
                        help='Fraction of dataset to process (0.4 = 40%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deterministic subset selection')
    
    args = parser.parse_args()
    
    # Setup DDP (handles device automatically)
    rank, world_size, local_rank, device = setup_ddp()
    
    if rank == 0:
        print(f"Using {world_size} GPUs for parallel processing")
        print(f"Device (rank {rank}): {device}")
    
    # Check checkpoint
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Load SAM model
    if rank == 0:
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
    if rank == 0:
        print(f"Scanning for images in {args.image_dir}")
    
    # Check if directory exists
    if not os.path.isdir(args.image_dir):
        if rank == 0:
            print(f"ERROR: Directory does not exist: {args.image_dir}")
        return
    
    if rank == 0:
        print(f"Directory exists. Searching for images recursively...")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
    image_paths = []
    
    for ext in image_extensions:
        pattern = os.path.join(args.image_dir, '**', ext)
        found = glob.glob(pattern, recursive=True)
        if found and rank == 0:
            print(f"  Found {len(found)} files matching {ext}")
        image_paths.extend(found)
    
    image_paths = sorted(image_paths)
    total_images = len(image_paths)
    
    if rank == 0:
        print(f"Total images found: {total_images}")
    
    if total_images == 0:
        if rank == 0:
            print("\nNo images found! Debugging info:")
            print(f"  Directory: {args.image_dir}")
            print(f"  Directory exists: {os.path.isdir(args.image_dir)}")
            print(f"  Tried patterns: {image_extensions}")
        return
    
    # Apply subset selection (deterministic across all ranks)
    if args.subset_fraction < 1.0:
        rng = np.random.RandomState(args.seed)
        n_total = len(image_paths)
        n_subset = int(n_total * args.subset_fraction)
        indices = rng.choice(n_total, size=n_subset, replace=False)
        indices = sorted(indices)  # Keep sorted for reproducibility
        image_paths = [image_paths[i] for i in indices]
        
        if rank == 0:
            print(f"Subset selection: {args.subset_fraction:.1%} of dataset ({len(image_paths)} images)")
    
    # Distribute images across GPUs (strided for load balancing)
    image_paths = image_paths[rank::world_size]
    
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    
    # Print workload for ALL ranks to verify distribution
    print(f"[Rank {rank}] Assigned {len(image_paths)} images to process")
    
    if len(image_paths) == 0:
        print(f"[Rank {rank}] No images to process on this GPU.")
        return
    
    # Filter existing if skip_existing
    if args.skip_existing:
        filtered_paths = []
        skipped_valid = 0
        skipped_corrupted = 0
        
        for img_path in image_paths:
            output_base = get_output_path(img_path, args.image_dir, args.output_dir)
            npz_path = output_base.parent / "masks_npz" / f"{output_base.name}.npz"
            
            if npz_path.exists():
                if validate_npz_file(npz_path):
                    skipped_valid += 1
                    continue  # Skip valid files
                else:
                    skipped_corrupted += 1
                    # Will reprocess corrupted files
            
            filtered_paths.append(img_path)
        
        if rank == 0:
            print(f"Skipped {skipped_valid} valid existing embeddings")
        if skipped_corrupted > 0 and rank == 0:
            print(f"Found {skipped_corrupted} corrupted files - will reprocess")
        
        image_paths = filtered_paths
        
        if len(image_paths) == 0:
            if rank == 0:
                print("All embeddings already exist on this GPU!")
            return
    
    # Process images
    print(f"[Rank {rank}] Starting processing of {len(image_paths)} images...")
    total_masks = 0
    
    # AMG parameters for metadata
    amg_params = {
        "points_per_side": args.points_per_side,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale,
        "min_mask_region_area_post": args.min_mask_region_area,
        "dedup_iou_thresh": args.dedup_iou_thresh,
        "max_keep": args.max_keep,
    }
    
    # Show progress bar for all ranks with position to avoid overlap
    pbar = tqdm(
        image_paths, 
        desc=f"Rank {rank}", 
        position=rank, 
        leave=True,
        dynamic_ncols=True
    )
    
    for img_path in pbar:
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
                args.skip_existing,
                amg_params
            )
            
            if n_masks is not None:
                total_masks += n_masks
                
        except Exception as e:
            if rank == 0:
                print(f"\nError processing {img_path}: {e}")
            continue
    
    # Aggregate statistics across GPUs
    if world_size > 1:
        dist.barrier()
        
        processed_tensor = torch.tensor([len(image_paths)], device=device)
        masks_tensor = torch.tensor([total_masks], device=device)
        
        dist.all_reduce(processed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(masks_tensor, op=dist.ReduceOp.SUM)
        
        total_processed = processed_tensor.item()
        total_masks_all = masks_tensor.item()
    else:
        total_processed = len(image_paths)
        total_masks_all = total_masks
    
    if rank == 0:
        print(f"\n✓ Successfully processed {total_processed} images")
        print(f"✓ Total masks generated: {total_masks_all}")
        print(f"✓ Average masks per image: {total_masks_all / total_processed:.1f}")
        print(f"✓ Saved to {args.output_dir}/masks_npz/")
    
    # Cleanup DDP
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
