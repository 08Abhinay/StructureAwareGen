#!/usr/bin/env python3
"""
Pre-compute DINOv2 CLS + REN region token embeddings → H5 shards.

This script:
  1. Loads DINOv2 ViT-L/14 backbone (frozen, from torch.hub).
  2. Loads the pretrained REN RegionEncoder checkpoint.
  3. For each image:
     a. Extracts DINOv2 feature maps + CLS token (single forward pass).
     b. Generates SLIC (or grid) prompts.
     c. Runs RegionEncoder → TokenAggregator to get learned region tokens.
     d. Writes results to per-rank H5 shard.

No SAM dependency — prompts come from SLIC superpixels or uniform grid.

H5 shard schema (flat, compatible with merge_h5_shards.py):
    emb:         (N_total_regions, 1024)  float32 — REN pred_tokens (backbone space)
    scores:      (N_total_regions,)       float32 — attention mass per region (sorting proxy)
    offsets:     (N_images,)              int64   — start offset per image
    n_segments:  (N_images,)              int32   — region count per image
    class_ids:   (N_images,)              int32   — class label per image
    names:       (N_images,)              string  — sample basename
    cls_emb:     (N_images, 1024)         float32 — DINOv2 CLS token per image

Usage:
    python3 precompute_ren_embeddings_h5.py \\
        --image_dir /path/to/imagenet/train \\
        --output_dir /path/to/shards/ \\
        --ren_config configs/ren_dinov2_vitl14.yaml \\
        --ren_checkpoint /path/to/ren-dinov2-vitl14/checkpoint.pth
"""

import gc
import os
import sys
import glob
import yaml
import argparse
import numpy as np
import h5py
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from ren_model import (
    DINOv2Extractor, RegionEncoder, TokenAggregator, SLICPrompter,
)


# ---------------------------------------------------------------------------
# DDP helpers
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
        print(f"[Post-init] Rank {rank} initialized on device {device}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Single GPU mode] Using device {device}")
    return rank, world_size, local_rank, device


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_and_resize(image_path, resolution):
    """Load image, resize to resolution, return [1, 3, H, W] float tensor."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((resolution, resolution), Image.BICUBIC)
    import torchvision.transforms as T
    tensor = T.ToTensor()(img)  # [3, H, W], 0-1 range
    return tensor.unsqueeze(0)  # [1, 3, H, W]


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(
    image_path,
    dinov2_extractor,
    region_encoder,
    token_aggregator,
    slic_prompter,
    device,
    grid_size,
    image_resolution,
    use_slic,
):
    """
    Process a single image through DINOv2 + REN pipeline.

    Returns dict with:
        emb:        (N_regions, 1024) float32 — REN aggregated pred_tokens
        scores:     (N_regions,) float32      — attention mass per region
        cls_emb:    (1024,) float32           — z-score normalized DINOv2 CLS
        n_segments: int
    """
    # Load and resize
    img_tensor = load_and_resize(image_path, image_resolution).to(device)

    # Single backbone forward: feature maps + CLS
    feature_maps, cls_tokens = dinov2_extractor.extract(img_tensor)
    # feature_maps: [1, 1024, 37, 37], cls_tokens: [1, 1024]

    cls_raw = cls_tokens[0]  # [1024]

    # Z-score normalize CLS
    with torch.no_grad():
        cls_mean = cls_raw.mean()
        cls_std = cls_raw.std().clamp(min=1e-6)
        cls_normed = (cls_raw - cls_mean) / cls_std

    # SLIC prompts (on CPU, per-image)
    num_segments = grid_size * grid_size
    prompts = slic_prompter(img_tensor, num_segments, use_slic=use_slic)

    # REN forward
    with torch.no_grad():
        ren_out = region_encoder(feature_maps, prompts)

    # Token aggregation
    with torch.no_grad():
        agg_out = token_aggregator(
            ren_out['pred_tokens'], ren_out['proj_tokens'],
            ren_out['attn_scores'][-1], prompts,
        )

    region_tokens = agg_out['aggregated_pred_tokens'][0]  # [N, 1024]
    agg_attn = agg_out['aggregated_attn_scores'][0]       # [heads, N, hw]

    # Attention mass: softmax the raw scores, mean across heads,
    # sum across spatial → [N] (proportion of spatial attention per region)
    agg_attn_weights = F.softmax(agg_attn, dim=-1)   # [heads, N, hw]
    attn_mass = agg_attn_weights.mean(dim=0).sum(dim=-1)  # [N]

    N = region_tokens.shape[0]
    if N == 0:
        return {
            "emb": np.zeros((0, 1024), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "cls_emb": cls_normed.cpu().numpy().astype(np.float32),
            "n_segments": 0,
        }

    return {
        "emb": region_tokens.cpu().numpy().astype(np.float32),
        "scores": attn_mass.cpu().numpy().astype(np.float32),
        "cls_emb": cls_normed.cpu().numpy().astype(np.float32),
        "n_segments": N,
    }


# ---------------------------------------------------------------------------
# H5 Shard Writer (adapted from precompute_region_embeddings_h5.py)
# ---------------------------------------------------------------------------

class H5ShardWriter:
    """
    Append one image at a time to a resizable H5 shard.
    Supports crash-safe resume via existing_names tracking.
    """

    EMB_DIM = 1024

    def __init__(self, output_path):
        self.output_path = output_path
        D = self.EMB_DIM
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            self.f = h5py.File(output_path, "a")
            self.n_samples = int(self.f.attrs.get("total_samples", 0))
            self.seg_cursor = int(self.f.attrs.get("total_segments", 0))
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
            self.f = h5py.File(output_path, "w")
            self.n_samples = 0
            self.seg_cursor = 0
            self.existing_names = set()
            self._create_datasets(D)
            self.f.attrs["total_samples"] = 0
            self.f.attrs["total_segments"] = 0
            self.f.attrs["emb_dim"] = D
            self.f.attrs["emb_dtype"] = "float32"
            self.f.attrs["source"] = "ren_dinov2_vitl14"
            self.f.flush()

    def _create_datasets(self, D):
        f = self.f
        # Flat segment arrays
        f.create_dataset("emb", shape=(0, D), maxshape=(None, D),
                         dtype="float32", chunks=(512, D))
        f.create_dataset("scores", shape=(0,), maxshape=(None,),
                         dtype="float32", chunks=(4096,))
        # Per-sample arrays
        f.create_dataset("offsets", shape=(0,), maxshape=(None,),
                         dtype="int64", chunks=(4096,))
        f.create_dataset("n_segments", shape=(0,), maxshape=(None,),
                         dtype="int32", chunks=(4096,))
        f.create_dataset("class_ids", shape=(0,), maxshape=(None,),
                         dtype="int32", chunks=(4096,))
        f.create_dataset("names", shape=(0,), maxshape=(None,),
                         dtype=h5py.string_dtype())
        f.create_dataset("cls_emb", shape=(0, D), maxshape=(None, D),
                         dtype="float32", chunks=(512, D))

    def has_image(self, name):
        return name in self.existing_names

    def append(self, result):
        f = self.f
        n = result["n_segments"]
        i = self.n_samples
        c = self.seg_cursor

        if n > 0:
            new_seg = c + n
            f["emb"].resize(new_seg, axis=0)
            f["scores"].resize(new_seg, axis=0)
            f["emb"][c:new_seg] = result["emb"]
            f["scores"][c:new_seg] = result["scores"]

        f["offsets"].resize(i + 1, axis=0)
        f["n_segments"].resize(i + 1, axis=0)
        f["class_ids"].resize(i + 1, axis=0)
        f["names"].resize(i + 1, axis=0)
        f["cls_emb"].resize(i + 1, axis=0)

        f["offsets"][i] = c
        f["n_segments"][i] = n
        f["class_ids"][i] = result["class_id"]
        f["names"][i] = result["name"]
        f["cls_emb"][i] = result["cls_emb"]

        self.seg_cursor = c + n
        self.n_samples = i + 1
        self.existing_names.add(result["name"])

        f.attrs["total_samples"] = self.n_samples
        f.attrs["total_segments"] = self.seg_cursor
        f.flush()

    def close(self):
        if self.f:
            self.f.attrs["total_samples"] = self.n_samples
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
        description="Pre-compute DINOv2 CLS + REN region tokens → H5 shard"
    )

    # Paths
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for H5 shard files")

    # REN config & checkpoint
    parser.add_argument("--ren_config", type=str, required=True,
                        help="Path to REN YAML config")
    parser.add_argument("--ren_checkpoint", type=str, required=True,
                        help="Path to REN checkpoint (.pth)")

    # Extraction parameters
    parser.add_argument("--image_resolution", type=int, default=518,
                        help="Input image resolution (must match REN config)")
    parser.add_argument("--grid_size", type=int, default=37,
                        help="SLIC/grid prompt grid size (37 → 1369 prompts)")
    parser.add_argument("--use_slic", action="store_true", default=True,
                        help="Use SLIC superpixels for prompts")
    parser.add_argument("--no_slic", dest="use_slic", action="store_false",
                        help="Use uniform grid prompts instead of SLIC")
    parser.add_argument("--merge_similarity", type=float, default=0.975,
                        help="Cosine similarity threshold for token merging")

    # Cache
    parser.add_argument("--torch_home", type=str, default=None,
                        help="TORCH_HOME for caching DINOv2 weights")

    # Runtime
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--subset_fraction", type=float, default=1.0,
                        help="Fraction of images per class (stratified)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ---- DDP setup ----
    rank, world_size, local_rank, device = setup_ddp()

    if rank == 0:
        print(f"Using {world_size} GPUs for parallel processing")
        print(f"REN config: {args.ren_config}")
        print(f"REN checkpoint: {args.ren_checkpoint}")

    # ---- Load REN config ----
    with open(args.ren_config, "r") as f:
        config = yaml.safe_load(f)
    ren_config = config["ren"]

    # ---- Load DINOv2 backbone ----
    if rank == 0:
        print("Loading DINOv2 ViT-L/14 backbone...")
    dinov2_extractor = DINOv2Extractor(device, torch_home=args.torch_home)

    # ---- Load REN RegionEncoder ----
    if rank == 0:
        print("Loading REN RegionEncoder...")
    region_encoder = RegionEncoder(ren_config).to(device).eval()
    for p in region_encoder.parameters():
        p.requires_grad_(False)

    checkpoint = torch.load(args.ren_checkpoint, map_location=device)
    region_encoder.load_state_dict(checkpoint['region_encoder_state'])
    if rank == 0:
        print(f"  Loaded REN checkpoint from epoch {checkpoint.get('epoch', '?')}, "
              f"iter {checkpoint.get('iter_count', '?')}")

    # ---- Create TokenAggregator ----
    token_aggregator = TokenAggregator(
        merge_similarity=args.merge_similarity)

    # ---- Create SLIC prompter ----
    slic_prompter = SLICPrompter(image_resolution=args.image_resolution)

    # ---- Free fragmented CUDA memory ----
    gc.collect()
    torch.cuda.empty_cache()
    if rank == 0:
        free_mem = torch.cuda.mem_get_info(device)[0] / (1024**3)
        print(f"GPU memory after model loading: {free_mem:.1f} GiB free")

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
        end = (len(image_paths) if args.end_index < 0
               else min(len(image_paths), int(args.end_index)))
        if end < start:
            end = start
        image_paths = image_paths[start:end]
        if rank == 0:
            print(f"Applied index slice: [{start}:{end}] → {len(image_paths)} images")

    # ---- Stratified subset selection ----
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

    # ---- DDP sharding (round-robin) ----
    image_paths = image_paths[rank::world_size]
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    print(f"[Rank {rank}] Assigned {len(image_paths)} images")

    if len(image_paths) == 0:
        if rank == 0:
            print("No images to process!")
        return

    # ---- Open H5 shard ----
    shard_name = f"region_ren_shard_r{rank}.h5"
    shard_path = os.path.join(args.output_dir, shard_name)
    writer = H5ShardWriter(shard_path)

    # ---- Process images ----
    n_processed = 0
    n_skipped = 0
    total_segs_written = 0
    errors = []

    pbar = tqdm(image_paths, desc=f"Rank {rank}", position=rank,
                leave=True, dynamic_ncols=True)
    for img_path in pbar:
        try:
            rel = Path(img_path).relative_to(args.image_dir)
            class_id_str = rel.parts[0]
            name = rel.stem

            if args.skip_existing and writer.has_image(name):
                n_skipped += 1
                continue

            try:
                class_id = int(class_id_str)
            except ValueError:
                class_id = hash(class_id_str) % (2**31)

            result = process_image(
                image_path=img_path,
                dinov2_extractor=dinov2_extractor,
                region_encoder=region_encoder,
                token_aggregator=token_aggregator,
                slic_prompter=slic_prompter,
                device=device,
                grid_size=args.grid_size,
                image_resolution=args.image_resolution,
                use_slic=args.use_slic,
            )

            result["class_id"] = class_id
            result["name"] = name

            writer.append(result)
            n_processed += 1
            total_segs_written += result["n_segments"]

            pbar.set_postfix(
                regs=result["n_segments"],
                total=total_segs_written,
                saved=n_processed,
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                gc.collect()
                torch.cuda.empty_cache()
                errors.append((img_path, f"OOM: {e}"))
                print(f"\n  OOM (cleared cache): {img_path}")
            else:
                errors.append((img_path, str(e)))
                print(f"\n  Error: {img_path}: {e}")
            continue
        except Exception as e:
            errors.append((img_path, str(e)))
            print(f"\n  Error: {img_path}: {e}")
            continue

        if n_processed % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ---- Close H5 shard ----
    writer.close()
    if n_skipped > 0:
        print(f"[Rank {rank}] Skipped {n_skipped} already-processed images")

    # ---- Aggregate stats across ranks ----
    if world_size > 1:
        dist.barrier()
        processed_t = torch.tensor([n_processed], device=device)
        segs_t = torch.tensor([total_segs_written], device=device)
        dist.all_reduce(processed_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(segs_t, op=dist.ReduceOp.SUM)
        total_processed = processed_t.item()
        total_segs_all = segs_t.item()
    else:
        total_processed = n_processed
        total_segs_all = total_segs_written

    if rank == 0:
        avg = total_segs_all / max(1, total_processed)
        print(f"\nDone! Processed {total_processed} images, "
              f"{total_segs_all} regions ({avg:.1f}/img)")
        print(f"Shards saved to {args.output_dir}/")

    if errors:
        print(f"[Rank {rank}] Errors: {len(errors)}")
        for p, e in errors[:5]:
            print(f"  {p}: {e}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
