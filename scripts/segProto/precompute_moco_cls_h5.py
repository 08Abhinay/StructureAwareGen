#!/usr/bin/env python3
"""
Pre-compute MoCo v3 CLS embeddings (through MLP head) for all ImageNet images.
Writes flat H5 shards per DDP rank, then rank 0 merges into a single file.

Key difference from precompute_region_embeddings_h5.py:
  - NO SAM masks needed — this is CLS-only extraction.
  - Uses MLP projection head (3-layer MLP: 1024→4096→4096→256).
  - Z-score normalizes per sample (matching ddpm.py's ijepa_emb path).
  - Output is 256-dim, ready for the cls_emb path in ddpm.py.

Output schema (flat H5, matching ijepa_emb_flat.h5):
    emb:        (N, 256)   float32 — z-score normalized MoCo CLS embeddings
    class_ids:  (N,)       int32   — class label per image
    names:      (N,)       string  — sample basename (no extension)

Lookup JSON:
    { "class_id/name": row_index, ... }

Usage (single GPU):
    python3 precompute_moco_cls_h5.py \
        --image_dir /path/to/imagenet/train \
        --output_dir /path/to/shards/ \
        --merged_h5  /path/to/moco_cls_flat.h5 \
        --merged_json /path/to/moco_cls_lookup.json

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 precompute_moco_cls_h5.py \
        --image_dir /path/to/imagenet/train \
        --output_dir /path/to/shards/ \
        --merged_h5  /path/to/moco_cls_flat.h5 \
        --merged_json /path/to/moco_cls_lookup.json
"""

import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import h5py
import torch
import torch.distributed as dist
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Add SEG-RDM to path for MoCo model loading
SCRIPT_DIR = Path(__file__).resolve().parent
SEG_RDM_DIR = SCRIPT_DIR.parent / "SEG-RDM"
if str(SEG_RDM_DIR) not in sys.path:
    sys.path.insert(0, str(SEG_RDM_DIR))


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialize DDP; returns (rank, world_size, local_rank, device)."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[DDP] Rank {rank}/{world_size}, local_rank {local_rank}, "
              f"host={os.uname().nodename}, device={device}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Single GPU] device={device}")
    return rank, world_size, local_rank, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Simple ImageNet dataset (returns image tensor + metadata)
# ---------------------------------------------------------------------------

class ImageNetPathDataset(Dataset):
    """Loads ImageNet images and returns (tensor, class_id_str, basename)."""

    def __init__(self, image_dir: str, transform):
        self.transform = transform
        self.samples = []  # [(path, class_id_str, basename), ...]

        for class_dir in sorted(os.listdir(image_dir)):
            class_path = os.path.join(image_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                    continue
                basename = os.path.splitext(fname)[0]
                self.samples.append((
                    os.path.join(class_path, fname),
                    class_dir,
                    basename,
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_id, basename = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, class_id, basename


# ---------------------------------------------------------------------------
# MoCo v3 model loading
# ---------------------------------------------------------------------------

def load_moco_model(checkpoint_path: str, device: torch.device):
    """Load MoCo v3 ViT-L with MLP head, return model."""
    from rdm.pretrained_enc.models_pretrained_enc import (
        mocov3_vit_large, load_pretrained_moco,
    )
    model = mocov3_vit_large(proj_dim=256)
    model = load_pretrained_moco(model, checkpoint_path)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_shard(
    model,
    dataloader,
    device: torch.device,
    rank: int,
    output_path: str,
    total_samples: int,
):
    """
    Run forward pass on all batches, project CLS through MLP head,
    z-score normalise per sample, and write to H5 shard.
    """
    EMB_DIM = 256
    CHUNK = 8192

    all_embs = []
    all_class_ids = []
    all_names = []
    n_done = 0
    t0 = time.time()

    for batch_idx, (images, class_ids, basenames) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = model.forward_features(images)  # [B, 1+N, 1024]
            cls_raw = out[:, 0, :].float()         # [B, 1024]
            cls_proj = model.head(cls_raw).float()  # [B, 256]

        # Per-sample z-score normalisation
        mu = cls_proj.mean(dim=1, keepdim=True)
        sigma = cls_proj.std(dim=1, keepdim=True).clamp(min=1e-6)
        cls_norm = ((cls_proj - mu) / sigma).cpu().numpy()

        all_embs.append(cls_norm)
        all_class_ids.extend(class_ids)
        all_names.extend(basenames)
        n_done += len(class_ids)

        if (batch_idx + 1) % 50 == 0 or n_done >= total_samples:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-3)
            print(f"  [Rank {rank}] {n_done}/{total_samples} "
                  f"({100*n_done/total_samples:.1f}%) "
                  f"[{elapsed:.0f}s, {rate:.0f} img/s]")

    # Concatenate
    embs = np.concatenate(all_embs, axis=0).astype(np.float32)  # (N, 256)
    n = embs.shape[0]

    # Try to convert class_ids to int (ImageNet uses numeric folder names)
    int_class_ids = []
    for cid in all_class_ids:
        try:
            int_class_ids.append(int(cid))
        except ValueError:
            int_class_ids.append(hash(cid) % (2**31))
    class_ids_arr = np.array(int_class_ids, dtype=np.int32)

    # Write H5 shard
    print(f"  [Rank {rank}] Writing {n} embeddings to {output_path} ...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('emb', data=embs, chunks=(min(CHUNK, n), EMB_DIM),
                         compression='gzip', compression_opts=4)
        f.create_dataset('class_ids', data=class_ids_arr)
        dt = h5py.string_dtype()
        f.create_dataset('names', data=np.array(all_names, dtype=object), dtype=dt)
        f.attrs['total_samples'] = n
        f.attrs['emb_dim'] = EMB_DIM
        f.attrs['rank'] = rank
        # Also store string class IDs for lookup
        f.create_dataset('class_id_strs', data=np.array(all_class_ids, dtype=object), dtype=dt)

    elapsed = time.time() - t0
    print(f"  [Rank {rank}] Done: {n} samples in {elapsed:.1f}s "
          f"({n/max(elapsed,1):.0f} img/s)")
    return n


# ---------------------------------------------------------------------------
# Merge shards → single flat H5 + JSON lookup
# ---------------------------------------------------------------------------

def merge_shards(shard_dir: str, shard_pattern: str,
                 merged_h5: str, merged_json: str):
    """Merge per-rank H5 shards into one flat file + JSON lookup."""
    print("\n" + "=" * 60)
    print("Merging shards...")
    print("=" * 60)

    shard_paths = sorted(glob.glob(os.path.join(shard_dir, shard_pattern)))
    print(f"  Found {len(shard_paths)} shards")

    if len(shard_paths) == 0:
        print("  ERROR: no shards found!")
        return

    # Scan sizes
    total = 0
    emb_dim = None
    for sp in shard_paths:
        with h5py.File(sp, 'r') as f:
            n = f['emb'].shape[0]
            d = f['emb'].shape[1]
            total += n
            if emb_dim is None:
                emb_dim = d
    print(f"  Total samples: {total:,}, emb_dim: {emb_dim}")

    # Write merged file
    CHUNK = 8192
    os.makedirs(os.path.dirname(merged_h5), exist_ok=True)
    lookup = {}
    offset = 0

    with h5py.File(merged_h5, 'w') as out:
        emb_ds = out.create_dataset(
            'emb', shape=(total, emb_dim), dtype='float32',
            chunks=(min(CHUNK, total), emb_dim),
            compression='gzip', compression_opts=4,
        )
        cid_ds = out.create_dataset('class_ids', shape=(total,), dtype='int32')
        dt = h5py.string_dtype()
        name_ds = out.create_dataset('names', shape=(total,), dtype=dt)

        for sp in shard_paths:
            with h5py.File(sp, 'r') as f:
                n = f['emb'].shape[0]
                emb_ds[offset:offset + n] = f['emb'][:]
                cid_ds[offset:offset + n] = f['class_ids'][:]

                names = [x.decode() if isinstance(x, bytes) else str(x)
                         for x in f['names'][:]]
                class_strs = [x.decode() if isinstance(x, bytes) else str(x)
                              for x in f['class_id_strs'][:]]

                for i in range(n):
                    name_ds[offset + i] = names[i]
                    key = f"{class_strs[i]}/{names[i]}"
                    lookup[key] = offset + i

                offset += n
            print(f"    Merged {os.path.basename(sp)}: +{n} → {offset:,}")

        out.attrs['total_samples'] = total
        out.attrs['emb_dim'] = emb_dim

    # Write JSON lookup
    with open(merged_json, 'w') as fp:
        json.dump(lookup, fp)

    size_mb = os.path.getsize(merged_h5) / 1e6
    print(f"\n  Merged H5:   {merged_h5}  ({size_mb:.1f} MB, {total:,} samples)")
    print(f"  Lookup JSON: {merged_json}  ({len(lookup):,} entries)")

    # Quick verification
    with h5py.File(merged_h5, 'r') as f:
        embs = f['emb']
        print(f"  Verify: shape={embs.shape}, dtype={embs.dtype}")
        sample = embs[:min(1000, total)]
        norms = np.linalg.norm(sample, axis=1)
        print(f"  L2 norms: mean={norms.mean():.3f}, std={norms.std():.3f}")
        means = sample.mean(axis=1)
        print(f"  Per-sample mean: mean={means.mean():.4f} (should be ~0)")
        stds = sample.std(axis=1)
        print(f"  Per-sample std:  mean={stds.mean():.4f} (should be ~1)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract MoCo v3 CLS embeddings (through MLP head) → flat H5"
    )
    parser.add_argument("--image_dir", type=str, required=True,
                        help="ImageNet train directory (with class subfolders)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for per-rank H5 shards")
    parser.add_argument("--moco_checkpoint", type=str, required=True,
                        help="Path to MoCo v3 ViT-L checkpoint (vitl.pth.tar)")
    parser.add_argument("--merged_h5", type=str, required=True,
                        help="Output path for merged flat H5 file")
    parser.add_argument("--merged_json", type=str, required=True,
                        help="Output path for JSON lookup file")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size per GPU (default: 256)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers per GPU (default: 8)")
    parser.add_argument("--shard_pattern", type=str,
                        default="moco_cls_shard_*.h5",
                        help="Glob pattern for shard files")
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_ddp()

    # Build dataset
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_dataset = ImageNetPathDataset(args.image_dir, transform)
    total_images = len(full_dataset)
    if rank == 0:
        print(f"\nTotal images: {total_images:,}")
        print(f"World size: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")

    # Split across DDP ranks
    sampler = torch.utils.data.distributed.DistributedSampler(
        full_dataset, num_replicas=world_size, rank=rank, shuffle=False,
    ) if world_size > 1 else None

    dataloader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    samples_per_rank = len(dataloader.dataset) // max(world_size, 1)
    if rank < len(dataloader.dataset) % max(world_size, 1):
        samples_per_rank += 1

    # Load model
    if rank == 0:
        print(f"\nLoading MoCo v3 ViT-L from {args.moco_checkpoint} ...")
    model = load_moco_model(args.moco_checkpoint, device)
    if rank == 0:
        # Verify head output dim
        test_out = model.head(torch.randn(1, 1024, device=device))
        print(f"  MLP head: 1024 → {test_out.shape[1]}d")

    # Extract
    shard_path = os.path.join(args.output_dir, f"moco_cls_shard_{rank}.h5")
    n_extracted = extract_shard(
        model, dataloader, device, rank, shard_path, samples_per_rank,
    )

    # Synchronise
    if dist.is_initialized():
        dist.barrier()

    # Rank 0 merges
    if rank == 0:
        merge_shards(
            args.output_dir, args.shard_pattern,
            args.merged_h5, args.merged_json,
        )

    cleanup_ddp()
    print(f"[Rank {rank}] All done.")


if __name__ == "__main__":
    main()
