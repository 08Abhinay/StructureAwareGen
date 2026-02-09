#!/usr/bin/env python3
"""
Precompute SAM embeddings for a subset of ImageNet (for RDM training).
This script extracts SAM embeddings in parallel across multiple GPUs using DDP.
The cache is shared between RDM and StyleGAN2 training to avoid duplicate extraction.

Usage:
    # Extract for 40% of ImageNet with 4 GPUs:
    torchrun --nproc_per_node=4 scripts/precompute_sam_embeddings.py \
        --data_path /path/to/imagenet/train \
        --cache_dir /path/to/sam_cache \
        --subset_fraction 0.4 \
        --seed 42

    # Extract full dataset:
    torchrun --nproc_per_node=4 scripts/precompute_sam_embeddings.py \
        --data_path /path/to/imagenet/train \
        --cache_dir /path/to/sam_cache
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add StyleGAN2 path for SAMExtractor import
stylegan2_path = Path(__file__).parent.parent / "scripts/StyleGAN2/seg-aware-stylegan2"
sys.path.insert(0, str(stylegan2_path))

from training.sam_extractor import SAMExtractor


class ImageNetSubset(Dataset):
    """ImageNet dataset with optional deterministic subset selection."""
    
    def __init__(
        self,
        root: str,
        subset_fraction: float = 1.0,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.root = Path(root)
        self.subset_fraction = subset_fraction
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        
        # Collect all images
        self.samples = self._collect_samples()
        
        # Apply subset selection (deterministic across ranks)
        if subset_fraction < 1.0:
            self._apply_subset()
        
        # Distribute across ranks (strided for load balancing)
        self.samples = self.samples[rank::world_size]
        
        # Transform for SAM (expects 1024x1024)
        self.transform = transforms.Compose([
            transforms.Resize(1024, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(1024),
            transforms.ToTensor(),
        ])
    
    def _collect_samples(self) -> List[Tuple[Path, str]]:
        """Collect all image paths with their class names."""
        samples = []
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for img_path in sorted(class_dir.glob("*.JPEG")):
                samples.append((img_path, class_name))
        return samples
    
    def _apply_subset(self):
        """Select deterministic subset based on seed."""
        rng = np.random.RandomState(self.seed)
        n_total = len(self.samples)
        n_subset = int(n_total * self.subset_fraction)
        indices = rng.choice(n_total, size=n_subset, replace=False)
        indices = sorted(indices)  # Keep sorted for reproducibility
        self.samples = [self.samples[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        img_path, class_name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        img_stem = img_path.stem
        return img_tensor, class_name, img_stem


def setup_ddp():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    return rank, world_size, local_rank


def cache_exists(cache_dir: Path, class_name: str, img_stem: str) -> bool:
    """Check if embedding is already cached."""
    cache_path = cache_dir / class_name / f"{img_stem}.npz"
    return cache_path.exists()


def main():
    parser = argparse.ArgumentParser(description="Precompute SAM embeddings for ImageNet subset")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to ImageNet train directory")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Directory to save SAM embeddings (.npz files)")
    parser.add_argument("--subset_fraction", type=float, default=0.4,
                        help="Fraction of dataset to extract (default: 0.4 for RDM)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic subset selection")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per GPU (default: 8)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers per GPU")
    parser.add_argument("--sam_model", type=str, default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM model variant (default: vit_h)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip images that already have cached embeddings")
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"=" * 80)
        print(f"SAM Embedding Precomputation")
        print(f"=" * 80)
        print(f"Data path: {args.data_path}")
        print(f"Cache dir: {args.cache_dir}")
        print(f"Subset fraction: {args.subset_fraction:.1%}")
        print(f"Seed: {args.seed}")
        print(f"World size: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"SAM model: {args.sam_model}")
        print(f"Skip existing: {args.skip_existing}")
        print(f"=" * 80)
    
    # Create cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset (each rank gets its strided subset)
    dataset = ImageNetSubset(
        root=args.data_path,
        subset_fraction=args.subset_fraction,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    
    if rank == 0:
        total_images = len(dataset) * world_size
        print(f"Total images to process: {total_images:,}")
        print(f"Images per GPU (rank {rank}): {len(dataset):,}")
    
    # Create dataloader (no shuffling - deterministic order)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    # Initialize SAM extractor (reuse StyleGAN2 implementation)
    sam_extractor = SAMExtractor(
        cache_dir=str(cache_dir),
        model_type=args.sam_model,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    
    # Force SAM initialization (lazy init in extract_segments)
    sam_extractor._lazy_init_sam()
    
    if rank == 0:
        print(f"SAM model loaded: {args.sam_model}")
        print(f"Starting extraction...")
        print(f"=" * 80)
    
    # Process batches
    start_time = time.time()
    processed = 0
    skipped = 0
    
    pbar = None
    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc=f"GPU {rank}")
    
    for batch_imgs, batch_classes, batch_stems in dataloader:
        batch_imgs = batch_imgs.to(device)
        batch_size = batch_imgs.size(0)
        
        # Check cache for skip_existing mode
        if args.skip_existing:
            to_process = []
            to_process_imgs = []
            for i in range(batch_size):
                class_name = batch_classes[i]
                img_stem = batch_stems[i]
                if not cache_exists(cache_dir, class_name, img_stem):
                    to_process.append(i)
                    to_process_imgs.append(batch_imgs[i])
                else:
                    skipped += 1
            
            if len(to_process) == 0:
                if pbar:
                    pbar.update(1)
                continue
            
            # Process only non-cached images
            batch_imgs = torch.stack(to_process_imgs)
            batch_classes = [batch_classes[i] for i in to_process]
            batch_stems = [batch_stems[i] for i in to_process]
            batch_size = len(to_process)
        
        # Extract embeddings
        # SAMExtractor.extract_segments expects (B, 3, H, W) and returns list of dicts
        # Each dict: {'embeddings': (N, 256), 'scores': (N,)}
        results = sam_extractor.extract_segments(batch_imgs)
        
        # Save to cache (SAMExtractor already saves internally, but verify)
        for i in range(batch_size):
            class_name = batch_classes[i]
            img_stem = batch_stems[i]
            
            # Create class subdirectory
            class_cache_dir = cache_dir / class_name
            class_cache_dir.mkdir(exist_ok=True)
            
            # Save .npz file (format compatible with both RDM and StyleGAN2)
            cache_path = class_cache_dir / f"{img_stem}.npz"
            if not cache_path.exists():  # Double-check (SAMExtractor should handle this)
                emb = results[i]['embeddings'].cpu().numpy().astype(np.float16)
                scores = results[i]['scores'].cpu().numpy().astype(np.float32)
                np.savez_compressed(cache_path, emb=emb, scores=scores)
            
            processed += 1
        
        if pbar:
            pbar.update(1)
    
    if pbar:
        pbar.close()
    
    # Synchronize across ranks
    if world_size > 1:
        dist.barrier()
    
    # Gather statistics
    if world_size > 1:
        processed_tensor = torch.tensor([processed], device=device)
        skipped_tensor = torch.tensor([skipped], device=device)
        dist.all_reduce(processed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(skipped_tensor, op=dist.ReduceOp.SUM)
        processed = processed_tensor.item()
        skipped = skipped_tensor.item()
    
    elapsed = time.time() - start_time
    
    if rank == 0:
        print(f"=" * 80)
        print(f"Extraction complete!")
        print(f"Processed: {processed:,} images")
        if args.skip_existing:
            print(f"Skipped (cached): {skipped:,} images")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Speed: {processed/elapsed:.1f} images/sec")
        print(f"Cache directory: {cache_dir}")
        print(f"=" * 80)
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
