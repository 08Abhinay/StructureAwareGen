#!/usr/bin/env python3
"""
Verify SAM embedding cache completeness for ImageNet dataset.
Checks how many images have cached embeddings and estimates training time savings.

Usage:
    python scripts/verify_sam_cache.py \
        --data_path /path/to/imagenet/train \
        --cache_dir /path/to/sam_cache \
        --subset_fraction 0.4
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm


def collect_image_paths(data_path: Path, subset_fraction: float = 1.0, seed: int = 42) -> List[Tuple[Path, str]]:
    """Collect all image paths matching subset selection."""
    samples = []
    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_path in sorted(class_dir.glob("*.JPEG")):
            samples.append((img_path, class_name))
    
    # Apply subset selection (deterministic)
    if subset_fraction < 1.0:
        rng = np.random.RandomState(seed)
        n_total = len(samples)
        n_subset = int(n_total * subset_fraction)
        indices = rng.choice(n_total, size=n_subset, replace=False)
        indices = sorted(indices)
        samples = [samples[i] for i in indices]
    
    return samples


def check_cache(cache_dir: Path, class_name: str, img_stem: str) -> Tuple[bool, int]:
    """
    Check if embedding is cached and return (exists, num_segments).
    Returns (False, 0) if not cached.
    """
    cache_path = cache_dir / class_name / f"{img_stem}.npz"
    if not cache_path.exists():
        return False, 0
    
    try:
        data = np.load(cache_path)
        emb = data['emb']  # Shape: (N, 256)
        num_segments = emb.shape[0]
        return True, num_segments
    except Exception as e:
        print(f"Warning: Failed to load {cache_path}: {e}")
        return False, 0


def main():
    parser = argparse.ArgumentParser(description="Verify SAM embedding cache completeness")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to ImageNet train directory")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Directory containing SAM embeddings")
    parser.add_argument("--subset_fraction", type=float, default=1.0,
                        help="Fraction of dataset to check (default: 1.0 for full dataset)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subset selection (must match extraction)")
    parser.add_argument("--show_missing", action="store_true",
                        help="Print paths of missing cached embeddings")
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    cache_dir = Path(args.cache_dir)
    
    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        return
    
    if not cache_dir.exists():
        print(f"Error: Cache directory does not exist: {cache_dir}")
        return
    
    print("=" * 80)
    print("SAM Cache Verification")
    print("=" * 80)
    print(f"Data path: {data_path}")
    print(f"Cache dir: {cache_dir}")
    print(f"Subset fraction: {args.subset_fraction:.1%}")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    
    # Collect image paths
    print("Collecting image paths...")
    samples = collect_image_paths(data_path, args.subset_fraction, args.seed)
    print(f"Total images to check: {len(samples):,}")
    
    # Check cache for each image
    print("Checking cache...")
    cached = 0
    missing = 0
    total_segments = 0
    missing_paths = []
    
    for img_path, class_name in tqdm(samples, desc="Verifying"):
        img_stem = img_path.stem
        exists, num_segments = check_cache(cache_dir, class_name, img_stem)
        
        if exists:
            cached += 1
            total_segments += num_segments
        else:
            missing += 1
            if args.show_missing:
                missing_paths.append(img_path)
    
    # Statistics
    cache_rate = cached / len(samples) if len(samples) > 0 else 0.0
    avg_segments = total_segments / cached if cached > 0 else 0.0
    
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Total images: {len(samples):,}")
    print(f"Cached: {cached:,} ({cache_rate:.1%})")
    print(f"Missing: {missing:,} ({1-cache_rate:.1%})")
    print(f"Total segments: {total_segments:,}")
    print(f"Avg segments per image: {avg_segments:.1f}")
    print("=" * 80)
    
    # Estimate training impact
    if missing > 0:
        print("\nTraining Impact Estimate:")
        print(f"  Missing embeddings will be extracted on-the-fly during training")
        print(f"  Extraction overhead: ~0.5-1.0s per image (depends on GPU)")
        print(f"  Estimated startup delay: {missing * 0.75 / 3600:.1f} hours")
        print(f"  Recommendation: Run precompute_sam_embeddings.py first")
    else:
        print("\n✅ All embeddings are cached! Training will start immediately.")
    
    # Show missing paths
    if args.show_missing and missing_paths:
        print("\n" + "=" * 80)
        print("Missing Cache Files (first 20):")
        print("=" * 80)
        for path in missing_paths[:20]:
            print(f"  {path}")
        if len(missing_paths) > 20:
            print(f"  ... and {len(missing_paths) - 20} more")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
