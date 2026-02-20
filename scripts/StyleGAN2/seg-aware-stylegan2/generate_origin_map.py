#!/usr/bin/env python3
"""
Generate origin_map.json that maps zip archive filenames → original filenames.

This replays the exact same sorting logic from dataset_tool.py's open_image_folder()
to reconstruct which sequential index corresponds to which original file.

The origin_map.json is required by AlignedSegDataset to translate zip-internal names
(e.g. "00000/img00000005.png") back to original ImageNet names (e.g. "0/980")
so that pre-computed .npz embeddings can be found.

Usage (full ImageNet):
    python generate_origin_map.py \
        --source /scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/train \
        --dest   /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/256/origin_map.json

For debug subset (flat folder, no class subfolders):
    python generate_origin_map.py \
        --source /scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet_debug_subset \
        --dest   /path/to/debug_zip_folder/origin_map.json

If you used --max-images with dataset_tool.py, pass the same value here:
    python generate_origin_map.py --source ... --dest ... --max-images 50000
"""

import argparse
import json
from pathlib import Path

import PIL.Image


def file_ext(name):
    return str(name).split('.')[-1]


def is_image_ext(fname):
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION


def main():
    parser = argparse.ArgumentParser(
        description="Generate origin_map.json: zip archive names → original image stems"
    )
    parser.add_argument("--source", required=True,
                        help="Original --source directory used with dataset_tool.py")
    parser.add_argument("--dest", required=True,
                        help="Output path for origin_map.json (place next to or near the .zip)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="If you used --max-images with dataset_tool.py, set the same value here")
    args = parser.parse_args()

    PIL.Image.init()

    source_dir = Path(args.source)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # ---- Replay the EXACT same logic as dataset_tool.py open_image_folder() ----
    imgs = [p for p in sorted(source_dir.rglob("*"))
            if is_image_ext(p) and p.is_file()]

    if args.max_images is not None:
        imgs = imgs[:args.max_images]

    print(f"Found {len(imgs)} images in {source_dir}")

    # Build the mapping:
    #   zip_fname  →  class_name/original_stem
    #
    # zip_fname  = "00000/img00000005.png"  (what dataset_tool.py produces)
    # orig_key   = "0/980"                  (what precompute scripts used as folder/stem)
    origin_map = {}

    for idx, path in enumerate(imgs):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        # Relative path from source, e.g. "0/980.JPEG" or just "980.JPEG" (flat)
        rel_path = path.relative_to(source_dir)
        # class_name/stem without extension: e.g. "0/980" or "./980"
        orig_key = str(rel_path.parent / rel_path.stem)

        origin_map[archive_fname] = orig_key

    # Save
    dest_path = Path(args.dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, 'w') as f:
        json.dump(origin_map, f)

    print(f"\nSaved origin_map.json with {len(origin_map)} entries → {dest_path}")

    # Sanity check: print first and last few entries
    items = list(origin_map.items())
    print("\nFirst 5 entries:")
    for k, v in items[:5]:
        print(f"  {k}  →  {v}")
    if len(items) > 10:
        print(f"  ... ({len(items) - 10} more) ...")
        print("\nLast 5 entries:")
        for k, v in items[-5:]:
            print(f"  {k}  →  {v}")


if __name__ == "__main__":
    main()
