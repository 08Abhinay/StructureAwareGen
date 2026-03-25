"""
Generate a small test H5 file from REN DINOv2 extraction for SEG-RDM verification.
Processes a handful of real ImageNet images and writes a flat H5 file in the exact
format that seg_dataset.py expects.

Usage (single GPU):
    python create_test_ren_h5.py
"""
import os
import sys
import h5py
import numpy as np
import torch
import yaml

# Paths
PROJECT_ROOT = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen"
SCRIPT_DIR = os.path.join(PROJECT_ROOT, "scripts", "segProto")
sys.path.insert(0, SCRIPT_DIR)

from ren_model import DINOv2Extractor, RegionEncoder, TokenAggregator, SLICPrompter
from precompute_ren_embeddings_h5 import process_image, load_and_resize

IMAGE_DIR = os.path.join(PROJECT_ROOT, "dataset", "imagenet-1K-hf", "train")
REN_CONFIG = os.path.join(SCRIPT_DIR, "configs", "ren_dinov2_vitl14.yaml")
REN_CKPT = "/scratch/gilbreth/abelde/Thesis/REN/logs/ren-dinov2-vitl14/checkpoint.pth"
TORCH_HOME = os.path.join(PROJECT_ROOT, ".cache", "torch")
OUTPUT_H5 = os.path.join(PROJECT_ROOT, "h5_embeddings", "region_ren_dinov2_test.h5")

MAX_IMAGES = 50  # Small subset for testing
IMAGE_RES = 518
GRID_SIZE = 37
MERGE_SIMILARITY = 0.975


def main():
    os.environ["TORCH_HOME"] = TORCH_HOME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load REN config
    with open(REN_CONFIG) as f:
        cfg = yaml.safe_load(f)

    # Build models
    print("Loading DINOv2 extractor...")
    extractor = DINOv2Extractor(device=device)

    print("Loading RegionEncoder...")
    ren = RegionEncoder(cfg["ren"])
    ckpt = torch.load(REN_CKPT, map_location="cpu")
    ren.load_state_dict(ckpt['region_encoder_state'])
    ren = ren.to(device).eval()
    for p in ren.parameters():
        p.requires_grad_(False)

    aggregator = TokenAggregator(
        merge_similarity=cfg["ren"]["parameters"].get("merge_similarity", MERGE_SIMILARITY)
    )
    prompter = SLICPrompter(image_resolution=IMAGE_RES)

    # Collect images
    all_images = []
    class_dirs = sorted([d for d in os.listdir(IMAGE_DIR)
                         if os.path.isdir(os.path.join(IMAGE_DIR, d)) and d.isdigit()])
    for class_id in class_dirs:
        class_path = os.path.join(IMAGE_DIR, class_id)
        for img_file in sorted(os.listdir(class_path)):
            if img_file.endswith((".JPEG", ".jpg", ".jpeg", ".png")):
                all_images.append((class_id, os.path.splitext(img_file)[0],
                                   os.path.join(class_path, img_file)))
            if len(all_images) >= MAX_IMAGES:
                break
        if len(all_images) >= MAX_IMAGES:
            break

    print(f"Processing {len(all_images)} images...")

    # Process each image
    D = 1024
    all_embs, all_scores = [], []
    offsets, n_segments_list = [], []
    class_ids, names, cls_embs = [], [], []
    seg_cursor = 0
    stats = {"total_regions": 0, "min_regions": float("inf"), "max_regions": 0,
             "region_counts": []}

    for i, (class_id, name, img_path) in enumerate(all_images):
        result = process_image(
            img_path, extractor, ren, aggregator, prompter,
            device, GRID_SIZE, IMAGE_RES, True
        )

        if result is None or result["n_segments"] == 0:
            print(f"  [{i+1}] {class_id}/{name}: SKIPPED (no regions)")
            continue

        emb_np = result["emb"]        # (N, 1024)
        scores_np = result["scores"]  # (N,)
        cls_np = result["cls_emb"]    # (1024,)
        n = emb_np.shape[0]

        all_embs.append(emb_np)
        all_scores.append(scores_np)
        offsets.append(seg_cursor)
        n_segments_list.append(n)
        class_ids.append(int(class_id))
        names.append(name)
        cls_embs.append(cls_np)
        seg_cursor += n

        stats["total_regions"] += n
        stats["min_regions"] = min(stats["min_regions"], n)
        stats["max_regions"] = max(stats["max_regions"], n)
        stats["region_counts"].append(n)

        print(f"  [{i+1}/{len(all_images)}] {class_id}/{name}: {n} regions, "
              f"emb shape {emb_np.shape}")

    # Write H5
    os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)

    all_embs_cat = np.concatenate(all_embs, axis=0)      # (total_segments, 1024)
    all_scores_cat = np.concatenate(all_scores, axis=0)   # (total_segments,)
    cls_embs_cat = np.stack(cls_embs, axis=0)             # (n_images, 1024)

    n_images = len(class_ids)
    print(f"\nWriting {OUTPUT_H5} ...")
    print(f"  Images: {n_images}")
    print(f"  Total segments: {seg_cursor}")

    with h5py.File(OUTPUT_H5, "w") as f:
        f.create_dataset("emb", data=all_embs_cat, dtype="float32")
        f.create_dataset("scores", data=all_scores_cat, dtype="float32")
        f.create_dataset("cls_emb", data=cls_embs_cat, dtype="float32")
        f.create_dataset("offsets", data=np.array(offsets, dtype=np.int64))
        f.create_dataset("n_segments", data=np.array(n_segments_list, dtype=np.int32))
        f.create_dataset("class_ids", data=np.array(class_ids, dtype=np.int32))
        f.create_dataset("names", data=np.array(names, dtype=h5py.string_dtype()))

        f.attrs["total_samples"] = n_images
        f.attrs["total_segments"] = seg_cursor
        f.attrs["emb_dim"] = D
        f.attrs["source"] = "ren_dinov2_vitl14"

    # Print statistics
    region_counts = np.array(stats["region_counts"])
    print(f"\n{'='*60}")
    print(f"EXTRACTION STATISTICS")
    print(f"{'='*60}")
    print(f"Images processed:   {n_images}")
    print(f"Total regions:      {stats['total_regions']}")
    print(f"Avg regions/image:  {region_counts.mean():.1f}")
    print(f"Median regions:     {np.median(region_counts):.1f}")
    print(f"Min regions:        {stats['min_regions']}")
    print(f"Max regions:        {stats['max_regions']}")
    print(f"Std regions:        {region_counts.std():.1f}")
    print(f"Embedding dim:      {D}")
    print(f"Emb shape:          {all_embs_cat.shape}")
    print(f"CLS shape:          {cls_embs_cat.shape}")
    print(f"H5 file:            {OUTPUT_H5}")
    print(f"H5 file size:       {os.path.getsize(OUTPUT_H5) / 1024 / 1024:.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
