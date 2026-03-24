#!/usr/bin/env python3
"""
Merge H5 shards from parallel region extraction into one flat H5 file.

Reads all region_moco_shard_*.h5 files from a directory and concatenates
them into a single flat H5 file with recomputed offsets.

Final schema (matches h5_convert.py / seg_dataset.py expectations):
    emb:               (N_total_segments, 256)  float32
    scores:            (N_total_segments,)      float32
    areas:             (N_total_segments,)      float32
    bboxes:            (N_total_segments, 4)    float32  (XYWH)
    pred_ious:         (N_total_segments,)      float32
    stability_scores:  (N_total_segments,)      float32
    offsets:           (N_total_images,)        int64
    n_segments:        (N_total_images,)        int32
    class_ids:         (N_total_images,)        int32
    names:             (N_total_images,)        string
    mask_shapes:       (N_total_images, 3)      int32
    cls_emb:           (N_total_images, 256)    float32

Usage:
    python3 merge_h5_shards.py \\
        --shard_dir /path/to/shards/ \\
        --output /path/to/region_moco_flat.h5
"""

import os
import sys
import glob
import argparse
import numpy as np
import h5py
import time
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Merge H5 shards into a single flat H5 file"
    )
    parser.add_argument("--shard_dir", type=str, required=True,
                        help="Directory containing region_moco_shard_*.h5 files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for merged flat H5 file")
    parser.add_argument("--shard_pattern", type=str,
                        default="region_moco_shard_*.h5",
                        help="Glob pattern for shard files")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Run verification after merge")
    args = parser.parse_args()

    t0 = time.time()

    # ---- 1. Discover shards ----
    shard_paths = sorted(
        glob.glob(os.path.join(args.shard_dir, args.shard_pattern))
    )
    print(f"Found {len(shard_paths)} shards in {args.shard_dir}")
    if len(shard_paths) == 0:
        print("No shards found! Check --shard_dir and --shard_pattern.")
        sys.exit(1)

    # ---- 2. Scan shards for sizes ----
    print("Scanning shard sizes...")
    shard_info = []
    total_samples = 0
    total_segments = 0
    emb_dim = None
    has_cls_emb = True
    has_emb_mean = True
    has_areas = True
    has_bboxes = True
    has_pred_ious = True
    has_stability = True

    for sp in tqdm(shard_paths, desc="Scanning"):
        with h5py.File(sp, "r") as f:
            ns = f.attrs.get("total_samples", f["n_segments"].shape[0])
            nsegs = f.attrs.get("total_segments", f["emb"].shape[0])
            ed = f["emb"].shape[1] if nsegs > 0 else 1024

            if emb_dim is None:
                emb_dim = ed
            elif ed != emb_dim:
                print(f"  WARNING: emb_dim mismatch in {sp}: {ed} vs {emb_dim}")

            if "cls_emb" not in f:
                has_cls_emb = False
            if "emb_image_mean" not in f:
                has_emb_mean = False
            if "areas" not in f:
                has_areas = False
            if "bboxes" not in f:
                has_bboxes = False
            if "pred_ious" not in f:
                has_pred_ious = False
            if "stability_scores" not in f:
                has_stability = False

            shard_info.append({
                "path": sp,
                "n_samples": ns,
                "n_segments": nsegs,
            })
            total_samples += ns
            total_segments += nsegs

    print(f"  Total: {total_samples:,} samples, {total_segments:,} segments, "
          f"emb_dim={emb_dim}")
    print(f"  Has cls_emb: {has_cls_emb}, Has emb_image_mean: {has_emb_mean}")
    print(f"  Has areas: {has_areas}, Has bboxes: {has_bboxes}, "
          f"Has pred_ious: {has_pred_ious}, Has stability: {has_stability}")

    # ---- 3. Create output H5 and write ----
    print(f"\nWriting merged H5: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with h5py.File(args.output, "w") as dst:
        # Flat segment arrays
        emb_ds = dst.create_dataset(
            "emb", shape=(total_segments, emb_dim), dtype="float32",
            chunks=(min(8192, max(1, total_segments)), emb_dim),
        )
        scores_ds = dst.create_dataset(
            "scores", shape=(total_segments,), dtype="float32",
            chunks=(min(65536, max(1, total_segments)),),
        )

        if has_areas:
            areas_ds = dst.create_dataset(
                "areas", shape=(total_segments,), dtype="float32",
                chunks=(min(65536, max(1, total_segments)),),
            )
        if has_bboxes:
            bboxes_ds = dst.create_dataset(
                "bboxes", shape=(total_segments, 4), dtype="float32",
                chunks=(min(8192, max(1, total_segments)), 4),
            )
        if has_pred_ious:
            pred_ious_ds = dst.create_dataset(
                "pred_ious", shape=(total_segments,), dtype="float32",
                chunks=(min(65536, max(1, total_segments)),),
            )
        if has_stability:
            stability_ds = dst.create_dataset(
                "stability_scores", shape=(total_segments,), dtype="float32",
                chunks=(min(65536, max(1, total_segments)),),
            )

        # Per-sample arrays
        offsets_ds = dst.create_dataset(
            "offsets", shape=(total_samples,), dtype="int64"
        )
        n_seg_ds = dst.create_dataset(
            "n_segments", shape=(total_samples,), dtype="int32"
        )
        class_ids_ds = dst.create_dataset(
            "class_ids", shape=(total_samples,), dtype="int32"
        )
        names_ds = dst.create_dataset(
            "names", shape=(total_samples,), dtype=h5py.string_dtype()
        )
        mask_shapes_ds = dst.create_dataset(
            "mask_shapes", shape=(total_samples, 3), dtype="int32"
        )

        if has_cls_emb:
            cls_emb_ds = dst.create_dataset(
                "cls_emb", shape=(total_samples, emb_dim), dtype="float32",
                chunks=(min(8192, total_samples), emb_dim),
            )

        if has_emb_mean:
            emb_mean_ds = dst.create_dataset(
                "emb_image_mean", shape=(total_samples, emb_dim),
                dtype="float32",
                chunks=(min(8192, total_samples), emb_dim),
            )

        # Write shards sequentially
        sample_cursor = 0
        seg_cursor = 0

        for si, info in enumerate(tqdm(shard_info, desc="Merging shards")):
            sp = info["path"]
            ns = info["n_samples"]
            nsegs = info["n_segments"]

            if ns == 0:
                continue

            with h5py.File(sp, "r") as src:
                # Copy segment data
                if nsegs > 0:
                    # Read in chunks to avoid memory spikes for large shards
                    CHUNK = 100000
                    for c_start in range(0, nsegs, CHUNK):
                        c_end = min(c_start + CHUNK, nsegs)
                        emb_ds[seg_cursor + c_start:seg_cursor + c_end] = \
                            src["emb"][c_start:c_end]
                        scores_ds[seg_cursor + c_start:seg_cursor + c_end] = \
                            src["scores"][c_start:c_end]
                        if has_areas and "areas" in src:
                            areas_ds[seg_cursor + c_start:seg_cursor + c_end] = \
                                src["areas"][c_start:c_end]
                        if has_bboxes and "bboxes" in src:
                            bboxes_ds[seg_cursor + c_start:seg_cursor + c_end] = \
                                src["bboxes"][c_start:c_end]
                        if has_pred_ious and "pred_ious" in src:
                            pred_ious_ds[seg_cursor + c_start:seg_cursor + c_end] = \
                                src["pred_ious"][c_start:c_end]
                        if has_stability and "stability_scores" in src:
                            stability_ds[seg_cursor + c_start:seg_cursor + c_end] = \
                                src["stability_scores"][c_start:c_end]

                # Copy per-sample data with recomputed offsets
                src_offsets = src["offsets"][:]      # (ns,) int64
                src_n_segs = src["n_segments"][:]    # (ns,) int32
                src_class_ids = src["class_ids"][:]  # (ns,) int32
                src_names = src["names"][:]          # (ns,) string
                src_shapes = src["mask_shapes"][:]   # (ns, 3) int32

                # Recompute offsets relative to global cursor
                new_offsets = src_offsets + seg_cursor

                s = sample_cursor
                e = sample_cursor + ns
                offsets_ds[s:e] = new_offsets
                n_seg_ds[s:e] = src_n_segs
                class_ids_ds[s:e] = src_class_ids
                names_ds[s:e] = src_names
                mask_shapes_ds[s:e] = src_shapes

                if has_cls_emb and "cls_emb" in src:
                    cls_emb_ds[s:e] = src["cls_emb"][:]

                if has_emb_mean and "emb_image_mean" in src:
                    emb_mean_ds[s:e] = src["emb_image_mean"][:]

            sample_cursor += ns
            seg_cursor += nsegs

        # File-level attributes
        dst.attrs["total_samples"] = total_samples
        dst.attrs["total_segments"] = total_segments
        dst.attrs["emb_dim"] = emb_dim
        dst.attrs["emb_dtype"] = "float32"
        dst.attrs["source"] = "mocov3_region_extraction_merged"

    el = time.time() - t0
    sz = os.path.getsize(args.output)
    print(f"\nMerge complete in {el:.0f}s ({el / 60:.1f} min)")
    print(f"  {total_samples:,} samples, {total_segments:,} segments")
    print(f"  File: {sz / (1024**3):.2f} GB")

    # ---- 4. Verify ----
    if args.verify:
        print("\nVerifying merged H5...")
        rng = np.random.default_rng(42)
        with h5py.File(args.output, "r") as f:
            assert f["emb"].shape == (total_segments, emb_dim), \
                f"emb shape mismatch: {f['emb'].shape}"
            assert f["offsets"].shape == (total_samples,), \
                f"offsets shape mismatch: {f['offsets'].shape}"
            assert f["n_segments"].shape == (total_samples,), \
                f"n_segments shape mismatch"
            if has_cls_emb:
                assert f["cls_emb"].shape == (total_samples, emb_dim), \
                    f"cls_emb shape mismatch: {f['cls_emb'].shape}"

            # Spot check 10 random samples
            n_check = min(10, total_samples)
            idxs = rng.choice(total_samples, n_check, replace=False)
            print(f"  Spot-checking {n_check} random samples...")
            for idx in idxs:
                off = int(f["offsets"][idx])
                ns = int(f["n_segments"][idx])
                cid = int(f["class_ids"][idx])
                nm = f["names"][idx]
                nm = nm.decode() if isinstance(nm, bytes) else nm
                if ns > 0:
                    emb = f["emb"][off:off + ns]
                    assert emb.shape == (ns, emb_dim), \
                        f"emb shape at idx={idx}: {emb.shape}"
                    assert not np.isnan(emb).any(), f"NaN in emb at idx={idx}"
                if has_cls_emb:
                    cls = f["cls_emb"][idx]
                    assert cls.shape == (emb_dim,), \
                        f"cls_emb shape at idx={idx}: {cls.shape}"
                    assert not np.isnan(cls).any(), \
                        f"NaN in cls_emb at idx={idx}"
                print(f"    idx={idx:>7d}  class={cid}  name={nm}  "
                      f"segs={ns}  off={off}")

        print("  Verification passed!")


if __name__ == "__main__":
    main()
