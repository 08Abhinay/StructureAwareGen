#!/usr/bin/env python3
"""
Merge per-rank region delta H5 files into a flat base region H5.

Usage:
  python tools/merge_region_h5_deltas.py \
    --base-h5 /path/region_emb_flat.h5 \
    --delta-dir /path/delta_dir \
    --out-h5 /path/region_emb_flat_merged.h5
"""

import argparse
import glob
import os

import numpy as np

try:
    import h5py
except ImportError as exc:
    raise SystemExit("h5py is required: pip install h5py") from exc


def _decode_name(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode()
    return str(value)


def _copy_large_2d(src_ds, dst_ds, block=131072):
    n = src_ds.shape[0]
    for start in range(0, n, block):
        end = min(n, start + block)
        dst_ds[start:end] = src_ds[start:end]


def _copy_large_1d(src_ds, dst_ds, block=524288):
    n = src_ds.shape[0]
    for start in range(0, n, block):
        end = min(n, start + block)
        dst_ds[start:end] = src_ds[start:end]


def merge(base_h5: str, delta_dir: str, out_h5: str):
    delta_paths = sorted(glob.glob(os.path.join(delta_dir, "*.h5")))
    print(f"Base H5: {base_h5}")
    print(f"Delta dir: {delta_dir}")
    print(f"Delta files: {len(delta_paths)}")
    print(f"Out H5: {out_h5}")

    with h5py.File(base_h5, "r") as base, h5py.File(out_h5, "w") as out:
        str_dt = h5py.string_dtype(encoding="utf-8")

        n_samples = int(base["class_ids"].shape[0])
        n_segments = int(base["emb"].shape[0])
        print(f"Copying base: {n_samples} samples, {n_segments} segments")

        out.create_dataset("class_ids", shape=(n_samples,), maxshape=(None,), dtype=np.int32)
        out.create_dataset("names", shape=(n_samples,), maxshape=(None,), dtype=str_dt)
        out.create_dataset("offsets", shape=(n_samples,), maxshape=(None,), dtype=np.int64)
        out.create_dataset("n_segments", shape=(n_samples,), maxshape=(None,), dtype=np.int32)
        out.create_dataset("mask_shapes", shape=(n_samples, 3), maxshape=(None, 3), dtype=np.int32)
        out.create_dataset("emb", shape=(n_segments, 256), maxshape=(None, 256), dtype=np.float32)
        out.create_dataset("scores", shape=(n_segments,), maxshape=(None,), dtype=np.float32)

        out["class_ids"][:] = base["class_ids"][:]
        out["names"][:] = base["names"][:]
        out["offsets"][:] = base["offsets"][:]
        out["n_segments"][:] = base["n_segments"][:]
        out["mask_shapes"][:] = base["mask_shapes"][:]
        _copy_large_2d(base["emb"], out["emb"])
        _copy_large_1d(base["scores"], out["scores"])

        for k, v in base.attrs.items():
            out.attrs[k] = v

        existing_keys = set()
        base_class_ids = out["class_ids"][:]
        base_names = out["names"][:]
        for cid, name in zip(base_class_ids, base_names):
            existing_keys.add((str(int(cid)), _decode_name(name)))
        print(f"Indexed base keys: {len(existing_keys)}")

        appended = 0
        skipped = 0
        for delta_path in delta_paths:
            print(f"Merging {delta_path}")
            with h5py.File(delta_path, "r") as delta:
                if "class_ids" not in delta:
                    print(f"  skipping (missing expected datasets)")
                    continue
                dn = int(delta["class_ids"].shape[0])
                for i in range(dn):
                    class_id = int(delta["class_ids"][i])
                    name = _decode_name(delta["names"][i])
                    key = (str(class_id), name)
                    if key in existing_keys:
                        skipped += 1
                        continue

                    off = int(delta["offsets"][i])
                    ns = int(delta["n_segments"][i])
                    mask_shape = np.asarray(delta["mask_shapes"][i], dtype=np.int32)

                    sidx = out["class_ids"].shape[0]
                    seg_off = out["emb"].shape[0]

                    out["class_ids"].resize((sidx + 1,))
                    out["names"].resize((sidx + 1,))
                    out["offsets"].resize((sidx + 1,))
                    out["n_segments"].resize((sidx + 1,))
                    out["mask_shapes"].resize((sidx + 1, 3))

                    out["class_ids"][sidx] = class_id
                    out["names"][sidx] = name
                    out["offsets"][sidx] = seg_off
                    out["n_segments"][sidx] = ns
                    out["mask_shapes"][sidx] = mask_shape[:3]

                    if ns > 0:
                        out["emb"].resize((seg_off + ns, 256))
                        out["scores"].resize((seg_off + ns,))
                        out["emb"][seg_off:seg_off + ns] = np.asarray(delta["emb"][off:off + ns], dtype=np.float32)
                        out["scores"][seg_off:seg_off + ns] = np.asarray(delta["scores"][off:off + ns], dtype=np.float32)

                    existing_keys.add(key)
                    appended += 1

        out.attrs["total_samples"] = int(out["class_ids"].shape[0])
        out.attrs["total_segments"] = int(out["emb"].shape[0])
        out.attrs["source"] = "merged_base_plus_region_deltas"
        out.flush()

    print("Merge complete")
    print(f"  appended samples: {appended}")
    print(f"  skipped duplicates: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Merge per-rank region delta H5 files into a flat region H5")
    parser.add_argument("--base-h5", required=True, type=str)
    parser.add_argument("--delta-dir", required=True, type=str)
    parser.add_argument("--out-h5", required=True, type=str)
    args = parser.parse_args()
    merge(args.base_h5, args.delta_dir, args.out_h5)


if __name__ == "__main__":
    main()
