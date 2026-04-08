#!/usr/bin/env python3
"""
Convert ImageNet-1K image folder → single flat HDF5 file.

Layout of the output H5:
    images      [N, jpeg_bytes]  variable-length uint8  – raw JPEG bytes (no decode)
    class_ids   [N]              int32                  – numeric class label
    names       [N]              string                 – basename without extension
    write_cursor  (attr)         int                    – resume marker: next index to write
    
The JPEG bytes are stored as-is (no re-encoding, no quality loss).
At read time the consumer does:
    buf = h5f['images'][idx]
    img = Image.open(io.BytesIO(buf.tobytes())).convert('RGB')
    img = transform(img)

Supports RESUME: if the output H5 already exists and was partially written,
the script detects the last written index and continues from there.

Usage:
    python h5_convert_imagenet.py                        # full run (or resume)
    python h5_convert_imagenet.py --dry_run              # scan only, no write
    python h5_convert_imagenet.py --limit 100            # convert first 100 images (test)
"""
import argparse
import os
import time

import h5py
import numpy as np
from tqdm import tqdm


def scan_imagenet(image_dir: str):
    """Scan ImageNet class-folder structure.
    Returns list of (class_id_int, basename_str, full_path_str)."""
    class_dirs = sorted(
        [d for d in os.listdir(image_dir)
         if os.path.isdir(os.path.join(image_dir, d)) and d.isdigit()],
        key=int,
    )
    samples = []
    for cid_str in tqdm(class_dirs, desc="Scanning classes"):
        cid = int(cid_str)
        cdir = os.path.join(image_dir, cid_str)
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(('.jpeg', '.jpg', '.png')):
                basename = os.path.splitext(fn)[0]
                samples.append((cid, basename, os.path.join(cdir, fn)))
    return samples


def convert(image_dir: str, dst_h5: str, limit: int = 0, dry_run: bool = False,
            flush_every: int = 5000):
    t0 = time.time()
    print(f"1/3  Scanning {image_dir} ...")
    samples = scan_imagenet(image_dir)
    n_total = len(samples)
    if limit > 0:
        samples = samples[:limit]
        print(f"     Using first {len(samples)} of {n_total} (--limit)")
    n = len(samples)
    print(f"     {n:,} images found  [{time.time()-t0:.1f}s]")

    if dry_run:
        print("     --dry_run: stopping here (no write).")
        return

    os.makedirs(os.path.dirname(dst_h5) or '.', exist_ok=True)

    # ── Resume detection ──
    start_idx = 0
    if os.path.exists(dst_h5):
        try:
            with h5py.File(dst_h5, 'r') as h5f:
                existing_n = h5f['images'].shape[0]
                if existing_n == n:
                    start_idx = int(h5f.attrs.get('write_cursor', 0))
                    if start_idx >= n:
                        print(f"     H5 already complete ({n:,} images). Nothing to do.")
                        return
                    print(f"     RESUMING from index {start_idx:,}/{n:,} "
                          f"({start_idx/n*100:.1f}% already done)")
                else:
                    print(f"     Existing H5 has {existing_n} slots but need {n}. "
                          f"Recreating from scratch.")
                    start_idx = 0
        except Exception as e:
            print(f"     Could not read existing H5 ({e}). Starting fresh.")
            start_idx = 0

    print(f"\n2/3  Writing {dst_h5} ...")
    t1 = time.time()

    vlen_uint8 = h5py.vlen_dtype(np.dtype('uint8'))

    # Open in append mode if resuming, write mode if starting fresh
    mode = 'a' if start_idx > 0 else 'w'
    with h5py.File(dst_h5, mode) as h5f:
        if start_idx == 0:
            img_ds   = h5f.create_dataset('images',    shape=(n,), dtype=vlen_uint8)
            cid_ds   = h5f.create_dataset('class_ids', shape=(n,), dtype='int32')
            name_ds  = h5f.create_dataset('names',     shape=(n,), dtype=h5py.string_dtype())
        else:
            img_ds   = h5f['images']
            cid_ds   = h5f['class_ids']
            name_ds  = h5f['names']

        errors = []
        for i in tqdm(range(start_idx, n), desc="Writing H5",
                      initial=start_idx, total=n):
            cid, basename, fpath = samples[i]
            try:
                with open(fpath, 'rb') as f:
                    raw = f.read()
                img_ds[i]  = np.frombuffer(raw, dtype=np.uint8)
                cid_ds[i]  = cid
                name_ds[i] = basename
            except Exception as e:
                errors.append((fpath, str(e)))
                img_ds[i]  = np.zeros(0, dtype=np.uint8)
                cid_ds[i]  = cid
                name_ds[i] = basename

            # Periodic flush + cursor update for safe resume
            if (i + 1) % flush_every == 0:
                h5f.attrs['write_cursor'] = i + 1
                h5f.flush()

        h5f.attrs['total_images']  = n
        h5f.attrs['source_dir']    = image_dir
        h5f.attrs['format']        = 'jpeg_bytes'
        h5f.attrs['write_cursor']  = n  # mark complete

    elapsed = time.time() - t0
    fsize = os.path.getsize(dst_h5) / (1024**3)
    print(f"\n3/3  Done.  {n:,} images → {fsize:.2f} GB  [{elapsed:.1f}s total]")
    if errors:
        print(f"     {len(errors)} errors:")
        for p, e in errors[:10]:
            print(f"       {p}: {e}")


def verify(dst_h5: str, n_check: int = 5):
    """Quick verification: read back a few samples."""
    import io
    from PIL import Image

    print(f"\nVerifying {dst_h5} ...")
    with h5py.File(dst_h5, 'r') as h5f:
        n = h5f.attrs['total_images']
        print(f"  total_images: {n}")
        print(f"  datasets: {list(h5f.keys())}")
        print(f"  images shape: {h5f['images'].shape}, dtype: {h5f['images'].dtype}")
        print(f"  class_ids shape: {h5f['class_ids'].shape}")
        print(f"  names shape: {h5f['names'].shape}")

        indices = np.random.default_rng(42).choice(n, min(n_check, n), replace=False)
        for idx in indices:
            raw = h5f['images'][idx]
            cid = h5f['class_ids'][idx]
            nm  = h5f['names'][idx]
            if isinstance(nm, bytes):
                nm = nm.decode()
            img = Image.open(io.BytesIO(raw.tobytes())).convert('RGB')
            print(f"  [{idx}] class={cid}  name={nm}  "
                  f"jpeg_bytes={len(raw):,}  decoded={img.size}")
    print("  Verification passed.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str,
                        default='/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/train')
    parser.add_argument('--dst_h5', type=str,
                        default='/scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/imagenet_train_images.h5')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of images to convert (0 = all)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only scan, do not write H5')
    parser.add_argument('--verify', action='store_true',
                        help='Run verification after conversion')
    args = parser.parse_args()

    convert(args.image_dir, args.dst_h5, limit=args.limit, dry_run=args.dry_run)
    if args.verify or args.limit > 0:
        verify(args.dst_h5)
