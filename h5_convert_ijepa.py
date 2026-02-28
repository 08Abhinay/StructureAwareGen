# ============================================================
# Convert ijepa_embeddings.h5 (bloated) → efficient flat H5
#
# PROBLEM with current file:
#   - 8-level deep group nesting mirroring the filesystem path
#   - ~1.3M individual groups + ~1.3M individual datasets
#   - No chunking, no compression, 25% metadata overhead (8 GB vs 6.2 GB raw)
#
# OUTPUT: ijepa_emb_flat.h5
#   emb        : (N, 1280) float32  — one contiguous matrix, chunked
#   class_ids  : (N,)      int32    — class index per sample
#   names      : (N,)      string   — sample name (e.g. "n01440764_10026")
#
# LOOKUP FILE: ijepa_lookup.json
#   { "0/n01440764_10026": 0, "0/n01440764_10027": 1, ... }
#   Lets the dataloader do O(1) lookup: idx = lookup[f"{class_id}/{name}"]
#   then read h5["emb"][idx]
# ============================================================
import os, h5py, numpy as np, json, time
from tqdm import tqdm

SRC_H5    = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/ijepa_embeddings.h5"
DST_H5    = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/ijepa_emb_flat.h5"
DST_JSON  = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/ijepa_lookup.json"
H5_PREFIX = "scratch/gilbreth/abelde/Thesis/StructureAwareGen/ijepa_embeddings"
EMB_DIM   = 1280
CHUNK_ROWS = 8192

print(f"Source : {SRC_H5}")
print(f"Output : {DST_H5}")
print(f"Lookup : {DST_JSON}")

# ── 1. Scan the source H5 to build full sample list ──────────────────────────
print("\n1/3  Scanning source H5 (class list)...")
t0 = time.time()

with h5py.File(SRC_H5, "r") as src:
    base    = src[H5_PREFIX]
    classes = sorted(base.keys(), key=lambda x: int(x))
    n_cls   = len(classes)

    # Sample 20 classes to estimate total
    import random
    random.seed(42)
    probe_cls = random.sample(classes, min(20, n_cls))
    avg_per_cls = sum(len(base[c]) for c in probe_cls) / len(probe_cls)
    est_total = int(avg_per_cls * n_cls)
    print(f"     {n_cls} classes, avg ~{avg_per_cls:.0f} samples/class → est. {est_total:,} total")

    # Full scan for exact count
    print("     Full scan for exact count...")
    sample_list = []   # (class_id_str, sample_name)
    for cls_id in tqdm(classes, desc="Scanning"):
        for name in base[cls_id].keys():
            sample_list.append((cls_id, name))

n_samples = len(sample_list)
print(f"     Exact total: {n_samples:,}  [{time.time()-t0:.1f}s]")

# ── 2. Write flat H5 ─────────────────────────────────────────────────────────
print(f"\n2/3  Writing flat H5...")
t1 = time.time()
os.makedirs(os.path.dirname(DST_H5), exist_ok=True)

lookup = {}  # "class_id/name" → row index

with h5py.File(SRC_H5, "r") as src, h5py.File(DST_H5, "w") as dst:
    base = src[H5_PREFIX]

    emb_ds       = dst.create_dataset("emb",       shape=(n_samples, EMB_DIM), dtype="float32",
                                       chunks=(CHUNK_ROWS, EMB_DIM))
    class_ids_ds = dst.create_dataset("class_ids", shape=(n_samples,), dtype="int32")
    names_ds     = dst.create_dataset("names",     shape=(n_samples,), dtype=h5py.string_dtype())

    errs = []
    for idx, (cls_id, name) in enumerate(tqdm(sample_list, desc="Writing")):
        try:
            item = base[cls_id][name]
            if isinstance(item, h5py.Group):
                emb = item["emb"][:]
            else:  # Dataset directly
                emb = item[:]

            emb_ds[idx]       = emb
            class_ids_ds[idx] = int(cls_id)
            names_ds[idx]     = name
            lookup[f"{cls_id}/{name}"] = idx
        except Exception as e:
            errs.append((cls_id, name, str(e)))

    dst.attrs["n_samples"] = n_samples
    dst.attrs["emb_dim"]   = EMB_DIM
    dst.attrs["source"]    = os.path.basename(SRC_H5)

print(f"     Written {n_samples:,} rows  [{time.time()-t1:.1f}s]")
if errs:
    print(f"     Errors: {len(errs)}")
    for c, n, m in errs[:5]:
        print(f"       {c}/{n}: {m}")

# ── 3. Save lookup JSON ───────────────────────────────────────────────────────
print(f"\n3/3  Writing lookup JSON ({len(lookup):,} entries)...")
with open(DST_JSON, "w") as jf:
    json.dump(lookup, jf, separators=(",", ":"))
print(f"     Saved {DST_JSON}")

# ── Summary ───────────────────────────────────────────────────────────────────
src_mb = os.path.getsize(SRC_H5) / (1024**2)
dst_mb = os.path.getsize(DST_H5) / (1024**2)
print(f"\nDone in {time.time()-t0:.1f}s")
print(f"  Old H5 : {src_mb:,.0f} MB")
print(f"  New H5 : {dst_mb:,.0f} MB  ({(1 - dst_mb/src_mb)*100:.0f}% smaller)")
print(f"\nTo use in the dataloader, load the lookup once at init:")
print(f"  import json, h5py, numpy as np")
print(f"  lookup = json.load(open('{DST_JSON}'))")
print(f"  h5 = h5py.File('{DST_H5}', 'r')")
print(f"  emb = h5['emb'][lookup[f'{{class_id}}/{{name}}']")
