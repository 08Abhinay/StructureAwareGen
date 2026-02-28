# ============================================================
# Convert raw NPZ folder → efficient flat H5
# Source: region_emb_extract-a100-0.65dedup/{class}/masks_npz/*.npz
# Each NPZ has: emb (N,256) float32, scores (N,) float32, shape (3,) int32
# Drops: label_map (redundant), packed (not used in training), meta/
# ============================================================
import os, numpy as np, h5py, time
from tqdm import tqdm

NPZ_ROOT = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/region_emb_extract-a100-0.65dedup"
DST_H5   = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/region_emb_flat.h5"
EMB_DIM  = 256

# ── 1. Scan filesystem (fast, ~30s) ──
print("1/3  Scanning filesystem...")
t0 = time.time()
classes = sorted(
    [d for d in os.listdir(NPZ_ROOT)
     if os.path.isdir(os.path.join(NPZ_ROOT, d)) and d.isdigit()],
    key=int,
)
sample_list = []  # (class_id_int, sample_name, npz_path)
for c in tqdm(classes, desc="Scanning classes"):
    mdir = os.path.join(NPZ_ROOT, c, "masks_npz")
    if not os.path.exists(mdir):
        continue
    for fn in sorted(os.listdir(mdir)):
        if fn.endswith(".npz"):
            sample_list.append((int(c), fn[:-4], os.path.join(mdir, fn)))
n_samples = len(sample_list)
print(f"     {n_samples:,} NPZ files, {len(classes)} classes  [{time.time()-t0:.1f}s]")

# Probe 200 files to estimate total segments
rng = np.random.default_rng(42)
probe = rng.choice(n_samples, min(200, n_samples), replace=False)
seg_counts = [np.load(sample_list[i][2], allow_pickle=True)["emb"].shape[0] for i in probe]
est_segs = int(np.mean(seg_counts) * n_samples * 1.05)
print(f"     Est. segments: {est_segs:,}  (avg {np.mean(seg_counts):.1f}/img)")

# ── 2. Write flat H5 (single pass over NPZs) ──
print(f"\n2/3  Writing {DST_H5} ...")
t1 = time.time()
os.makedirs(os.path.dirname(DST_H5), exist_ok=True)

with h5py.File(DST_H5, "w") as dst:
    # Flat arrays — resizable in case estimate is low
    emb_ds    = dst.create_dataset("emb",    shape=(est_segs, EMB_DIM), dtype="float32",
                                   maxshape=(None, EMB_DIM), chunks=(min(8192, est_segs), EMB_DIM))
    scores_ds = dst.create_dataset("scores", shape=(est_segs,), dtype="float32",
                                   maxshape=(None,), chunks=(min(65536, est_segs),))
    # Per-sample index (exact size)
    offsets_ds    = dst.create_dataset("offsets",     shape=(n_samples,), dtype="int64")
    n_seg_ds      = dst.create_dataset("n_segments",  shape=(n_samples,), dtype="int32")
    class_ids_ds  = dst.create_dataset("class_ids",   shape=(n_samples,), dtype="int32")
    names_ds      = dst.create_dataset("names",       shape=(n_samples,), dtype=h5py.string_dtype())
    mask_shapes_ds= dst.create_dataset("mask_shapes", shape=(n_samples, 3), dtype="int32")

    cursor = 0
    errs = []
    for si, (cid, name, path) in enumerate(tqdm(sample_list, desc="Writing H5")):
        try:
            d      = np.load(path, allow_pickle=True)
            emb    = d["emb"]       # (N, 256) float32
            sc     = d["scores"]    # (N,)
            shp    = d["shape"]     # (3,)  [N, H, W]
            n      = emb.shape[0]

            # Grow if needed
            if cursor + n > emb_ds.shape[0]:
                new = int(emb_ds.shape[0] * 1.5)
                emb_ds.resize(new, axis=0)
                scores_ds.resize(new, axis=0)

            emb_ds[cursor:cursor+n]    = emb
            scores_ds[cursor:cursor+n] = sc
            offsets_ds[si]    = cursor
            n_seg_ds[si]      = n
            class_ids_ds[si]  = cid
            names_ds[si]      = name
            mask_shapes_ds[si]= shp
            cursor += n
        except Exception as e:
            errs.append((path, str(e)))
            offsets_ds[si]    = cursor
            n_seg_ds[si]      = 0
            class_ids_ds[si]  = cid
            names_ds[si]      = name
            mask_shapes_ds[si]= [0,0,0]

    # Trim
    emb_ds.resize(cursor, axis=0)
    scores_ds.resize(cursor, axis=0)

    dst.attrs["total_samples"]  = n_samples
    dst.attrs["total_segments"] = cursor
    dst.attrs["emb_dim"]        = EMB_DIM
    dst.attrs["emb_dtype"]      = "float32"
    dst.attrs["source"]         = os.path.basename(NPZ_ROOT)

el = time.time()-t0
sz = os.path.getsize(DST_H5)
print(f"\n3/3  Done in {el:.0f}s ({el/60:.1f} min)")
print(f"     {cursor:,} segments, {n_samples:,} samples")
print(f"     File: {sz/(1024**3):.2f} GB")
if errs: print(f"     Errors: {len(errs)}  {errs[:3]}")

# Quick verify
print("\n     Verify (5 random):")
with h5py.File(DST_H5, "r") as f:
    for _ in range(5):
        i = rng.integers(0, n_samples)
        off, ns = int(f["offsets"][i]), int(f["n_segments"][i])
        cid = int(f["class_ids"][i])
        nm  = f["names"][i]; nm = nm.decode() if isinstance(nm, bytes) else nm
        h5e = f["emb"][off:off+ns]
        orig = np.load(os.path.join(NPZ_ROOT, str(cid), "masks_npz", f"{nm}.npz"),
                       allow_pickle=True)["emb"]
        ok = np.array_equal(h5e, orig)
        print(f"       idx={i} class={cid} name={nm} seg={ns} exact_match={ok}")