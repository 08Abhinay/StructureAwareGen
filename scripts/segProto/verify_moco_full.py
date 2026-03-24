#!/usr/bin/env python3
"""Memory-efficient quality verification of MoCo CLS extraction.
Processes one H5 at a time, frees memory between phases.
"""
import os, sys, json, gc
import numpy as np

# Force temp to scratch
os.makedirs("/scratch/gilbreth/abelde/tmp", exist_ok=True)
os.environ["TMPDIR"] = "/scratch/gilbreth/abelde/tmp"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Unbuffered output
print = lambda *a, **kw: __builtins__['print' if isinstance(__builtins__, dict) else 'print'](*a, **kw, flush=True)
import builtins
_print = builtins.print
def print(*a, **kw):
    kw['flush'] = True
    _print(*a, **kw)

import h5py

PROJECT = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen"
H5_MOCO   = f"{PROJECT}/h5_embeddings/moco_cls_flat.h5"
JSON_MOCO = f"{PROJECT}/h5_embeddings/moco_cls_lookup.json"
H5_REGION = f"{PROJECT}/h5_embeddings/region_emb_flat.h5"
H5_IJEPA  = f"{PROJECT}/h5_embeddings/ijepa_emb_flat.h5"

rng = np.random.default_rng(42)

def effective_rank(x, eps=1e-12):
    x = x.astype(np.float64)
    x -= x.mean(0, keepdims=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    ev = np.linalg.eigvalsh(cov).clip(eps)
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))

def cos_stats(x, n_pairs=50000):
    x = x.astype(np.float32).copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
    n = x.shape[0]
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    mask = i != j
    sims = (x[i[mask]] * x[j[mask]]).sum(1)
    return dict(mean=float(sims.mean()), std=float(sims.std()),
                p05=float(np.percentile(sims, 5)),
                p50=float(np.percentile(sims, 50)),
                p95=float(np.percentile(sims, 95)))

print("=" * 65)
print("FULL PRODUCTION MoCo CLS EXTRACTION — QUALITY REPORT")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════════
# PHASE 1: MoCo CLS — load, analyze, save stats, free
# ═══════════════════════════════════════════════════════════════════

print("\n[1] FILE INTEGRITY")
with h5py.File(H5_MOCO, "r") as f:
    shape = f["emb"].shape
    dtype = f["emb"].dtype
    print(f"  H5 shape:  {shape}")
    print(f"  H5 dtype:  {dtype}")
    data = f["emb"][:]
print(f"  Loaded {data.nbytes / 1e9:.2f} GB into memory")

with open(JSON_MOCO) as f:
    lookup = json.load(f)
print(f"  JSON keys: {len(lookup):,}")
diff = shape[0] - len(lookup)
print(f"  Shape vs JSON: {shape[0]} vs {len(lookup)} -> {'MATCH' if diff <= 1 else 'MISMATCH'}")

print("\n[2] NaN / Inf CHECK")
n_nan = int(np.isnan(data).sum())
n_inf = int(np.isinf(data).sum())
print(f"  NaN: {n_nan}  Inf: {n_inf}  Status: {'PASS' if n_nan == 0 and n_inf == 0 else 'FAIL'}")

print("\n[3] Z-SCORE NORMALIZATION")
pm = data.mean(axis=1)
ps = data.std(axis=1)
print(f"  Per-sample mean:  mean={pm.mean():.6f}, std={pm.std():.6f}")
print(f"  Per-sample std:   mean={ps.mean():.6f}, std={ps.std():.6f}")
zscore_ok = abs(pm.mean()) < 0.01 and abs(ps.mean() - 1.0) < 0.05
print(f"  Status: {'PASS' if zscore_ok else 'FAIL'}")
del pm, ps

print("\n[4] L2 NORM")
l2 = np.linalg.norm(data, axis=1)
moco_l2_mean = float(l2.mean())
print(f"  mean={l2.mean():.4f}  std={l2.std():.4f}  min={l2.min():.4f}  max={l2.max():.4f}")
print(f"  Expected sqrt(256)={np.sqrt(256):.4f}")
del l2

print("\n[5] EFFECTIVE RANK (50k subsample)")
idx = rng.choice(len(data), 50000, replace=False)
sub = data[idx].copy()
er_moco = effective_rank(sub)
print(f"  eff_rank: {er_moco:.1f} / 256  ({100 * er_moco / 256:.1f}%)")

print("\n[6] INTER-SAMPLE COSINE SIMILARITY")
cs_moco = cos_stats(sub)
print(f"  mean={cs_moco['mean']:.4f}  std={cs_moco['std']:.4f}")
print(f"  p05={cs_moco['p05']:.4f}  p50={cs_moco['p50']:.4f}  p95={cs_moco['p95']:.4f}")

# FREE all MoCo data
del data, sub, idx
gc.collect()
print("  [memory freed: MoCo data]")

# ═══════════════════════════════════════════════════════════════════
# PHASE 2: Region H5 — just L2 for scale check
# ═══════════════════════════════════════════════════════════════════

print("\n[7] SCALE COMPATIBILITY WITH REGION H5")
with h5py.File(H5_REGION, "r") as f:
    n_reg = f["emb"].shape[0]
    print(f"  Region H5 rows: {n_reg:,}")
    reg_idx = np.sort(rng.choice(n_reg, 50000, replace=False))
    reg_sub = f["emb"][reg_idx]
reg_l2 = float(np.linalg.norm(reg_sub, axis=1).mean())
ratio = moco_l2_mean / reg_l2
print(f"  MoCo CLS L2: {moco_l2_mean:.4f}  Region L2: {reg_l2:.4f}  Ratio: {ratio:.4f}x")
print(f"  Status: {'PASS' if 0.5 < ratio < 2.0 else 'WARNING'}")
del reg_sub
gc.collect()
print("  [memory freed: region data]")

# ═══════════════════════════════════════════════════════════════════
# PHASE 3: I-JEPA H5 — subsample for eff_rank / cos comparison
# ═══════════════════════════════════════════════════════════════════

print("\n[8] COMPARISON: MoCo CLS vs I-JEPA CLS")
with h5py.File(H5_IJEPA, "r") as f:
    ijepa_n = f["emb"].shape[0]
    dim_ij = f["emb"].shape[1]
    print(f"  I-JEPA H5: {ijepa_n:,} x {dim_ij}")
    ij_idx = np.sort(rng.choice(ijepa_n, 50000, replace=False))
    ij_sub = f["emb"][ij_idx]

er_ij = effective_rank(ij_sub)
cs_ij = cos_stats(ij_sub)

print(f"  {'Metric':<25}  {'MoCo (256d)':>14}  {'I-JEPA (' + str(dim_ij) + 'd)':>14}")
print(f"  {'eff_rank':<25}  {er_moco:>14.1f}  {er_ij:>14.1f}")
print(f"  {'eff_rank %':<25}  {100*er_moco/256:>13.1f}%  {100*er_ij/dim_ij:>13.1f}%")
print(f"  {'cos mean':<25}  {cs_moco['mean']:>14.4f}  {cs_ij['mean']:>14.4f}")
print(f"  {'cos std':<25}  {cs_moco['std']:>14.4f}  {cs_ij['std']:>14.4f}")
del ij_sub
gc.collect()

# ═══════════════════════════════════════════════════════════════════
# PHASE 4: Lookup / class checks (no H5 needed)
# ═══════════════════════════════════════════════════════════════════

print("\n[9] LOOKUP KEY FORMAT")
keys = list(lookup.keys())
print(f"  First 3: {keys[:3]}")
print(f"  Last 3:  {keys[-3:]}")
n_valid = sum(1 for k in keys if "/" in k)
print(f"  Valid (class_id/name): {n_valid:,} / {len(keys):,} ({100*n_valid/len(keys):.1f}%)")

print("\n[10] CLASS COVERAGE")
classes = set(k.split("/")[0] for k in keys if "/" in k)
cls_ok = len(classes) == 1000
print(f"  Unique classes: {len(classes)} / 1000  Status: {'PASS' if cls_ok else 'CHECK'}")

print("\n[11] PER-CLASS DISTRIBUTION")
from collections import Counter
counts = list(Counter(k.split("/")[0] for k in keys if "/" in k).values())
print(f"  min={min(counts)}  max={max(counts)}  mean={np.mean(counts):.1f}  std={np.std(counts):.1f}")

# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
all_ok = n_nan == 0 and n_inf == 0 and zscore_ok and cls_ok
print("OVERALL: ALL CHECKS PASSED" if all_ok else "SOME CHECKS NEED ATTENTION")
print("=" * 65)
