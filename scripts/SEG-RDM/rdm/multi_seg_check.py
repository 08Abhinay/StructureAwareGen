#!/usr/bin/env python3
"""
Run sample_quality_check across multiple num_segments values in one process.
Loads model once, generates samples for each requested segment count.
"""
import os, sys, json
import numpy as np
import torch
import h5py
from pathlib import Path

RDM_ROOT = Path(__file__).resolve().parent
SEG_RDM_ROOT = RDM_ROOT.parent
sys.path.insert(0, str(SEG_RDM_ROOT))

from omegaconf import OmegaConf
from rdm import util

# ── helpers ──────────────────────────────────────────────────────────────────

def effective_rank(X):
    if X.shape[0] < 2:
        return 0.0
    _, s, _ = np.linalg.svd(X - X.mean(axis=0, keepdims=True), full_matrices=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def cos_stats(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norms, 1e-8, None)
    sim = Xn @ Xn.T
    triu = np.triu_indices(len(X), k=1)
    vals = sim[triu]
    return {"mean": float(vals.mean()), "std": float(vals.std())}


# ── config ────────────────────────────────────────────────────────────────────

CONFIG   = str(RDM_ROOT / 'configs' / 'unified_seg_rdm_hybrid.yaml')
CKPT     = str(RDM_ROOT / 'rdm_out_final' / 'hybrid_moco_cls' / '2_nodes' / 'checkpoint-last.pth')
MOCO_H5  = str(RDM_ROOT.parent.parent / 'h5_embeddings' / 'moco_cls_flat.h5')
N_SAMPLES = 64
SEG_COUNTS = [64]

print("=" * 70)
print("Loading model …")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = OmegaConf.load(CONFIG)
ckpt   = torch.load(CKPT, map_location='cpu', weights_only=False)
epoch  = ckpt.get('epoch', '?')
print(f"  Checkpoint epoch: {epoch}")

if ckpt.get('model_ema') is not None:
    print("  Using EMA weights")
    state_dict = ckpt['model_ema']
else:
    print("  Using raw weights")
    state_dict = ckpt['model']

model = util.load_model_from_config(config.model, state_dict, device=str(device))
model.eval()
C = model.channels
print(f"  channels={C}, device={device}")

# ── load real MoCo CLS for comparison ─────────────────────────────────────────
real_cls = None
er_real = cs_real = l2_real = None
if os.path.exists(MOCO_H5):
    with h5py.File(MOCO_H5, 'r') as f:
        n = min(2000, f['emb'].shape[0])
        idx = np.sort(np.random.choice(f['emb'].shape[0], n, replace=False))
        real_cls = f['emb'][idx].astype(np.float32)
    er_real = effective_rank(real_cls[:500])
    cs_real = cos_stats(real_cls[:500])
    l2_real = np.linalg.norm(real_cls, axis=1)
    print(f"\n  Real MoCo CLS (n={n}):")
    print(f"    eff_rank  = {er_real:.2f} / {real_cls.shape[1]}  ({100*er_real/real_cls.shape[1]:.1f}%)")
    print(f"    cos_mean  = {cs_real['mean']:.4f}  ±  {cs_real['std']:.4f}")
    print(f"    L2 mean   = {l2_real.mean():.4f}  ±  {l2_real.std():.4f}")
else:
    print(f"  WARNING: MoCo H5 not found at {MOCO_H5}")

# ── run per num_segments ──────────────────────────────────────────────────────
with torch.no_grad():
    for ns in SEG_COUNTS:
        print(f"\n{'='*70}")
        print(f"num_segments = {ns}  |  n_samples = {N_SAMPLES}")
        print(f"{'='*70}")

        samples = model.sample(
            cond=None,
            batch_size=N_SAMPLES,
            num_segments=ns,
            return_intermediates=False
        )
        tokens = model._from_diffusion_format(samples).cpu().numpy()  # [B, ns+1, C]
        global_tokens = tokens[:, 0, :]   # [B, C]
        seg_tokens    = tokens[:, 1:, :]  # [B, ns, C]

        # ── Global token metrics ──────────────────────────────────────────
        er_g  = effective_rank(global_tokens)
        cs_g  = cos_stats(global_tokens)
        l2_g  = np.linalg.norm(global_tokens, axis=1)
        print(f"\nGlobal tokens (sampled, {C}d):")
        print(f"  eff_rank          = {er_g:.2f} / {C}  ({100*er_g/C:.1f}%)")
        print(f"  inter-cos mean    = {cs_g['mean']:.4f}  ±  {cs_g['std']:.4f}")
        print(f"  L2 mean           = {l2_g.mean():.4f}  ±  {l2_g.std():.4f}")
        print(f"  value mean/std    = {global_tokens.mean():.4f} / {global_tokens.std():.4f}")

        if real_cls is not None:
            # NN cosine retrieval
            gn = global_tokens / np.clip(np.linalg.norm(global_tokens, axis=1, keepdims=True), 1e-8, None)
            rn = real_cls      / np.clip(np.linalg.norm(real_cls,      axis=1, keepdims=True), 1e-8, None)
            nn_sims = (gn @ rn.T).max(axis=1)
            print(f"  NN cos vs real    = {nn_sims.mean():.4f}  ±  {nn_sims.std():.4f}")
            print(f"\nComparison vs Real MoCo CLS ({real_cls.shape[1]}d):")
            print(f"  {'Metric':<24}  {'Sampled':>10}  {'Real':>10}")
            print(f"  {'eff_rank'::<24}  {er_g:>10.2f}  {er_real:>10.2f}")
            print(f"  {'eff_rank %':<24}  {100*er_g/C:>9.1f}%  {100*er_real/real_cls.shape[1]:>9.1f}%")
            print(f"  {'inter-cos mean':<24}  {cs_g['mean']:>10.4f}  {cs_real['mean']:>10.4f}")
            print(f"  {'L2 mean':<24}  {l2_g.mean():>10.4f}  {l2_real.mean():>10.4f}")
            print(f"  {'L2 std':<24}  {l2_g.std():>10.4f}  {l2_real.std():>10.4f}")

        # ── Segment token metrics ─────────────────────────────────────────
        seg_flat = seg_tokens.reshape(-1, C)
        er_seg_all = effective_rank(seg_flat[:2000])
        ers_per = [effective_rank(seg_tokens[i]) for i in range(N_SAMPLES)]
        cs_seg  = cos_stats(seg_flat[:500])
        l2_seg  = np.linalg.norm(seg_flat, axis=1)

        # intra-sample cos
        intra = []
        for i in range(min(N_SAMPLES, 32)):
            sm = seg_tokens[i] / np.clip(np.linalg.norm(seg_tokens[i], axis=1, keepdims=True), 1e-8, None)
            mat = sm @ sm.T
            idx = np.triu_indices(len(sm), k=1)
            intra.append(mat[idx].mean())

        print(f"\nSegment tokens (sampled, {C}d):")
        print(f"  eff_rank (pooled) = {er_seg_all:.2f} / {C}  ({100*er_seg_all/C:.1f}%)")
        print(f"  eff_rank per-sample (avg) = {np.mean(ers_per):.2f} ± {np.std(ers_per):.2f}  ({100*np.mean(ers_per)/C:.1f}%)")
        print(f"  inter-cos mean    = {cs_seg['mean']:.4f}  ±  {cs_seg['std']:.4f}")
        print(f"  intra-cos mean    = {np.mean(intra):.4f}  ±  {np.std(intra):.4f}")
        print(f"  L2 mean           = {l2_seg.mean():.4f}  ±  {l2_seg.std():.4f}")

        # ── Summary verdict ──────────────────────────────────────────────
        print(f"\nSummary:")
        if er_g < 5:
            print(f"  ⚠  Global eff_rank = {er_g:.1f} - LOW (collapse risk)")
        elif er_g < 20:
            print(f"  △  Global eff_rank = {er_g:.1f} - moderate (still converging)")
        else:
            print(f"  ✓  Global eff_rank = {er_g:.1f} - good diversity")

        if cs_g['mean'] > 0.5:
            print(f"  ⚠  High inter-cos ({cs_g['mean']:.3f}) → sampled globals are too similar")
        elif cs_g['mean'] > 0.2:
            print(f"  △  Moderate inter-cos ({cs_g['mean']:.3f}) → some diversity")
        else:
            print(f"  ✓  Low inter-cos ({cs_g['mean']:.3f}) → good diversity")

print(f"\n{'='*70}")
