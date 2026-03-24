#!/usr/bin/env python3
"""
Sample from UnifiedSegRDM checkpoint and evaluate embedding quality.

Since the RDM generates embedding sequences (not images), quality evaluation
compares generated embeddings against real embeddings from the H5 database.

Metrics:
  1. Per-sample statistics: mean, std, min, max of generated tokens
  2. Effective rank of generated segment tokens vs real segment tokens
  3. Cosine similarity structure within generated samples
  4. Global vs segment token separation (should be distinguishable)
  5. Nearest-neighbor retrieval from real embedding database
"""

import os, sys, json
import numpy as np
import torch
import h5py
from pathlib import Path

# Add rdm to path
RDM_ROOT = Path(__file__).resolve().parent
SEG_RDM_ROOT = RDM_ROOT.parent
sys.path.insert(0, str(SEG_RDM_ROOT))

from omegaconf import OmegaConf
from rdm import util


def effective_rank(X: np.ndarray) -> float:
    """Compute effective rank = exp(entropy of normalized singular values)."""
    if X.shape[0] < 2:
        return 0.0
    _, s, _ = np.linalg.svd(X - X.mean(axis=0, keepdims=True), full_matrices=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * np.log(p + 1e-12)).sum()
    return float(np.exp(entropy))


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    X_normed = X / norms
    return X_normed @ X_normed.T


def compute_mean_cov(X: np.ndarray):
    """Return sample mean/covariance for [N, D] embeddings."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array [N, D], got shape={X.shape}")
    if X.shape[0] < 2:
        raise ValueError(f"Need at least 2 samples for covariance, got N={X.shape[0]}")
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov = 0.5 * (cov + cov.T)  # Ensure symmetry
    return mu, cov


def frechet_distance_from_stats(mu_real, cov_real, mu_gen, cov_gen, eps=1e-6) -> float:
    """
    Numerically-stable Fréchet distance between two Gaussians.

    FID(mu_r, Sigma_r, mu_g, Sigma_g) =
      ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 * sqrt(Sigma_r^1/2 * Sigma_g * Sigma_r^1/2))
    """
    mu_real = np.asarray(mu_real, dtype=np.float64)
    mu_gen = np.asarray(mu_gen, dtype=np.float64)
    cov_real = np.asarray(cov_real, dtype=np.float64)
    cov_gen = np.asarray(cov_gen, dtype=np.float64)

    d = mu_real.shape[0]
    eye = np.eye(d, dtype=np.float64)

    cov_real = 0.5 * (cov_real + cov_real.T) + eps * eye
    cov_gen = 0.5 * (cov_gen + cov_gen.T) + eps * eye

    # sqrt(cov_real) via eigendecomposition (PSD-safe)
    evals_r, evecs_r = np.linalg.eigh(cov_real)
    evals_r = np.clip(evals_r, 0.0, None)
    sqrt_cov_real = (evecs_r * np.sqrt(evals_r)) @ evecs_r.T

    middle = sqrt_cov_real @ cov_gen @ sqrt_cov_real
    middle = 0.5 * (middle + middle.T)
    evals_m = np.linalg.eigvalsh(middle)
    evals_m = np.clip(evals_m, 0.0, None)
    tr_sqrt = np.sum(np.sqrt(evals_m))

    diff = mu_real - mu_gen
    fid = float(diff @ diff + np.trace(cov_real) + np.trace(cov_gen) - 2.0 * tr_sqrt)
    # Small negative values can happen due to floating point noise.
    return max(fid, 0.0)


@torch.no_grad()
def generate_embeddings_for_rep_fid(
    model,
    n_global: int,
    num_segments: int,
    batch_size: int,
    n_segment_tokens: int = 0,
):
    """Generate embeddings for rep-FID in mini-batches."""
    if batch_size <= 0:
        raise ValueError(f"rep_fid_batch_size must be > 0, got {batch_size}")

    want_global = n_global > 0
    want_seg = n_segment_tokens > 0
    got_global = 0
    got_seg = 0
    global_chunks = []
    seg_chunks = []

    print("\n" + "="*60)
    print("rep-FID: generating embeddings")
    print("="*60)
    print(f"   Target generated globals: {n_global}")
    if want_seg:
        print(f"   Target generated segment tokens: {n_segment_tokens}")
    print(f"   Generation batch size: {batch_size}")

    while (want_global and got_global < n_global) or (want_seg and got_seg < n_segment_tokens):
        if want_global and got_global < n_global:
            cur_bs = min(batch_size, n_global - got_global)
        else:
            # If only segment tokens remain, estimate how many full samples are needed.
            tokens_per_sample = max(1, num_segments)
            needed_samples = int(np.ceil((n_segment_tokens - got_seg) / tokens_per_sample))
            cur_bs = min(batch_size, needed_samples)

        samples = model.sample(
            cond=None,
            batch_size=cur_bs,
            num_segments=num_segments,
            return_intermediates=False
        )
        tokens = model._from_diffusion_format(samples).cpu().numpy()  # [B, N+1, C]

        if want_global and got_global < n_global:
            remaining = n_global - got_global
            take = min(tokens.shape[0], remaining)
            global_chunks.append(tokens[:take, 0, :].astype(np.float32, copy=False))
            got_global += take

        if want_seg and got_seg < n_segment_tokens:
            remaining = n_segment_tokens - got_seg
            seg_flat = tokens[:, 1:, :].reshape(-1, tokens.shape[-1]).astype(np.float32, copy=False)
            take = min(seg_flat.shape[0], remaining)
            seg_chunks.append(seg_flat[:take])
            got_seg += take

        msg = f"   Progress globals: {got_global}/{n_global}"
        if want_seg:
            msg += f" | segment tokens: {got_seg}/{n_segment_tokens}"
        print(msg)

    gen_global = np.concatenate(global_chunks, axis=0) if want_global else None
    gen_seg = np.concatenate(seg_chunks, axis=0) if want_seg else None
    return gen_global, gen_seg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default=str(RDM_ROOT / 'configs' / 'unified_seg_rdm_hybrid.yaml'))
    parser.add_argument('--ckpt', type=str,
                        default=str(RDM_ROOT / 'rdm_out_final' / 'hybrid_moco_cls' / '2_nodes' / 'checkpoint-last.pth'))
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA weights (default: True)')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false')
    parser.add_argument('--n_samples', type=int, default=16,
                        help='Number of samples to generate')
    parser.add_argument('--num_segments', type=int, default=64,
                        help='Number of segment tokens per sample')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save analysis results')
    parser.add_argument('--compute_rep_fid', action='store_true', default=False,
                        help='Compute representation Fréchet Distance (rep-FID) for global tokens')
    parser.add_argument('--compute_rep_fid_segments', action='store_true', default=False,
                        help='Also compute rep-FID for segment tokens vs real region embeddings')
    parser.add_argument('--rep_fid_n_gen', type=int, default=50000,
                        help='Number of generated global embeddings for rep-FID')
    parser.add_argument('--rep_fid_n_real', type=int, default=50000,
                        help='Number of real global embeddings for rep-FID')
    parser.add_argument('--rep_fid_n_seg_gen', type=int, default=50000,
                        help='Number of generated segment tokens for segment rep-FID')
    parser.add_argument('--rep_fid_n_seg_real', type=int, default=50000,
                        help='Number of real segment tokens for segment rep-FID')
    parser.add_argument('--rep_fid_batch_size', type=int, default=16,
                        help='Batch size used while generating embeddings for rep-FID')
    parser.add_argument('--rep_fid_eps', type=float, default=1e-6,
                        help='Covariance diagonal regularization for rep-FID')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 1. Load config and model ──
    print("\n" + "="*60)
    print("1. Loading model from checkpoint")
    print("="*60)
    config = OmegaConf.load(args.config)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch', '?')
    print(f"   Checkpoint epoch: {epoch}")

    # Choose weights: EMA or raw
    if args.use_ema and ckpt.get('model_ema') is not None:
        print("   Using EMA weights")
        state_dict = ckpt['model_ema']
    else:
        print("   Using raw (non-EMA) weights")
        state_dict = ckpt['model']

    model = util.load_model_from_config(config.model, state_dict, device=str(device))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {n_params/1e6:.2f}M")

    # ── 2. Generate samples ──
    print("\n" + "="*60)
    print(f"2. Generating {args.n_samples} samples ({args.num_segments} segments each)")
    print("="*60)
    with torch.no_grad():
        # Shape: [B, channels, 1, num_segments+1]
        shape = (args.n_samples, model.channels, 1, args.num_segments + 1)
        print(f"   Sample shape: {shape}")
        print(f"   Timesteps: {model.num_timesteps}")
        print(f"   Parameterization: {model.parameterization}")

        # Use the model's sample method (cond=None for unconditional)
        samples = model.sample(
            cond=None,
            batch_size=args.n_samples,
            num_segments=args.num_segments,
            return_intermediates=False
        )

    # Convert from diffusion format [B, C, 1, N+1] → [B, N+1, C]
    samples_tokens = model._from_diffusion_format(samples)  # [B, N+1, C]
    samples_np = samples_tokens.cpu().numpy()
    B, N_plus_1, C = samples_np.shape
    print(f"   Generated tensor shape: {samples_np.shape}")

    # Split global and segment tokens
    global_tokens = samples_np[:, 0, :]          # [B, C]
    segment_tokens = samples_np[:, 1:, :]         # [B, N, C]

    # ── 3. Basic Statistics ──
    print("\n" + "="*60)
    print("3. Basic Statistics of Generated Tokens")
    print("="*60)
    print(f"\n   All tokens:")
    print(f"     Mean:   {samples_np.mean():.4f}")
    print(f"     Std:    {samples_np.std():.4f}")
    print(f"     Min:    {samples_np.min():.4f}")
    print(f"     Max:    {samples_np.max():.4f}")

    print(f"\n   Global tokens (position 0, MoCo CLS):")
    print(f"     Mean:   {global_tokens.mean():.4f}")
    print(f"     Std:    {global_tokens.std():.4f}")
    print(f"     Min:    {global_tokens.min():.4f}")
    print(f"     Max:    {global_tokens.max():.4f}")
    print(f"     Per-sample L2 norms: {np.linalg.norm(global_tokens, axis=1).mean():.4f} ± {np.linalg.norm(global_tokens, axis=1).std():.4f}")

    print(f"\n   Segment tokens (positions 1..{N_plus_1-1}, I-JEPA regions):")
    seg_flat = segment_tokens.reshape(-1, C)
    print(f"     Mean:   {seg_flat.mean():.4f}")
    print(f"     Std:    {seg_flat.std():.4f}")
    print(f"     Min:    {seg_flat.min():.4f}")
    print(f"     Max:    {seg_flat.max():.4f}")
    print(f"     Per-token L2 norms: {np.linalg.norm(seg_flat, axis=1).mean():.4f} ± {np.linalg.norm(seg_flat, axis=1).std():.4f}")

    # ── 4. Effective Rank ──
    print("\n" + "="*60)
    print("4. Effective Rank Analysis")
    print("="*60)

    # Global tokens effective rank (across samples)
    eff_rank_global = effective_rank(global_tokens)
    print(f"   Global tokens (across {B} samples): eff_rank = {eff_rank_global:.2f} / {C} ({100*eff_rank_global/C:.1f}%)")

    # Segment tokens effective rank (per sample, then averaged)
    eff_ranks_per_sample = []
    for i in range(B):
        er = effective_rank(segment_tokens[i])
        eff_ranks_per_sample.append(er)
    mean_er = np.mean(eff_ranks_per_sample)
    std_er = np.std(eff_ranks_per_sample)
    print(f"   Segment tokens (per sample, avg): eff_rank = {mean_er:.2f} ± {std_er:.2f} / {C} ({100*mean_er/C:.1f}%)")

    # All segment tokens pooled
    eff_rank_all_seg = effective_rank(seg_flat[:2000])  # Cap for speed
    print(f"   All segment tokens (pooled, up to 2000): eff_rank = {eff_rank_all_seg:.2f} / {C} ({100*eff_rank_all_seg/C:.1f}%)")

    # ── 5. Cosine Similarity Structure ──
    print("\n" + "="*60)
    print("5. Cosine Similarity Structure")
    print("="*60)

    # Within-sample segment-segment similarity
    intra_sims = []
    for i in range(B):
        sim_mat = cosine_sim_matrix(segment_tokens[i])
        # Get upper triangle (excluding diagonal)
        triu_idx = np.triu_indices(sim_mat.shape[0], k=1)
        pairwise = sim_mat[triu_idx]
        intra_sims.append(pairwise.mean())
    mean_intra = np.mean(intra_sims)
    std_intra = np.std(intra_sims)
    print(f"   Mean intra-sample seg-seg cosine sim: {mean_intra:.4f} ± {std_intra:.4f}")
    print(f"   (Should be moderate ~0.1-0.5 for diverse segments, ~1.0 = collapsed)")

    # Global-to-segment similarity
    global_seg_sims = []
    for i in range(B):
        g = global_tokens[i]
        g_norm = g / (np.linalg.norm(g) + 1e-8)
        s_norms = segment_tokens[i] / (np.linalg.norm(segment_tokens[i], axis=1, keepdims=True) + 1e-8)
        sims = (g_norm[None, :] @ s_norms.T).flatten()
        global_seg_sims.append(sims.mean())
    mean_gs = np.mean(global_seg_sims)
    std_gs = np.std(global_seg_sims)
    print(f"   Mean global-segment cosine sim: {mean_gs:.4f} ± {std_gs:.4f}")

    # Cross-sample segment similarity (different samples should differ)
    if B >= 2:
        cross_sims = []
        for i in range(min(B, 8)):
            for j in range(i+1, min(B, 8)):
                # Compare mean segment embeddings across samples
                mean_i = segment_tokens[i].mean(axis=0)
                mean_j = segment_tokens[j].mean(axis=0)
                sim = np.dot(mean_i, mean_j) / (np.linalg.norm(mean_i) * np.linalg.norm(mean_j) + 1e-8)
                cross_sims.append(sim)
        print(f"   Mean cross-sample similarity (of mean segs): {np.mean(cross_sims):.4f} ± {np.std(cross_sims):.4f}")
        print(f"   (Should be lower than intra-sample sim → model generates diverse samples)")

    # ── 6. Token Distinguishability ──
    print("\n" + "="*60)
    print("6. Global vs Segment Token Distinguishability")
    print("="*60)
    # Check if global tokens live in a different subspace than segment tokens
    # by comparing their mean/std profiles
    global_mean_per_dim = global_tokens.mean(axis=0)
    seg_mean_per_dim = seg_flat.mean(axis=0)
    dim_corr = np.corrcoef(global_mean_per_dim, seg_mean_per_dim)[0, 1]
    print(f"   Correlation of per-dim means (global vs seg): {dim_corr:.4f}")
    print(f"   (Low correlation → model differentiates global/segment roles)")

    l2_diff = np.linalg.norm(global_mean_per_dim - seg_mean_per_dim)
    print(f"   L2 distance between mean global and mean segment: {l2_diff:.4f}")

    # ── 7. Compare with Real Embeddings ──
    print("\n" + "="*60)
    print("7. Comparison with Real Embeddings")
    print("="*60)

    # Load real MoCo CLS embeddings
    moco_h5_path = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/moco_cls_flat.h5"
    region_h5_path = "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/region_emb_flat.h5"
    real_cls = None
    real_seg = None

    if os.path.exists(moco_h5_path):
        with h5py.File(moco_h5_path, 'r') as f:
            # Sample some real MoCo CLS embeddings
            n_real = min(1000, f['emb'].shape[0])
            idx = np.random.choice(f['emb'].shape[0], n_real, replace=False)
            idx.sort()
            real_cls = f['emb'][idx]
        print(f"   Real MoCo CLS (n={n_real}):")
        print(f"     Mean: {real_cls.mean():.4f}, Std: {real_cls.std():.4f}")
        print(f"     L2 norm: {np.linalg.norm(real_cls, axis=1).mean():.4f} ± {np.linalg.norm(real_cls, axis=1).std():.4f}")
        eff_rank_real_cls = effective_rank(real_cls[:500])
        print(f"     Eff rank: {eff_rank_real_cls:.2f} / {C} ({100*eff_rank_real_cls/C:.1f}%)")

        # Compare generated global vs real MoCo CLS
        print(f"\n   Generated global tokens vs real MoCo CLS:")
        print(f"     Gen mean: {global_tokens.mean():.4f} vs Real mean: {real_cls.mean():.4f}")
        print(f"     Gen std: {global_tokens.std():.4f} vs Real std: {real_cls.std():.4f}")
        print(f"     Gen L2: {np.linalg.norm(global_tokens, axis=1).mean():.4f} vs Real L2: {np.linalg.norm(real_cls, axis=1).mean():.4f}")

        # NN retrieval: for each generated global token, find nearest real one
        gen_norms = global_tokens / (np.linalg.norm(global_tokens, axis=1, keepdims=True) + 1e-8)
        real_norms = real_cls / (np.linalg.norm(real_cls, axis=1, keepdims=True) + 1e-8)
        nn_sims = (gen_norms @ real_norms.T).max(axis=1)
        print(f"     NN cosine sim to real MoCo CLS: {nn_sims.mean():.4f} ± {nn_sims.std():.4f} (max: {nn_sims.max():.4f})")
    else:
        print(f"   WARNING: MoCo CLS H5 not found at {moco_h5_path}")

    if os.path.exists(region_h5_path):
        with h5py.File(region_h5_path, 'r') as f:
            n_real = min(2000, f['emb'].shape[0])
            idx = np.random.choice(f['emb'].shape[0], n_real, replace=False)
            idx.sort()
            real_seg = f['emb'][idx]
        print(f"\n   Real I-JEPA region embeddings (n={n_real}):")
        print(f"     Mean: {real_seg.mean():.4f}, Std: {real_seg.std():.4f}")
        print(f"     L2 norm: {np.linalg.norm(real_seg, axis=1).mean():.4f} ± {np.linalg.norm(real_seg, axis=1).std():.4f}")
        eff_rank_real_seg = effective_rank(real_seg[:500])
        print(f"     Eff rank: {eff_rank_real_seg:.2f} / {C} ({100*eff_rank_real_seg/C:.1f}%)")

        # Compare generated segment tokens vs real
        print(f"\n   Generated segment tokens vs real I-JEPA regions:")
        print(f"     Gen mean: {seg_flat.mean():.4f} vs Real mean: {real_seg.mean():.4f}")
        print(f"     Gen std: {seg_flat.std():.4f} vs Real std: {real_seg.std():.4f}")
        print(f"     Gen L2: {np.linalg.norm(seg_flat, axis=1).mean():.4f} vs Real L2: {np.linalg.norm(real_seg, axis=1).mean():.4f}")

        # NN retrieval for segments
        seg_sample = seg_flat[:100]  # First 100 generated segments
        seg_norms_gen = seg_sample / (np.linalg.norm(seg_sample, axis=1, keepdims=True) + 1e-8)
        real_seg_norms = real_seg / (np.linalg.norm(real_seg, axis=1, keepdims=True) + 1e-8)
        nn_sims_seg = (seg_norms_gen @ real_seg_norms.T).max(axis=1)
        print(f"     NN cosine sim to real I-JEPA regions: {nn_sims_seg.mean():.4f} ± {nn_sims_seg.std():.4f}")
    else:
        print(f"   WARNING: Region H5 not found at {region_h5_path}")

    # ── 7b. Representation Fréchet Distance (rep-FID) ──
    rep_fid_global = None
    rep_fid_segment = None
    if args.compute_rep_fid or args.compute_rep_fid_segments:
        print("\n" + "="*60)
        print("7b. Representation Fréchet Distance (rep-FID)")
        print("="*60)

        need_global = args.compute_rep_fid
        need_seg = args.compute_rep_fid_segments
        real_global_fid = None
        real_seg_fid = None

        if need_global:
            if os.path.exists(moco_h5_path):
                with h5py.File(moco_h5_path, 'r') as f:
                    n_real_fid = min(args.rep_fid_n_real, f['emb'].shape[0])
                    idx = np.random.choice(f['emb'].shape[0], n_real_fid, replace=False)
                    idx.sort()
                    real_global_fid = f['emb'][idx].astype(np.float32)
                print(f"   Global rep-FID real set: n={real_global_fid.shape[0]} from {moco_h5_path}")
            else:
                print(f"   WARNING: Cannot compute global rep-FID (missing {moco_h5_path})")
                need_global = False

        if need_seg:
            if os.path.exists(region_h5_path):
                with h5py.File(region_h5_path, 'r') as f:
                    n_real_seg_fid = min(args.rep_fid_n_seg_real, f['emb'].shape[0])
                    idx = np.random.choice(f['emb'].shape[0], n_real_seg_fid, replace=False)
                    idx.sort()
                    real_seg_fid = f['emb'][idx].astype(np.float32)
                print(f"   Segment rep-FID real set: n={real_seg_fid.shape[0]} from {region_h5_path}")
            else:
                print(f"   WARNING: Cannot compute segment rep-FID (missing {region_h5_path})")
                need_seg = False

        if need_global or need_seg:
            n_gen_global = args.rep_fid_n_gen if need_global else 0
            n_gen_seg = args.rep_fid_n_seg_gen if need_seg else 0

            if need_global and n_gen_global <= 0:
                n_gen_global = max(1, args.n_samples)
                print(f"   NOTE: rep_fid_n_gen <= 0, using n_samples={n_gen_global} for generated globals")
            if need_seg and n_gen_seg <= 0:
                n_gen_seg = max(1, args.n_samples * args.num_segments)
                print(f"   NOTE: rep_fid_n_seg_gen <= 0, using n_samples*num_segments={n_gen_seg}")

            gen_global_fid, gen_seg_fid = generate_embeddings_for_rep_fid(
                model=model,
                n_global=n_gen_global,
                num_segments=args.num_segments,
                batch_size=args.rep_fid_batch_size,
                n_segment_tokens=n_gen_seg
            )

            if need_global:
                mu_real, cov_real = compute_mean_cov(real_global_fid)
                mu_gen, cov_gen = compute_mean_cov(gen_global_fid)
                rep_fid_global = frechet_distance_from_stats(
                    mu_real, cov_real, mu_gen, cov_gen, eps=args.rep_fid_eps
                )
                print(f"   rep-FID (global, n_real={real_global_fid.shape[0]}, n_gen={gen_global_fid.shape[0]}): {rep_fid_global:.4f}")

            if need_seg:
                mu_real_s, cov_real_s = compute_mean_cov(real_seg_fid)
                mu_gen_s, cov_gen_s = compute_mean_cov(gen_seg_fid)
                rep_fid_segment = frechet_distance_from_stats(
                    mu_real_s, cov_real_s, mu_gen_s, cov_gen_s, eps=args.rep_fid_eps
                )
                print(f"   rep-FID (segment, n_real={real_seg_fid.shape[0]}, n_gen={gen_seg_fid.shape[0]}): {rep_fid_segment:.4f}")

    # ── 8. Summary ──
    print("\n" + "="*60)
    print("8. QUALITY SUMMARY")
    print("="*60)
    print(f"   Epoch: {epoch}")
    print(f"   Samples: {B} × {N_plus_1} tokens × {C}d")
    print(f"   Global eff rank: {eff_rank_global:.1f}/{C} ({100*eff_rank_global/C:.0f}%)")
    print(f"   Segment eff rank (per sample): {mean_er:.1f} ± {std_er:.1f}/{C} ({100*mean_er/C:.0f}%)")
    print(f"   Intra-sample seg-seg sim: {mean_intra:.3f}")
    print(f"   Global-segment sim: {mean_gs:.3f}")
    if B >= 2:
        print(f"   Cross-sample sim: {np.mean(cross_sims):.3f}")
    if rep_fid_global is not None:
        print(f"   rep-FID (global): {rep_fid_global:.4f}")
    if rep_fid_segment is not None:
        print(f"   rep-FID (segment): {rep_fid_segment:.4f}")

    # Verdict
    print(f"\n   Verdict:")
    if mean_intra > 0.95:
        print(f"   ⚠ HIGH intra-sample similarity ({mean_intra:.3f}) → possible segment collapse")
    elif mean_intra > 0.7:
        print(f"   △ Moderate intra-sample similarity ({mean_intra:.3f}) → model still converging")
    else:
        print(f"   ✓ Good intra-sample diversity ({mean_intra:.3f})")

    if eff_rank_global < 5:
        print(f"   ⚠ LOW global token diversity (eff_rank={eff_rank_global:.1f}) → mode collapse risk")
    else:
        print(f"   ✓ Global token diversity OK (eff_rank={eff_rank_global:.1f})")

    if mean_er < 5:
        print(f"   ⚠ LOW segment diversity (eff_rank={mean_er:.1f}) → segment collapse risk")
    else:
        print(f"   ✓ Segment diversity OK (eff_rank={mean_er:.1f})")

    print("\n   Note: At epoch ~20, high loss is expected. Quality improves")
    print("   significantly after 100+ epochs. These metrics establish a baseline.")

    # Save samples if requested
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(os.path.join(args.output_dir, f'samples_epoch{epoch}.npy'), samples_np)
        print(f"\n   Samples saved to {args.output_dir}/samples_epoch{epoch}.npy")


if __name__ == '__main__':
    main()
