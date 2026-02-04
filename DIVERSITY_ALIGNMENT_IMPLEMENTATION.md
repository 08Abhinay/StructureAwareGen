# Diversity & Alignment Losses + RDM Mixed Training Implementation

## Overview

This implementation adds **diversity and alignment losses** to prevent embedding collapse in both Stage 1 (RDM) and Stage 2 (StyleGAN2), plus **mixed training** with RDM-sampled embeddings to reduce train/inference distribution mismatch.

## Implementation Summary

### ✅ Stage 1: RDM (Representation Diffusion Model)

**File:** `scripts/SEG-RDM/rdm/models/diffusion/ddpm.py`

**Changes:**
1. Added `lambda_diversity` and `lambda_alignment` hyperparameters to `UnifiedSegRDM.__init__`
2. Implemented `compute_diversity_loss()` method:
   - Computes covariance matrix of segment embeddings
   - Loss: `-log det(C + εI)` penalizes collapsed/degenerate spectra
   - Encourages full-rank, diverse mask embeddings
3. Implemented `compute_alignment_loss()` method:
   - InfoNCE-style contrastive loss between global (I-JEPA) and segment (SAM) embeddings
   - Uses in-batch negatives for discrimination
   - Prevents trivial collapse where all embeddings converge
4. Modified `p_losses()` to include diversity + alignment losses:
   - Applied only during training (not validation)
   - Reconstructs x₀ from model output (works for both eps and x0 parameterization)
   - Properly handles padding masks

**Default Hyperparameters:**
- `lambda_diversity = 0.1`
- `lambda_alignment = 0.05`

**Training Command Example:**
```bash
python scripts/SEG-RDM/train_unified_seg_rdm.py \
    --config configs/rdm_config.yaml \
    --lambda_diversity 0.1 \
    --lambda_alignment 0.05
```

---

### ✅ Stage 2: StyleGAN2 (Pixel Decoder)

**Files Modified:**
- `scripts/StyleGAN2/seg-aware-stylegan2/training/loss.py`
- `scripts/StyleGAN2/seg-aware-stylegan2/training/training_loop.py`
- `scripts/StyleGAN2/seg-aware-stylegan2/training/rdm_sampler.py` (new)

**Changes in `loss.py`:**
1. Added `lambda_seg_align` and `lambda_seg_diversity` parameters to `StyleGAN2Loss.__init__`
2. Created SAM→I-JEPA projection MLP (256D → 512D → 2048D):
   - Allows alignment loss between different embedding spaces
   - Only created when `sam_enabled=True` and `lambda_seg_align > 0`
3. Implemented `compute_seg_diversity_loss()`:
   - Uses pairwise cosine similarity (O(N²) but efficient for monitoring)
   - Monitors real embeddings to detect collapse
   - Lower loss = more diverse
4. Implemented `compute_seg_alignment_loss()`:
   - Projects pooled SAM embeddings to I-JEPA space
   - Computes cosine similarity loss
   - Ensures semantic coherence between global and local representations
5. Integrated losses into `Gmain` phase:
   - Applied only when SAM is enabled (`real_seg_tokens is not None`)
   - Scaled by `sem_ramp` for gradual fade-in
   - Separate backward passes for each loss component

**Default Hyperparameters:**
- `lambda_seg_align = 0.1`
- `lambda_seg_diversity = 0.05`

**Changes in `training_loop.py`:**
1. Added RDM sampler initialization after loss construction:
   - Loads pretrained RDM checkpoint
   - Uses `CachedRDMSampler` for efficiency (pre-generates 500 samples)
   - Only on rank 0 for simplicity (can be distributed later)
2. Implemented mixed training logic in batch preparation:
   - After warmup (`rdm_warmup_kimg`), randomly replace real embeddings with RDM-sampled
   - Mix probability ramps up gradually over 10M images
   - Keeps real images but uses synthetic embeddings for conditioning
   - Teaches decoder to handle RDM-sampled embeddings

**New Hyperparameters:**
- `rdm_checkpoint`: Path to pretrained RDM checkpoint (`.pt` or `.ckpt`)
- `rdm_mix_prob`: Maximum probability of using RDM samples (default: 0.0, set to 0.1-0.3)
- `rdm_warmup_kimg`: Training kimg before starting RDM mixing (default: 10000)

**Training Command Example:**
```bash
python scripts/StyleGAN2/seg-aware-stylegan2/train.py \
    --outdir=training-runs \
    --data=/path/to/dataset \
    --gpus=4 \
    --batch=32 \
    --gamma=10 \
    --lambda_ijepa=1.0 \
    --ijepa_ckpt=/path/to/ijepa_checkpoint.pth \
    --sam_enabled=True \
    --sam_checkpoint=/path/to/sam_vit_b.pth \
    --sam_cache_dir=/path/to/sam_cache \
    --lambda_seg_align=0.1 \
    --lambda_seg_diversity=0.05 \
    --rdm_checkpoint=/path/to/rdm_checkpoint.pt \
    --rdm_mix_prob=0.2 \
    --rdm_warmup_kimg=10000
```

---

### ✅ RDM Sampler Wrapper

**File:** `scripts/StyleGAN2/seg-aware-stylegan2/training/rdm_sampler.py` (new)

**Features:**
1. `RDMSampler` class:
   - Loads pretrained `UnifiedSegRDM` from checkpoint
   - Supports both EMA and non-EMA weights
   - Uses DDIM for fast sampling (50 steps instead of 1000)
   - Returns dict with `global_vectors`, `seg_tokens`, `num_segments`
2. `CachedRDMSampler` class:
   - Pre-generates cache of samples for efficiency
   - Automatic cache refresh when exhausted
   - Default cache size: 500-1000 samples
   - Much faster than on-the-fly sampling during training

**Usage:**
```python
from training.rdm_sampler import RDMSampler, CachedRDMSampler

# Initialize
sampler = RDMSampler(
    rdm_checkpoint_path='path/to/rdm.pt',
    device='cuda',
    use_ema=True,
    ddim_steps=50
)

# Sample
samples = sampler.sample(batch_size=16, num_segments=180)
# samples['global_vectors']: [16, 256]
# samples['seg_tokens']: [16, 180, 256]

# Or use cached sampler for training
cached = CachedRDMSampler(sampler, cache_size=500)
batch = cached.sample(batch_size=8)
```

---

### ✅ Embedding Distribution Monitor

**File:** `scripts/embedding_distribution_monitor.py` (new)

**Features:**
1. Diversity metrics:
   - Pairwise cosine similarity (want low for diversity)
   - Covariance eigenvalue spectrum
   - Effective rank (number of significant dimensions)
   - Condition number (robustness indicator)
2. Alignment metrics:
   - Cosine similarity between global and segment embeddings
3. Distribution gap:
   - Compares real vs generated embedding statistics
   - Detects train/inference mismatch

**Usage:**
```python
from scripts.embedding_distribution_monitor import EmbeddingDistributionMonitor

monitor = EmbeddingDistributionMonitor(device='cuda')

# Update with real embeddings
monitor.update_stats('real', global_emb, seg_tokens, seg_pad_mask)

# Update with generated/RDM-sampled embeddings
monitor.update_stats('generated', gen_global, gen_seg, gen_mask)

# Print summary
monitor.print_summary()

# Save to file
monitor.save_stats('embedding_stats.json')
```

**Integration into training loop:**
- Can be called periodically (e.g., every 1000 iterations)
- Monitor diversity/alignment trends over training
- Alert if distribution gap increases (potential collapse)

---

## Training Strategy

### Unified SAM Cache Strategy

Both RDM and StyleGAN2 use identical SAM embeddings, so we use a **unified cache directory** to avoid duplicate extraction:

**Why Unified Cache?**
- **Efficiency**: Extract once, use twice (RDM training + StyleGAN2 training)
- **Storage**: No duplicate .npz files (~1-2 TB saved for full ImageNet)
- **Compatibility**: Both systems use identical format: `{'emb': [N,256] float16, 'scores': [N] float32}`

**Pre-extraction for RDM (Recommended):**

Pre-extract SAM embeddings for the 40-50% subset used in RDM training. This eliminates on-the-fly extraction overhead.

```bash
# Extract SAM embeddings for 40% of ImageNet (parallel, 4 GPUs):
torchrun --nproc_per_node=4 scripts/precompute_sam_embeddings.py \
    --data_path /path/to/imagenet/train \
    --cache_dir /shared/sam_cache \
    --subset_fraction 0.4 \
    --seed 42 \
    --batch_size 8

# Verify cache completeness before training:
python scripts/verify_sam_cache.py \
    --data_path /path/to/imagenet/train \
    --cache_dir /shared/sam_cache \
    --subset_fraction 0.4
```

**Extraction Speed**: 4 GPUs process ~500k images in 3.5-7 hours (vs 14-28 hours single GPU).

**Cache Directory Structure:**
```
/shared/sam_cache/
├── n01440764/           # Class folder
│   ├── n01440764_1.npz  # Image embeddings
│   ├── n01440764_2.npz
│   └── ...
├── n01443537/
│   └── ...
```

Each `.npz` file contains:
- `emb`: (N, 256) SAM segment embeddings, float16
- `scores`: (N,) Predicted IoU scores, float32

---

### Phase 1: Train RDM (Stage 1) on 40-50% Dataset

**Why 40-50% is sufficient:**
- Representation space is lower-dimensional (~5K-13K dims vs millions in pixels)
- I-JEPA and SAM are pre-trained; RDM only learns joint distribution structure
- Empirically validated in latent diffusion literature

**Important**: Use the same `--mask_npz_dir` (SAM cache) as StyleGAN2's `--sam_cache_dir` to enable cache sharing.

**Command:**
```bash
# Train RDM with diversity + alignment losses (reuses pre-extracted cache)
python scripts/SEG-RDM/train_unified_seg_rdm.py \
    --config configs/rdm_config.yaml \
    --mask_npz_dir /shared/sam_cache \
    --lambda_diversity 0.1 \
    --lambda_alignment 0.05 \
    --epochs 100
```

**Monitor RDM training:**
- Diversity loss should stabilize (not increase)
- Alignment loss should decrease
- Reconstruction loss (L_seg) should converge

---

### Phase 2: Train StyleGAN2 (Stage 2) with Mixed Training

**Training schedule:**
1. **Warm-up (0-10k kimg):** 100% ground-truth embeddings
   - Stabilizes decoder before introducing RDM samples
   - No distribution mismatch
2. **Gradual mixing (10k-20k kimg):** Ramp RDM mix from 0% → 30%
   - Teaches decoder to handle RDM-sampled embeddings
   - Gradual transition prevents sudden degradation
3. **Steady state (20k+ kimg):** 30% RDM-sampled, 70% ground-truth
   - Balance between ground-truth supervision and RDM generalization
   - Can increase to 50% if monitoring shows stable metrics

**Command:**
```bash
python scripts/StyleGAN2/seg-aware-stylegan2/train.py \
    --outdir=training-runs/seg-stylegan2-mixed \
    --data=/path/to/dataset \
    --gpus=4 \
    --batch=32 \
    --lambda_ijepa=1.0 \
    --ijepa_ckpt=/path/to/ijepa.pth \
    --sam_enabled=True \
    --sam_checkpoint=/path/to/sam_vit_b.pth \
    --sam_cache_dir=/shared/sam_cache \
    --lambda_seg_align=0.1 \
    --lambda_seg_diversity=0.05 \
    --rdm_checkpoint=/path/to/rdm_checkpoint.pt \
    --rdm_mix_prob=0.3 \
    --rdm_warmup_kimg=10000
```

**Monitor StyleGAN2 training:**
- FID should not degrade significantly when RDM mixing starts
- Diversity metrics (real vs generated) should remain close
- Alignment loss should stay low
- If FID increases > 10% after mixing, reduce `rdm_mix_prob`

---

## Hyperparameter Tuning

### RDM (Stage 1)

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `lambda_diversity` | 0.1 | 0.05-0.2 | Too high → over-regularized, too low → collapse |
| `lambda_alignment` | 0.05 | 0.01-0.1 | Balance with reconstruction loss |

**Tuning tips:**
- If diversity loss explodes (>10), reduce `lambda_diversity`
- If segments collapse (diversity metrics show high similarity), increase `lambda_diversity`
- If global-segment alignment is poor, increase `lambda_alignment`

### StyleGAN2 (Stage 2)

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `lambda_seg_align` | 0.1 | 0.05-0.2 | Semantic coherence weight |
| `lambda_seg_diversity` | 0.05 | 0.01-0.1 | Monitoring/regularization |
| `rdm_mix_prob` | 0.0 | 0.1-0.5 | Start conservative (0.1-0.2) |
| `rdm_warmup_kimg` | 10000 | 5000-20000 | Longer warmup = more stable |

**Tuning tips:**
- If FID degrades with RDM mixing, reduce `rdm_mix_prob` or increase warmup
- Monitor alignment gap: if real-generated gap > 0.2, increase `lambda_seg_align`
- If SAM embeddings collapse, increase `lambda_seg_diversity`

---

## Expected Results

### RDM Training
- **Reconstruction loss:** Should converge to ~0.01-0.05
- **Diversity loss:** Should stabilize (not increase over time)
- **Alignment loss:** Should decrease to ~0.5-0.8
- **Covariance rank:** Should maintain high effective rank (>50 out of 256)

### StyleGAN2 Training
- **FID:** Should improve or stay stable with RDM mixing
- **Diversity gap:** Real vs generated <0.1 pairwise similarity difference
- **Alignment gap:** <0.15 cosine similarity difference
- **Visual quality:** Generated images should have coherent segmentation

---

## Troubleshooting

### Issue: Diversity loss explodes in RDM
**Solution:** Reduce `lambda_diversity` from 0.1 → 0.05, check for NaN in covariance

### Issue: FID degrades significantly when RDM mixing starts
**Solution:** 
1. Reduce `rdm_mix_prob` from 0.3 → 0.1
2. Increase `rdm_warmup_kimg` from 10k → 20k
3. Check RDM sample quality (visualize sampled embeddings)

### Issue: Alignment loss doesn't decrease
**Solution:**
1. Verify SAM projection MLP is initialized correctly
2. Check that both global and segment embeddings are non-zero
3. Increase `lambda_seg_align`

### Issue: RDM sampler fails to load
**Solution:**
1. Check checkpoint path is correct
2. Verify checkpoint contains `state_dict` or `model` key
3. Ensure SEG-RDM modules are importable (check `sys.path`)

### Issue: Embeddings still collapse despite losses
**Solution:**
1. Increase both `lambda_diversity` and `lambda_alignment`
2. Check effective rank metric (should be >50)
3. Verify padding masks are applied correctly
4. Reduce learning rate if optimization is unstable

---

## File Summary

### Modified Files
1. `scripts/SEG-RDM/rdm/models/diffusion/ddpm.py` (+150 lines)
   - Diversity and alignment loss methods
   - Modified `p_losses` to include new losses

2. `scripts/StyleGAN2/seg-aware-stylegan2/training/loss.py` (+120 lines)
   - SAM projection MLP
   - Diversity and alignment loss methods
   - Integration into Gmain phase

3. `scripts/StyleGAN2/seg-aware-stylegan2/training/training_loop.py` (+50 lines)
   - RDM sampler initialization
   - Mixed training batch preparation

### New Files
1. `scripts/StyleGAN2/seg-aware-stylegan2/training/rdm_sampler.py` (280 lines)
   - RDM loading and sampling utilities

2. `scripts/embedding_distribution_monitor.py` (380 lines)
   - Diversity and alignment monitoring tools

3. `DIVERSITY_ALIGNMENT_IMPLEMENTATION.md` (this file)
   - Comprehensive documentation

---

## Theory Alignment

This implementation addresses the theoretical requirements from `theory.tex`:

1. **Diversity Loss (Lemma 1):** `-log det(C + εI)` prevents segment embedding collapse
2. **Alignment Loss (Contrastive):** Ensures global and segment embeddings remain semantically coherent
3. **No Trivial Minimizer (Lemma 2):** Combined losses prevent collapsed solutions
4. **Stage 1/Stage 2 Separation:** RDM learns `p(G,S)`, StyleGAN2 learns `p(X|G,S)`
5. **Structural Sufficiency (Proposition):** `(G,S)` is sufficient for region-level prediction tasks

---

## Citation

If you use this implementation, please cite:

```bibtex
@misc{seg-aware-stylegan2-2026,
  title={Segmentation-Aware StyleGAN2 with I-JEPA and SAM Conditioning},
  author={Your Name},
  year={2026},
  note={ECCV 2026 submission}
}
```

---

## Contact

For questions or issues, please open a GitHub issue or contact [your-email].
