# Structure-Aware Image Generation with Self-Supervised Representations

**A two-stage framework for controllable image synthesis using I-JEPA and SAM embeddings without semantic supervision.**

---

## 🎯 Core Idea

**Problem:** Existing structure-aware generation methods (e.g., SPADE, ControlNet) require semantic segmentation maps with labeled classes (sky, road, tree). This limits generalization and requires expensive annotation.

**Our Solution:** Use **self-supervised representations** for structure-aware generation:
- **I-JEPA** provides global image semantics (256-dim vector)
- **SAM** provides instance-level structure (per-mask embeddings, 256-dim each)
- Learn to generate images by first generating these embeddings, then synthesizing pixels

**Key Insight:** Self-supervised embeddings capture visual structure without semantic labels, enabling more flexible control and better domain transfer.

---

## 🏗️ Architecture Overview

### Two-Stage Pipeline

```
Stage 1: SEG-RDM (Segmentation-aware Rectified Diffusion Model)
├─ Input: Gaussian noise
├─ Output: Aligned embeddings (I-JEPA global + SAM per-mask)
└─ Goal: Learn the joint distribution of structure and content

Stage 2: Segmentation-Aware StyleGAN2
├─ Input: Generated embeddings from SEG-RDM
├─ Output: Photorealistic images
└─ Goal: Translate embeddings to pixel space
```

### Why Two Stages?

1. **Disentanglement**: Separate structure generation (Stage 1) from appearance synthesis (Stage 2)
2. **Controllability**: Manipulate embeddings in Stage 1 without retraining StyleGAN2
3. **Stability**: Diffusion models are better at embedding generation; GANs are better at high-res synthesis

---

## 📊 Data Preparation

### 1. I-JEPA Embedding Extraction

**Script:** `scripts/precompute_ijepa_embeddings.py`

**Process:**
```python
Image → I-JEPA ViT-H/14 encoder → Global vector [256-dim]
```

**Storage:**
```
dataset/ijepa_embeddings/
├── 0/
│   ├── 349.npz  # Contains: {'emb': array(256,)}
│   ├── 466.npz
│   └── ...
├── 1/
└── ...
```

**What it captures:** Global semantic content (object identity, scene type, overall composition)

---

### 2. SAM Embedding Extraction

**Script:** `scripts/segProto/precompute_sam_embeddings.py`

**Process:**
```python
Image → SAM ViT-B encoder → Automatic mask generation
     → Per-mask feature extraction [N × 256-dim]
```

**Storage:**
```
dataset/sam_embeddings/
├── masks_npz/
│   └── 0/
│       └── 349.npz  # Contains:
│                    #   - 'emb': (N, 256) per-mask embeddings
│                    #   - 'scores': (N,) mask quality scores
│                    #   - 'packed': (N, H×W/8) binary masks
│                    #   - 'label_map': (H, W) non-overlapping mask IDs
└── meta/
    └── 0/
        └── 349.json  # Mask statistics (bbox, area, etc.)
```

**What it captures:** Instance-level structure (where objects are, their shapes, spatial relationships)

**Key hyperparameters:**
- `max_keep=250`: Keep top 250 masks per image
- `dedup_iou_thresh=0.90`: Remove overlapping masks
- `min_mask_region_area=300`: Filter tiny noise masks

---

### 3. Data Alignment Verification

**Script:** `scripts/verify_data_alignment.py`

**Purpose:** Ensure triplets `(image, I-JEPA embedding, SAM embeddings)` are correctly aligned

**Checks:**
- All three files exist for each image
- Shape consistency (I-JEPA: 256, SAM: N×256)
- File integrity (not corrupted)

---

## 🔥 Stage 1: SEG-RDM (Rectified Diffusion Model)

### Overview

**Goal:** Learn to generate aligned pairs of `(global_vec, seg_embeddings)` that correspond to real images.

**Training paradigm:** Teacher forcing on pre-computed embeddings

### Architecture

**File:** `scripts/SEG-RDM/train_unified_seg_rdm.py`

**Model components:**
```python
SEG_RDM
├── Diffusion backbone (U-Net with attention)
├── Time embedding (sinusoidal)
├── Segmentation conditioning module
│   ├── Processes per-mask embeddings
│   └── Spatial attention over mask regions
└── Output heads
    ├── Global vector prediction [256]
    └── Per-mask embedding predictions [N × 256]
```

### Training Process

**Dataset:** `scripts/SEG-RDM/seg_dataset.py` (`SegmentationAwareDataset`)

```python
# Data loading
triplet = dataset[i]
├── image: (3, H, W)
├── global_vec: (256,)        # I-JEPA embedding
└── seg_embeddings: (N, 256)  # SAM embeddings
```

**Forward pass:**
```python
1. Sample timestep t ~ Uniform(0, T)
2. Add noise to target embeddings:
   - noisy_global = sqrt(alpha_t) * global_vec + sqrt(1-alpha_t) * noise
   - noisy_seg = sqrt(alpha_t) * seg_embeddings + sqrt(1-alpha_t) * noise

3. Model predicts denoised embeddings:
   pred_global, pred_seg = model(noisy_global, noisy_seg, t)

4. Compute loss (see below)
5. Backpropagate and update
```

### Loss Functions

**File:** `scripts/SEG-RDM/train_unified_seg_rdm.py` (lines ~250-350)

```python
# L1: Global vector reconstruction
loss_global = MSE(pred_global, target_global)

# L2: Per-mask embedding reconstruction
loss_seg = MSE(pred_seg, target_seg)  # Mean over all masks

# L3: Alignment loss (ensures global and seg are coherent)
loss_align = ContrastiveLoss(pred_global, pred_seg)
# Positive pairs: (global, its_own_masks)
# Negative pairs: (global, other_image_masks)

# L4: Mask diversity loss (prevents mode collapse)
loss_diversity = -log(det(Cov(pred_seg)))  # Encourage diverse masks

# Total loss
loss = λ1*loss_global + λ2*loss_seg + λ3*loss_align + λ4*loss_diversity
```

**Hyperparameters:**
```python
λ1 = 1.0   # Global weight
λ2 = 1.0   # Segmentation weight  
λ3 = 0.1   # Alignment weight
λ4 = 0.01  # Diversity weight
```

### Inference (Sampling)

**File:** `scripts/SEG-RDM/sample_from_rdm.py`

```python
# Start from pure noise
z_global = randn(256)
z_seg = randn(N, 256)

# Reverse diffusion process
for t in reversed(range(T)):
    # Denoise one step
    z_global, z_seg = model.denoise_step(z_global, z_seg, t)
    
# Final outputs are clean embeddings
generated_global, generated_seg = z_global, z_seg
```

---

## 🎨 Stage 2: Segmentation-Aware StyleGAN2

### Overview

**Goal:** Synthesize photorealistic images conditioned on I-JEPA + SAM embeddings

**Key innovation:** Inject structure at multiple scales using segmentation embeddings

### Architecture

**File:** `scripts/StyleGAN2/seg-aware-stylegan2/training/networks.py`

**Generator modifications:**
```python
StyleGAN2 Generator
├── Mapping Network
│   ├── Input: I-JEPA global vector [256]
│   └── Output: Style codes w [18 × 512]
│
├── Synthesis Network (18 layers)
│   ├── Layer 0 (4×4):  Constant input
│   ├── Layers 1-17:    Progressive upsampling to 256×256
│   │   └── Each layer:
│   │       ├── Modulated Conv (style from w)
│   │       ├── **Segmentation Modulation** ← NEW!
│   │       │   ├── Input: seg_embeddings (N, 256)
│   │       │   ├── Spatial projection to (H, W, 256)
│   │       │   └── Spatially-adaptive normalization
│   │       └── Noise injection
│   └── toRGB: Final image output
```

### Segmentation-Aware Modulation

**File:** `scripts/StyleGAN2/seg-aware-stylegan2/training/networks.py` (lines ~450-550)

**Concept:** Similar to SPADE, but using learned embeddings instead of semantic maps

```python
class SegmentationModulation(nn.Module):
    def forward(self, x, seg_embeddings, mask_map):
        """
        x: Feature map (B, C, H, W)
        seg_embeddings: Per-mask embeddings (B, N, 256)
        mask_map: Spatial mask assignments (B, H, W)
        """
        B, C, H, W = x.shape
        
        # Project seg embeddings to match feature channels
        gamma = self.fc_gamma(seg_embeddings)  # (B, N, C)
        beta = self.fc_beta(seg_embeddings)    # (B, N, C)
        
        # Spatially assign modulation parameters per mask
        gamma_map = torch.zeros(B, C, H, W)
        beta_map = torch.zeros(B, C, H, W)
        
        for mask_id in range(N):
            mask = (mask_map == mask_id)  # (B, H, W)
            gamma_map[mask] = gamma[:, mask_id, :]
            beta_map[mask] = beta[:, mask_id, :]
        
        # Apply spatially-adaptive normalization
        x_normalized = (x - x.mean(dim=[2,3])) / x.std(dim=[2,3])
        x_modulated = gamma_map * x_normalized + beta_map
        
        return x_modulated
```

**Why this works:**
- Each mask region gets its own normalization parameters
- Preserves spatial structure from SAM
- Allows per-region appearance control

### Training

**File:** `scripts/StyleGAN2/seg-aware-stylegan2/train.py`

**Dataset:** Custom dataset loading triplets `(image, global_vec, seg_embs)`

**Training loop:**
```python
# 1. Sample real data
real_imgs, global_vecs, seg_embs, mask_maps = dataset.sample()

# 2. Generate fake images
fake_imgs = generator(global_vecs, seg_embs, mask_maps)

# 3. Discriminator update
D_loss = D_loss_real(real_imgs) + D_loss_fake(fake_imgs)
D_loss += R1_regularization(real_imgs)  # Gradient penalty

# 4. Generator update
G_loss = G_loss_adversarial(fake_imgs, D)
G_loss += λ_percept * Perceptual_loss(fake_imgs, real_imgs)
G_loss += λ_l1 * L1_loss(fake_imgs, real_imgs)
```

**Loss functions:**
```python
# Adversarial loss (non-saturating)
L_adv = -log(D(G(z)))

# Perceptual loss (VGG features)
L_percept = ||VGG(fake) - VGG(real)||_2

# Pixel-wise L1 (only for teacher forcing)
L_l1 = |fake - real|

Total: L = L_adv + λ_percept*L_percept + λ_l1*L_l1
```

---

## 🔄 Complete Training Pipeline

### Phase 1: Data Preparation (Offline)

```bash
# 1. Extract I-JEPA embeddings (~15 hours on 1 GPU)
python scripts/precompute_ijepa_embeddings.py \
    --image_dir dataset/imagenet-1K-hf/train \
    --output_dir dataset/ijepa_embeddings \
    --ijepa_checkpoint checkpoints/IN1K-vit.h.14-300e.pth.tar

# 2. Extract SAM embeddings (100 parallel jobs, ~36 hours total)
./scripts/segProto/submit_sam_extraction.sh

# 3. Verify alignment
python scripts/verify_data_alignment.py \
    --image_dir dataset/imagenet-1K-hf/train \
    --sam_npz_dir dataset/sam_embeddings/masks_npz \
    --ijepa_npz_dir dataset/ijepa_embeddings
```

### Phase 2: Train SEG-RDM

```bash
python scripts/SEG-RDM/train_unified_seg_rdm.py \
    --config configs/seg_rdm_config.yaml \
    --image_dir dataset/imagenet-1K-hf/train \
    --sam_dir dataset/sam_embeddings/masks_npz \
    --ijepa_dir dataset/ijepa_embeddings \
    --epochs 100 \
    --batch_size 64
```

**Training time:** ~7 days on 4×A100 GPUs

**Output:** Trained diffusion model checkpoint (`seg_rdm_checkpoint.pth`)

### Phase 3: Train Segmentation-Aware StyleGAN2

```bash
python scripts/StyleGAN2/seg-aware-stylegan2/train.py \
    --outdir training-runs \
    --data dataset/imagenet-1K-hf/train \
    --sam-dir dataset/sam_embeddings/masks_npz \
    --ijepa-dir dataset/ijepa_embeddings \
    --gpus 8 \
    --batch 32 \
    --kimg 25000
```

**Training time:** ~10 days on 8×A100 GPUs

**Output:** Trained generator checkpoint (`network-snapshot-025000.pkl`)

---

## 🎮 Inference & Control

### Generate New Images

```python
from seg_rdm import SegRDM
from stylegan2 import Generator

# Load models
rdm = SegRDM.load('checkpoints/seg_rdm.pth')
generator = Generator.load('checkpoints/stylegan2.pkl')

# 1. Sample embeddings from diffusion model
global_vec, seg_embs, mask_map = rdm.sample()

# 2. Generate image
image = generator(global_vec, seg_embs, mask_map)
```

### Structure Transfer

```python
# Use structure from image A, content from image B
global_A, seg_A, mask_A = extract_embeddings(image_A)
global_B, seg_B, mask_B = extract_embeddings(image_B)

# Mix: content from B, structure from A
mixed_image = generator(global_B, seg_A, mask_A)
```

### Interactive Editing

```python
# 1. Extract embeddings from real image
global_vec, seg_embs, mask_map = extract_embeddings(image)

# 2. Edit specific mask region
seg_embs[mask_id] += delta  # Modify object appearance

# 3. Regenerate
edited_image = generator(global_vec, seg_embs, mask_map)
```

---

## 📁 Project Structure

```
StructureAwareGen/
├── scripts/
│   ├── precompute_ijepa_embeddings.py      # I-JEPA extraction
│   ├── verify_data_alignment.py            # Data validation
│   │
│   ├── segProto/
│   │   ├── precompute_sam_embeddings.py    # SAM extraction
│   │   └── submit_sam_extraction.sh        # Parallel job submission
│   │
│   ├── SEG-RDM/
│   │   ├── train_unified_seg_rdm.py        # Main training script
│   │   ├── seg_dataset.py                  # Dataset loader
│   │   ├── sample_from_rdm.py              # Inference script
│   │   └── configs/
│   │       └── seg_rdm_config.yaml         # Hyperparameters
│   │
│   └── StyleGAN2/
│       └── seg-aware-stylegan2/
│           ├── train.py                    # Main training script
│           ├── training/
│           │   └── networks.py             # Generator/Discriminator
│           └── torch_utils/
│
├── dataset/
│   ├── imagenet-1K-hf/train/               # Raw images
│   ├── ijepa_embeddings/                   # I-JEPA outputs
│   └── sam_embeddings/                     # SAM outputs
│       ├── masks_npz/
│       └── meta/
│
└── checkpoints/
    ├── IN1K-vit.h.14-300e.pth.tar         # I-JEPA pretrained
    ├── sam_vit_b_01ec64.pth               # SAM pretrained
    ├── seg_rdm_checkpoint.pth             # Trained SEG-RDM
    └── stylegan2_seg_aware.pkl            # Trained StyleGAN2
```

---

## 🔬 Key Research Questions

### 1. Why self-supervised over semantic segmentation?

**Advantages:**
- ✅ No annotation required (works on any image dataset)
- ✅ Better domain generalization (no class bias)
- ✅ Finer-grained control (instance-level, not class-level)
- ✅ Captures visual similarity, not semantic categories

**Trade-offs:**
- ⚠️ Less interpretable than "sky" or "road"
- ⚠️ Requires two-stage training (more complex)

### 2. Why two-stage (RDM → StyleGAN2) vs end-to-end diffusion?

**Advantages:**
- ✅ Disentangles structure from appearance
- ✅ Can manipulate embeddings without retraining
- ✅ StyleGAN2 provides better high-res synthesis
- ✅ Faster sampling (one diffusion sample → many StyleGAN variations)

**Trade-offs:**
- ⚠️ Two models to train and maintain
- ⚠️ Potential error propagation from Stage 1 to Stage 2

### 3. How does segmentation conditioning work?

**Mechanism:** Spatially-adaptive normalization (similar to SPADE)
- Each mask region gets unique normalization parameters
- Parameters derived from learned SAM embeddings
- Applied at multiple scales in StyleGAN2 synthesis network

**Why it works:**
- Preserves spatial structure from masks
- Allows per-region appearance control
- Integrates naturally with StyleGAN2's style modulation

---

## 📈 Expected Outcomes

### Quantitative Metrics

1. **Image Quality:**
   - FID (Fréchet Inception Distance)
   - IS (Inception Score)

2. **Structure Preservation:**
   - Mask IoU with target structures
   - Perceptual similarity (LPIPS)

3. **Controllability:**
   - User study: structural control accuracy
   - Ablation: SEG-RDM vs vanilla diffusion

### Qualitative Results

1. **Structure transfer:** Same structure, different content
2. **Instance editing:** Modify individual objects
3. **Domain transfer:** Transfer structure across datasets
4. **Interpolation:** Smooth transitions in embedding space

---

## 🎓 Relation to Prior Work

### SPADE (Semantic Image Synthesis)
- **Similarity:** Both use spatially-adaptive normalization
- **Difference:** We use self-supervised embeddings, not semantic labels

### ControlNet
- **Similarity:** Both condition generation on structural inputs
- **Difference:** We operate in embedding space (two-stage), not pixel space

### I-JEPA
- **Similarity:** We use I-JEPA for global semantics
- **Difference:** We combine with SAM for structure and train generative models

### SAM (Segment Anything Model)
- **Similarity:** We use SAM for instance segmentation
- **Difference:** We extract embeddings, not just masks, and use for generation

---

## 💡 Novel Contributions

1. **Self-supervised structure-aware generation:** First work to use I-JEPA + SAM embeddings for controllable image synthesis

2. **Two-stage embedding-space pipeline:** Separates structure generation (RDM) from appearance synthesis (StyleGAN2)

3. **Segmentation-aware StyleGAN2:** Novel conditioning mechanism using per-mask embeddings

4. **Aligned embedding dataset:** Large-scale dataset of (image, I-JEPA, SAM) triplets for future research

---

## 🚀 Future Directions

1. **Cross-domain transfer:** Train on ImageNet, generate on COCO/ADE20K
2. **Hierarchical structure:** Multi-level segmentation (object → parts)
3. **Text-guided control:** Combine with CLIP for language-based editing
4. **Real-time editing:** Optimize for interactive applications
5. **Video generation:** Extend to temporal consistency

---

## 📚 References

- **I-JEPA:** Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", CVPR 2023
- **SAM:** Kirillov et al., "Segment Anything", ICCV 2023
- **SPADE:** Park et al., "Semantic Image Synthesis with Spatially-Adaptive Normalization", CVPR 2019
- **StyleGAN2:** Karras et al., "Analyzing and Improving the Image Quality of StyleGAN", CVPR 2020
- **Rectified Flow:** Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023

---

## 👤 Author

**Abhinay Belde**
- Affiliation: [Your University]
- Email: [Your Email]
- Advisor: [Professor Name]

---

*Last updated: February 2026*