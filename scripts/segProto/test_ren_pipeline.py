#!/usr/bin/env python
"""
Quick single-image test of the REN DINOv2 extraction pipeline.
Run on a GPU node:
    TORCH_HOME=.cache/torch python scripts/segProto/test_ren_pipeline.py
"""
import os, sys, time
import torch
import numpy as np

# Add segProto to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

TORCH_HOME = '/scratch/gilbreth/abelde/Thesis/StructureAwareGen/.cache/torch'
os.environ['TORCH_HOME'] = TORCH_HOME

print(f"Python {sys.version}")
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU mem: {torch.cuda.mem_get_info(0)[1] / 1e9:.1f} GB")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Step 1: Load DINOv2 ---
print("\n=== Step 1: Loading DINOv2 ===")
t0 = time.time()
from ren_model import DINOv2Extractor, RegionEncoder, TokenAggregator, SLICPrompter, group_predictions

extractor = DINOv2Extractor(device, torch_home=TORCH_HOME)
print(f"DINOv2 loaded in {time.time()-t0:.1f}s")
print(f"DINOv2 params: {sum(p.numel() for p in extractor.model.parameters())/1e6:.1f}M")

# --- Step 2: Load REN checkpoint ---
print("\n=== Step 2: Loading REN RegionEncoder ===")
import yaml
config_path = '/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/configs/ren_dinov2_vitl14.yaml'
with open(config_path) as f:
    cfg = yaml.safe_load(f)

ren_cfg = cfg['ren']
params_cfg = cfg['parameters']

region_encoder = RegionEncoder(ren_cfg).to(device).eval()

ckpt_path = params_cfg['ren_ckpt']
print(f"Loading checkpoint: {ckpt_path}")
assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
region_encoder.load_state_dict(ckpt['region_encoder_state'], strict=True)
print(f"REN loaded! Params: {sum(p.numel() for p in region_encoder.parameters())/1e6:.1f}M")

aggregator = TokenAggregator(merge_similarity=params_cfg.get('merge_similarity', 0.975))

# --- Step 3: Create synthetic image and run pipeline ---
print("\n=== Step 3: Running pipeline on image ===")
H, W = 518, 518

# Try to find a real image first, fallback to synthetic
import glob
real_images = glob.glob('/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/**/*.JPEG', recursive=True)
if not real_images:
    real_images = glob.glob('/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/256/**/*.JPEG', recursive=True)

if real_images:
    from PIL import Image
    from torchvision import transforms as tvt
    img_path = real_images[0]
    print(f"Using real image: {img_path}")
    img_pil = Image.open(img_path).convert('RGB')
    transform = tvt.Compose([
        tvt.Resize((H, W)),
        tvt.ToTensor(),
    ])
    fake_img = transform(img_pil).unsqueeze(0)
else:
    print("Using synthetic image (no real images found)")
    # Create a more structured synthetic image (not pure noise)
    fake_img = torch.zeros(1, 3, H, W)
    # Create colored blocks that should form distinct regions
    fake_img[0, 0, :259, :259] = 1.0  # red top-left
    fake_img[0, 1, :259, 259:] = 1.0  # green top-right
    fake_img[0, 2, 259:, :259] = 1.0  # blue bottom-left
    fake_img[0, :, 259:, 259:] = 0.5  # gray bottom-right

# Extract DINOv2 features
t0 = time.time()
feature_maps, cls_tokens = extractor.extract(fake_img)
print(f"DINOv2 extraction: {time.time()-t0:.3f}s")
print(f"  feature_maps: {feature_maps.shape}")  # [1, 1024, 37, 37]
print(f"  cls_tokens:   {cls_tokens.shape}")      # [1, 1024]

# --- Step 4: SLIC prompts ---
print("\n=== Step 4: SLIC prompts ===")
prompter = SLICPrompter(image_resolution=H)
num_segments = params_cfg.get('grid_size', 37) ** 2  # 37*37=1369 by default

# Try SLIC first
try:
    prompts_slic = prompter(fake_img, num_segments=num_segments, use_slic=True)
    print(f"  SLIC prompts: {prompts_slic[0].shape}")  # [N, 2]
    use_slic = True
except Exception as e:
    print(f"  SLIC failed: {e}")
    use_slic = False

# Also test grid fallback
prompts_grid = prompter(fake_img, num_segments=num_segments, use_slic=False)
print(f"  Grid prompts: {prompts_grid[0].shape}")

prompts = prompts_slic if use_slic else prompts_grid

# --- Step 5: RegionEncoder forward ---
print("\n=== Step 5: RegionEncoder forward ===")
t0 = time.time()
with torch.inference_mode():
    ren_out = region_encoder(feature_maps, prompts)
pred_tokens = ren_out['pred_tokens']
proj_tokens = ren_out['proj_tokens']
attn_scores = ren_out['attn_scores'][-1]  # last layer attn: [B, N_prompts, N_patches]
print(f"RegionEncoder: {time.time()-t0:.3f}s")
print(f"  pred_tokens: {pred_tokens.shape}")  # [1, N_prompts, 1024]
print(f"  proj_tokens: {proj_tokens.shape}")   # [1, N_prompts, 1024]
print(f"  attn_scores: {attn_scores.shape}")

# --- Step 6: TokenAggregator ---
print("\n=== Step 6: TokenAggregator ===")
t0 = time.time()
agg_result = aggregator(pred_tokens, proj_tokens, attn_scores, prompts)
print(f"Aggregation: {time.time()-t0:.3f}s")
# Results are lists (one per batch item)
agg_pred = agg_result['aggregated_pred_tokens'][0]  # [M, 1024]
agg_proj = agg_result['aggregated_proj_tokens'][0]
agg_attn = agg_result['aggregated_attn_scores'][0]  # [N_layers?, M]
print(f"  aggregated_pred_tokens: {agg_pred.shape}")
print(f"  aggregated_proj_tokens: {agg_proj.shape}")
print(f"  aggregated_attn_scores: {agg_attn.shape}")
print(f"  num regions after merging: {agg_pred.shape[0]}")

# --- Step 7: Verify shapes and dtypes ---
print("\n=== Summary ===")
print(f"CLS token:     shape={cls_tokens.shape}, dtype={cls_tokens.dtype}")
print(f"Region tokens: shape={agg_pred.shape}, dtype={agg_pred.dtype}")
print(f"Attn scores:   shape={agg_attn.shape}, dtype={agg_attn.dtype}")
n_regions = agg_pred.shape[0]
print(f"Regions per image: {n_regions}")

# Z-score normalize CLS
mean = cls_tokens.mean()
std = cls_tokens.std()
cls_norm = (cls_tokens - mean) / (std + 1e-8)
print(f"CLS z-normed:  mean={cls_norm.mean():.4f}, std={cls_norm.std():.4f}")

# Compute attention mass (sum of attn weights per region)
# attn_scores: [1, M, N_patches] after aggregation
attn_mass = agg_attn.sum(dim=-1) if agg_attn.dim() >= 2 else agg_attn
print(f"Attn mass range: [{attn_mass.min():.4f}, {attn_mass.max():.4f}]")
print(f"Attn mass sum: {attn_mass.sum():.4f}")

print(f"\nGPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("\n=== ALL TESTS PASSED ===")
