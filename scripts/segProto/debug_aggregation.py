#!/usr/bin/env python
"""Debug TokenAggregator group_predictions issue."""
import os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))

TORCH_HOME = '/scratch/gilbreth/abelde/Thesis/StructureAwareGen/.cache/torch'
os.environ['TORCH_HOME'] = TORCH_HOME

from ren_model import DINOv2Extractor, RegionEncoder, SLICPrompter, group_predictions
import yaml
from PIL import Image
from torchvision import transforms as tvt
import glob

device = torch.device('cuda')

# Load models
extractor = DINOv2Extractor(device, torch_home=TORCH_HOME)
config_path = '/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/configs/ren_dinov2_vitl14.yaml'
with open(config_path) as f:
    cfg = yaml.safe_load(f)
region_encoder = RegionEncoder(cfg['ren']).to(device).eval()
ckpt = torch.load(cfg['parameters']['ren_ckpt'], map_location=device, weights_only=False)
region_encoder.load_state_dict(ckpt['region_encoder_state'], strict=True)

# Load real image
img_path = glob.glob('/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/**/*.JPEG', recursive=True)[0]
print(f"Image: {img_path}")
img = Image.open(img_path).convert('RGB')
transform = tvt.Compose([tvt.Resize((518, 518)), tvt.ToTensor()])
img_tensor = transform(img).unsqueeze(0).to(device)

# Forward
feature_maps, cls_tokens = extractor.extract(img_tensor)
prompter = SLICPrompter(518)
prompts = prompter(img_tensor, 256, use_slic=True)
print(f"prompts[0] shape: {prompts[0].shape}")

with torch.inference_mode():
    ren_out = region_encoder(feature_maps, prompts)

pred_tokens = ren_out['pred_tokens']  # [1, 256, 1024]
print(f"pred_tokens shape: {pred_tokens.shape}")

# Debug group_predictions
preds = pred_tokens[0]  # [256, 1024]
features = F.normalize(preds.view(256, -1), p=2, dim=1)
sim_matrix = torch.mm(features, features.t())

# Remove diagonal
diag_mask = torch.eye(256, device=sim_matrix.device).bool()
sim_no_diag = sim_matrix[~diag_mask].view(256, 255)

print(f"\nSimilarity statistics (off-diagonal):")
print(f"  min:    {sim_no_diag.min():.4f}")
print(f"  max:    {sim_no_diag.max():.4f}")
print(f"  mean:   {sim_no_diag.mean():.4f}")
print(f"  median: {sim_no_diag.median():.4f}")
print(f"  std:    {sim_no_diag.std():.4f}")

for thresh in [0.99, 0.975, 0.95, 0.9, 0.85, 0.8, 0.7]:
    n_pairs = (sim_no_diag >= thresh).sum().item() // 2  # undirected
    groups = group_predictions(preds, thresh)
    print(f"  thresh={thresh:.3f}: {n_pairs:6d} pairs, {len(groups):4d} groups (min_size>=3)")

# Also try with merge_small_groups=True
groups_merged = group_predictions(preds, 0.975, merge_small_groups=True)
print(f"\nWith merge_small_groups=True, thresh=0.975: {len(groups_merged)} groups")

# Try relaxed threshold
groups_relaxed = group_predictions(preds, 0.9)
print(f"With thresh=0.9: {len(groups_relaxed)} groups")
for i, g in enumerate(groups_relaxed[:5]):
    print(f"  group {i}: size {len(g)}")
