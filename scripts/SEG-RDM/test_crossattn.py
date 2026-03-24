#!/usr/bin/env python
"""Quick smoke test for CrossAttentionSegTransformer."""
import sys
sys.path.insert(0, "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM")

from rdm.modules.diffusionmodules.cross_attention_transformer import CrossAttentionSegTransformer
import torch

model = CrossAttentionSegTransformer(
    token_dim=256, d_model=768, n_heads=12, n_layers=8,
    d_ff=3072, dropout=0.1, max_seq_len=256, time_emb_dim=256,
)
total = sum(p.numel() for p in model.parameters())
print(f"Params: {total / 1e6:.2f}M")

B, C, N = 4, 256, 101
x = torch.randn(B, C, 1, N)
t = torch.randint(0, 1000, (B,))
pad = torch.zeros(B, N, dtype=torch.bool)
pad[:, 80:] = True

out = model(x, t, padding_mask=pad)
print(f"DDPM: {x.shape} -> {out.shape} match={out.shape == x.shape}")

x2 = torch.randn(B, N, C)
out2 = model(x2, t, padding_mask=pad)
print(f"3D: {x2.shape} -> {out2.shape} match={out2.shape == x2.shape}")

assert not torch.isnan(out).any().item(), "NaN in DDPM output"
assert not torch.isnan(out2).any().item(), "NaN in 3D output"
print("NaN check: OK")

out.sum().backward()
print("Backward: OK")

# Also verify the original UnifiedSegTransformer still works
from rdm.modules.diffusionmodules.unified_transformer import UnifiedSegTransformer
model_orig = UnifiedSegTransformer(
    token_dim=256, d_model=768, n_heads=12, n_layers=8,
    d_ff=3072, dropout=0.1, max_seq_len=256, time_emb_dim=256,
)
out_orig = model_orig(x.detach(), t, padding_mask=pad)
print(f"Original UnifiedSegTransformer: {x.shape} -> {out_orig.shape} match={out_orig.shape == x.shape}")

print("\nALL CHECKS PASSED")
