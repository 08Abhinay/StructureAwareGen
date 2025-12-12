# metrics/ms_ssim.py
"""Multi‑Scale SSIM (lower = more diverse) on 10 000 fake images.

Matches the protocol in Karras et al. “A Style‑Based Generator…”, Appx. B.1. 
"""

import numpy as np
import torch
from torchmetrics.functional import (
    multiscale_structural_similarity_index_measure as ms_ssim,
)

from . import metric_utils   # already in your repo

# ---------------------------------------------------------------------------

@torch.no_grad()
def _ms_ssim_10k(opts, num_gen=10_000, batch=64):
    G          = opts.G_ema           # Exposed by MetricOptions
    device     = opts.device
    rng        = np.random.RandomState(123)
    pair_score = []

    while len(pair_score) < num_gen:
        bs = min(batch, num_gen - len(pair_score))  # last iter might be smaller
        z  = torch.from_numpy(rng.randn(bs, G.z_dim)).to(device)
        c  = torch.zeros(bs, G.c_dim, device=device)

        imgs = G(z, c, noise_mode="const", force_fp32=True)  # (B,C,H,W) in ‑1…1
        imgs = (imgs + 1) / 2                                # → 0…1 float32

        # build random pairs inside the same mini‑batch
        perm = torch.randperm(bs, device=device)
        for a, b in perm.view(-1, 2):         # works when bs is even
            s = ms_ssim(
                imgs[a : a + 1], imgs[b : b + 1], data_range=1.0, channel=imgs.shape[1]
            )
            pair_score.append(float(s))

    return float(np.mean(pair_score))

# ---------------------------------------------------------------------------

def compute_ms_ssim(opts):
    return _ms_ssim_10k(opts)

