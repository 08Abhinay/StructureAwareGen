import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def build_ijepa_encoder(
        ckpt_path,
        device = "cpu",
        in_channels_override = None,
        img_size= 256,
):
    """
    Return a **frozen ResNet‑50 backbone** that maps (B,C,H,W) → (B, 2048).

    - Loads the “backbone_momentum only” state‑dict you saved earlier.
    - Optionally adapts the first conv to `in_channels_override`.
    - Adds global‑avg‑pool so the caller always gets a 1‑D embedding.
    """
    ckpt_path = os.path.expanduser(ckpt_path)
    sd        = torch.load(ckpt_path, map_location="cpu")

    # ------------------------------------------------------------------
    # 1) Build trunk.  timm's ResNet accepts `in_chans=...`
    # ------------------------------------------------------------------
    in_chans = in_channels_override or 3
    trunk = timm.create_model(
        "resnet50",
        pretrained=False,
        features_only=True,
        out_indices=[-1],
        in_chans=in_chans,
    )

    # ------------------------------------------------------------------
    # 2) Load weights
    # ------------------------------------------------------------------
    missing, unexpected = trunk.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")

    # ------------------------------------------------------------------
    # 3) Wrap with global‑pool → (B, 2048)
    # ------------------------------------------------------------------
    class ResNet50Global(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        @torch.no_grad()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x : (B, C, H, W) dtype float32 in [-1,1] or [0,1]
            Returns:
                (B, 2048) pooled embedding  (no grad, eval mode)
            """
            if x.shape[-1] != img_size:
                x = F.interpolate(
                    x, size=img_size, mode="bilinear", align_corners=False
                )
            feats = self.backbone(x)[-1]          # (B, 2048, h, w)
            return F.adaptive_avg_pool2d(feats, 1).flatten(1)

    enc = ResNet50Global(trunk).eval().to(device)
    enc.requires_grad_(False)

    meta = {
        "embed_dim": 2048,
        "in_chans":  in_chans,
        "img_size":  img_size,
    }
    return enc, meta