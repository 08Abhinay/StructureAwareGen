# # ijepa_utils.py
# # It locates and loads the I-JEPA VisionTransformer trunk from your checkpoint,
# # dynamically adjusts Python import paths so src/models/vision_transformer.py is found,
# # and adapts the patch embedding for differing channel counts.

# import importlib, os, re, sys
# from pathlib import Path
# import torch


# def build_ijepa_encoder(ckpt_path,
#                         device="cpu",
#                         in_channels_override=None, img_size=256):
#     """
#     Loads and returns a frozen VisionTransformer trunk from an I-JEPA checkpoint.

#     Args:
#         ckpt_path: path to the .pth.tar checkpoint containing 'encoder' weights.
#         device: torch device for the returned model.
#         in_channels_override: if set, force the model to expect this many input channels.

#     Returns:
#         enc  -- VisionTransformer instance with loaded weights, in eval mode.
#         meta -- dict containing {'embed_dim', 'patch_size', 'in_chans'} inferred from checkpoint.
#     """
#     # 0) Ensure the IJEPA repo root is on Python path so we can import src.models...
#     ckpt_path = os.path.expanduser(ckpt_path)
#     repo_root = Path(ckpt_path).parents[0]
#     if str(repo_root) not in sys.path:
#         sys.path.insert(0, str(repo_root))

#     # 1) Load checkpoint
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     if "encoder" not in ckpt:
#         raise ValueError("Checkpoint must contain key 'encoder'.")
#     enc_state = ckpt["encoder"]
#     # strip DataParallel prefix if present
#     enc_state = {re.sub(r"^module\.", "", k): v for k, v in enc_state.items()}

#     # 2) Infer trunk hyperparameters from patch_embed weight
#     w = enc_state["patch_embed.proj.weight"]  # shape: (embed_dim, in_chans_ckpt, p, p)
#     embed_dim, in_chans_ckpt, patch_size, _ = w.shape
#     in_chans = in_channels_override

#     # 3) Import the VisionTransformer factory from the src folder
#     vit_mod = importlib.import_module("src.models.vision_transformer")
#     # map embed_dim -> constructor name via VIT_EMBED_DIMS
#     ctor_name = {v: k for k, v in vit_mod.VIT_EMBED_DIMS.items()}[embed_dim]
#     vit_ctor = getattr(vit_mod, ctor_name)  # e.g. vit_huge
#     enc = vit_ctor(patch_size=patch_size, in_chans=in_chans, img_size=[img_size])
#     enc.eval().requires_grad_(False)

#     # 4) Adapt patch embedding weights if channel count differs
#     if in_chans != in_chans_ckpt:
#         # original w shape: (D, C_ckpt, p, p)
#         if in_chans == 1:
#             w_new = w.mean(dim=1, keepdim=True)
#         else:
#             rep = (in_chans + in_chans_ckpt - 1) // in_chans_ckpt
#             w_new = w.repeat(1, rep, 1, 1)[:, :in_chans]
#         enc_state["patch_embed.proj.weight"] = w_new

#     # 5) Load weights
#     missing, unexpected = enc.load_state_dict(enc_state, strict=False)
#     if missing:
#         raise RuntimeError(f"Missing keys when loading encoder: {missing}")

#     # unexpected keys (e.g. predictor) can be safely ignored
#     enc.to(device)
#     meta = {"embed_dim": embed_dim,
#             "patch_size": patch_size,
#             "in_chans": in_chans}

#     return enc, meta


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