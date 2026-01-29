import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.ijepa import vision_transformer as ijepa_vits


_EMBED_TO_HEADS = {
    192: 3,
    384: 6,
    768: 12,
    1024: 16,
    1280: 16,
    1408: 16,
}


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("target_encoder", "encoder", "model", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _strip_prefixes(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ("module.", "encoder.", "backbone."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        cleaned[new_k] = v
    return cleaned


def _infer_arch(state_dict):
    patch_size = None
    embed_dim = None
    in_chans = None
    proj_w = state_dict.get("patch_embed.proj.weight")
    if proj_w is not None:
        embed_dim = proj_w.shape[0]
        in_chans = proj_w.shape[1]
        patch_size = proj_w.shape[2]

    block_ids = []
    for k in state_dict.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_ids.append(int(parts[1]))
    depth = max(block_ids) + 1 if block_ids else None

    if embed_dim is None or depth is None:
        raise ValueError("Could not infer I-JEPA architecture from checkpoint.")

    num_heads = _EMBED_TO_HEADS.get(embed_dim)
    if num_heads is None:
        raise ValueError(f"Unsupported embed_dim for I-JEPA: {embed_dim}")

    mlp_ratio = 48 / 11 if embed_dim == 1408 else 4.0

    return {
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "patch_size": patch_size or 16,
        "in_chans": in_chans or 3,
    }


def _adapt_patch_embed_weight(weight, target_in_chans):
    if weight.shape[1] == target_in_chans:
        return weight
    if target_in_chans == 1:
        return weight.mean(dim=1, keepdim=True)
    if weight.shape[1] == 1:
        return weight.repeat(1, target_in_chans, 1, 1) / float(target_in_chans)
    repeat = target_in_chans // weight.shape[1]
    if target_in_chans % weight.shape[1] == 0:
        return weight.repeat(1, repeat, 1, 1) / float(repeat)
    return weight[:, :target_in_chans, :, :]


def build_ijepa_encoder(
    ckpt_path,
    device="cpu",
    in_channels_override=None,
    img_size=256,
    out_dim=None,
):
    """
    Return a **frozen I-JEPA ViT encoder** that maps (B,C,H,W) → (B,D).

    - Auto-detects ViT variant from the checkpoint.
    - Optionally adapts the patch embedding for a different input channel count.
    - Returns pooled patch-token embeddings.
    """
    ckpt_path = os.path.expanduser(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _strip_prefixes(_extract_state_dict(checkpoint))

    arch = _infer_arch(state_dict)
    in_chans = in_channels_override or arch["in_chans"]

    model = ijepa_vits.VisionTransformer(
        img_size=[img_size],
        patch_size=arch["patch_size"],
        in_chans=in_chans,
        embed_dim=arch["embed_dim"],
        depth=arch["depth"],
        num_heads=arch["num_heads"],
        mlp_ratio=arch["mlp_ratio"],
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
    )

    if "patch_embed.proj.weight" in state_dict:
        state_dict["patch_embed.proj.weight"] = _adapt_patch_embed_weight(
            state_dict["patch_embed.proj.weight"], in_chans
        )

    model_state = model.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered_state, strict=False)

    class IJEPAEncoder(nn.Module):
        def __init__(self, backbone, img_size, out_dim):
            super().__init__()
            self.backbone = backbone
            self.img_size = img_size
            self.out_dim = out_dim
            self.proj = None
            if out_dim is not None and out_dim != backbone.embed_dim:
                self.proj = nn.Linear(backbone.embed_dim, out_dim, bias=False)

        @torch.no_grad()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.min() < 0:
                x = (x + 1.0) / 2.0
            if x.shape[-1] != self.img_size:
                x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            if x.shape[1] != 3:
                reps = (x.shape[1] + 2) // 3
                mean = mean.repeat(1, reps, 1, 1)[:, :x.shape[1]]
                std = std.repeat(1, reps, 1, 1)[:, :x.shape[1]]
            x = (x - mean) / std
            feats = self.backbone.forward_features(x)
            if feats.dim() == 3:
                feats = feats.mean(dim=1)
            if self.proj is not None:
                feats = self.proj(feats)
            return feats

    enc = IJEPAEncoder(model, img_size, out_dim).eval().to(device)
    enc.requires_grad_(False)

    meta = {
        "embed_dim": arch["embed_dim"],
        "in_chans": in_chans,
        "img_size": img_size,
        "patch_size": arch["patch_size"],
        "depth": arch["depth"],
        "num_heads": arch["num_heads"],
        "mlp_ratio": arch["mlp_ratio"],
        "out_dim": out_dim or arch["embed_dim"],
    }
    return enc, meta
