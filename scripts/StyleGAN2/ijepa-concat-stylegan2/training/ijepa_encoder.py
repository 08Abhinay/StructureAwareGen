# ijepa_utils.py
# It locates and loads the I-JEPA VisionTransformer trunk from your checkpoint,
# dynamically adjusts Python import paths so src/models/vision_transformer.py is found,
# and adapts the patch embedding for differing channel counts.

import importlib, os, re, sys
from pathlib import Path
import torch


def build_ijepa_encoder(ckpt_path,
                        device="cpu",
                        in_channels_override=None, img_size=256):
    """
    Loads and returns a frozen VisionTransformer trunk from an I-JEPA checkpoint.

    Args:
        ckpt_path: path to the .pth.tar checkpoint containing 'encoder' weights.
        device: torch device for the returned model.
        in_channels_override: if set, force the model to expect this many input channels.

    Returns:
        enc  -- VisionTransformer instance with loaded weights, in eval mode.
        meta -- dict containing {'embed_dim', 'patch_size', 'in_chans'} inferred from checkpoint.
    """
    # 0) Ensure the IJEPA repo root is on Python path so we can import src.models...
    ckpt_path = os.path.expanduser(ckpt_path)
    repo_root = Path(ckpt_path).parents[0]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # 1) Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "encoder" not in ckpt:
        raise ValueError("Checkpoint must contain key 'encoder'.")
    enc_state = ckpt["encoder"]
    # strip DataParallel prefix if present
    enc_state = {re.sub(r"^module\.", "", k): v for k, v in enc_state.items()}

    # 2) Infer trunk hyperparameters from patch_embed weight
    w = enc_state["patch_embed.proj.weight"]  # shape: (embed_dim, in_chans_ckpt, p, p)
    embed_dim, in_chans_ckpt, patch_size, _ = w.shape
    in_chans = in_channels_override or in_chans_ckpt

    # 3) Import the VisionTransformer factory from the src folder
    vit_mod = importlib.import_module("src.models.vision_transformer")
    # map embed_dim -> constructor name via VIT_EMBED_DIMS
    ctor_name = {v: k for k, v in vit_mod.VIT_EMBED_DIMS.items()}[embed_dim]
    vit_ctor = getattr(vit_mod, ctor_name)  # e.g. vit_huge
    enc = vit_ctor(patch_size=patch_size, in_chans=in_chans, img_size=[img_size])
    enc.eval().requires_grad_(False)

    # 4) Adapt patch embedding weights if channel count differs
    if in_chans != in_chans_ckpt:
        # original w shape: (D, C_ckpt, p, p)
        if in_chans == 1:
            w_new = w.mean(dim=1, keepdim=True)
        else:
            rep = (in_chans + in_chans_ckpt - 1) // in_chans_ckpt
            w_new = w.repeat(1, rep, 1, 1)[:, :in_chans]
        enc_state["patch_embed.proj.weight"] = w_new

    # 5) Load weights
    missing, unexpected = enc.load_state_dict(enc_state, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading encoder: {missing}")

    # unexpected keys (e.g. predictor) can be safely ignored
    enc.to(device)
    meta = {"embed_dim": embed_dim,
            "patch_size": patch_size,
            "in_chans": in_chans}

    return enc, meta
