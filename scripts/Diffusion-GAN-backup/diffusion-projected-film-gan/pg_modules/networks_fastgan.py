# original implementation: https://github.com/odegeasslbc/FastGAN-pytorch/blob/main/models.py
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
import numpy as np
import torch
import torch.nn as nn
from pg_modules.blocks import (InitLayer, UpBlockBig, UpBlockBigCond, UpBlockSmall, 
                               UpBlockSmallCond, SEBlock, conv2d, 
                               UpBlockBigFiLM, FiLM, UpBlockSmallFiLM)
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from torch_utils.ops import conv2d_resample

def normalize_second_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class DummyMapping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, c, **kwargs):
        return z.unsqueeze(1)  # to fit the StyleGAN API

# -------------------------------------------------------------
# Fast-GAN synthesis with FiLM (unconditional) and vanilla CCBN
# -------------------------------------------------------------
class FastganSynthesis(nn.Module):
    def __init__(self, ngf=128, z_dim=256, nc=3, img_resolution=256, lite=False):
        super().__init__()
        self.img_resolution = img_resolution
        # -------- channel table ---------------------------------
        nfc_multi = {2:16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1,
                     256:0.5, 512:0.25, 1024:0.125}
        nfc = {k: int(v * ngf) for k, v in nfc_multi.items()}

        # ------------- stem -------------------------------------
        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)

        # ------------- FiLM up-blocks ---------------------------
        UpBlock = UpBlockSmallFiLM if lite else UpBlockBigFiLM
        self.feat_8   = UpBlock(nfc[4],   nfc[8],   z_dim)
        self.feat_16  = UpBlock(nfc[8],   nfc[16],  z_dim)
        self.feat_32  = UpBlock(nfc[16],  nfc[32],  z_dim)
        self.feat_64  = UpBlock(nfc[32],  nfc[64],  z_dim)
        self.feat_128 = UpBlock(nfc[64],  nfc[128], z_dim)
        self.feat_256 = UpBlock(nfc[128], nfc[256], z_dim)
        if img_resolution > 256:
            self.feat_512  = UpBlock(nfc[256], nfc[512],  z_dim)
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024], z_dim)

        # ------------- SE blocks & head -------------------------
        self.se_64   = SEBlock(nfc[4],  nfc[64])
        self.se_128  = SEBlock(nfc[8],  nfc[128])
        self.se_256  = SEBlock(nfc[16], nfc[256])
        if img_resolution > 256:
            self.se_512 = SEBlock(nfc[32], nfc[512])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

    # -----------------------------------------------------------
    def forward(self, w, c=None, **_):
        # w : (B, 1, z_dim)  from mapping
        z_lat = w[:, 0]                             # (B, z_dim)
        z_lat = normalize_second_moment(z_lat)

        feat_4  = self.init(z_lat)                  # 4×4
        feat_8  = self.feat_8 (feat_4,  z_lat)
        feat_16 = self.feat_16(feat_8,  z_lat)
        feat_32 = self.feat_32(feat_16, z_lat)

        if self.img_resolution == 32:
            return self.to_big(feat_32)

        feat_64 = self.se_64(feat_4,  self.feat_64(feat_32, z_lat))
        if self.img_resolution == 64:
            return self.to_big(feat_64)

        feat_128 = self.se_128(feat_8, self.feat_128(feat_64, z_lat))
        feat_last = feat_128  # ≥128 px by construction

        if self.img_resolution >= 256:
            feat_last = self.se_256(feat_16, self.feat_256(feat_last, z_lat))

        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_32, self.feat_512(feat_last, z_lat))

        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last, z_lat)

        return self.to_big(feat_last)


# ------------------------------------------------------------------
# Conditional synthesis that feeds IJepa embeddings into every CCBN
# ------------------------------------------------------------------
class FastganSynthesisCond(nn.Module):
    def __init__(self, ngf=64, z_dim=256, nc=3, img_resolution=256,
                 num_classes=1000, lite=False):
        super().__init__()
        self.img_resolution = img_resolution

        # -------- channel table ---------------------------------
        nfc_multi = {2:16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1,
                     256:0.5, 512:0.25, 1024:0.125, 2048:0.125}
        nfc = {k: int(v * ngf) for k, v in nfc_multi.items()}

        # ------------- stem -------------------------------------
        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)

        UpBlock = UpBlockSmallCond if lite else UpBlockBigCond
        self.feat_8   = UpBlock(nfc[4],   nfc[8],   z_dim)
        self.feat_16  = UpBlock(nfc[8],   nfc[16],  z_dim)
        self.feat_32  = UpBlock(nfc[16],  nfc[32],  z_dim)
        self.feat_64  = UpBlock(nfc[32],  nfc[64],  z_dim)
        self.feat_128 = UpBlock(nfc[64],  nfc[128], z_dim)
        self.feat_256 = UpBlock(nfc[128], nfc[256], z_dim)
        if img_resolution > 256:
            self.feat_512  = UpBlock(nfc[256], nfc[512],  z_dim)
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024], z_dim)

        # ------------- SE & head --------------------------------
        self.se_64   = SEBlock(nfc[4],  nfc[64])
        self.se_128  = SEBlock(nfc[8],  nfc[128])
        self.se_256  = SEBlock(nfc[16], nfc[256])
        if img_resolution > 256:
            self.se_512 = SEBlock(nfc[32], nfc[512])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)
        self.embed  = nn.Embedding(num_classes, z_dim)

    # -----------------------------------------------------------
    def forward(self, w, c, *, e_ijepa=None, sem_ramp=1.0, **_):
        """
        w        : (B, 1, z_dim)   latent after mapping
        c        : (B, num_classes) one-hot or soft class labels
        e_ijepa  : (B, _SEM_DIM)    IJepa global embedding (may be None)
        """
        # fade IJepa in/out once here
        if e_ijepa is not None:
            e_ijepa = e_ijepa * sem_ramp
        
        c_embed = self.embed(c.argmax(1))           # (B, z_dim)
        z_lat   = normalize_second_moment(w[:, 0])  # (B, z_dim)

        # ---------- main branch ---------------------------------
        feat_4  = self.init(z_lat)
        feat_8  = self.feat_8 (feat_4,  c_embed, e_ijepa=e_ijepa)
        feat_16 = self.feat_16(feat_8,  c_embed, e_ijepa=e_ijepa)
        feat_32 = self.feat_32(feat_16, c_embed, e_ijepa=e_ijepa)

        if self.img_resolution == 32:
            return self.to_big(feat_32)

        feat_64 = self.se_64(
            feat_4,
            self.feat_64(feat_32, c_embed, e_ijepa=e_ijepa)
        )
        if self.img_resolution == 64:
            return self.to_big(feat_64)

        feat_128 = self.se_128(
            feat_8,
            self.feat_128(feat_64, c_embed, e_ijepa=e_ijepa)
        )
        feat_last = feat_128

        if self.img_resolution >= 256:
            feat_last = self.se_256(
                feat_16,
                self.feat_256(feat_last, c_embed, e_ijepa=e_ijepa)
            )

        if self.img_resolution >= 512:
            feat_last = self.se_512(
                feat_32,
                self.feat_512(feat_last, c_embed, e_ijepa=e_ijepa)
            )

        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last, c_embed, e_ijepa=e_ijepa)

        return self.to_big(feat_last)


#----------------------------------------------------------------------------
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

#----------------------------------------------------------------------------


class IJEPAAddFusion(nn.Module):
    """
    Fast-GAN mapping wrapper that *adds* a projected I-JEPA embedding to z
    with (1) learnable strength α, (2) sem_ramp fade-in, and (3) per-sample
    dropout controlled by sem_mixing_prob.

    Parameters
    ----------
    z_dim : int          latent dimensionality
    ijepa_dim : int      dimensionality of encoder embedding (default 2048)
    proj_hidden : int    hidden size in projection MLP
    lr_multiplier : float  LR multiplier for FC layers
    sem_mixing_prob : float  probability of *dropping* the embedding
    """
    def __init__(
        self,
        z_dim,
        ijepa_dim=2048,
        proj_hid1=1024,
        proj_hid2=512,
        proj_hid3=256,
        lr_multiplier=0.01,
        sem_mixing_prob=0.70,
    ):
        super().__init__()
        self.sem_mixing_prob = sem_mixing_prob

        # ── I-JEPA → z projection ─────────────────────────────────────
        self.proj = nn.Sequential(
            FullyConnectedLayer(ijepa_dim, proj_hid1,
                                activation='lrelu', lr_multiplier=lr_multiplier),
            FullyConnectedLayer(proj_hid1, proj_hid2,
                                activation='lrelu', lr_multiplier=lr_multiplier),
            FullyConnectedLayer(proj_hid2, proj_hid3,
                                activation='lrelu', lr_multiplier=lr_multiplier),
            FullyConnectedLayer(proj_hid3, z_dim,
                                activation='linear', lr_multiplier=lr_multiplier),
        )

        # learnable global scaling (initial 0.2 ≈ “20 %” strength)
        self.alpha = nn.Parameter(torch.tensor(0.20))

    # ------------------------------------------------------------------
    def forward(
        self,
        z,                 # (B, z_dim)
        c,                 # ignored, keeps StyleGAN-API signature
        *,
        e_ijepa=None,      # (B, ijepa_dim) or None
        sem_ramp=1.0,      # fade-in from Loss
        **kwargs,
    ):

        if (e_ijepa is None) or (sem_ramp == 0):
            return z.unsqueeze(1)                     # (B,1,z_dim)

        B, _ = z.shape
        # 1) project & normalize
        e_proj = normalize_second_moment(self.proj(e_ijepa))   # (B, z_dim)

        # 2) per-sample dropout (like cutoff logic)
        keep = (torch.rand(B, device=z.device) >= self.sem_mixing_prob).float().view(B, 1)

        # 3) compute additive shift
        strength = self.alpha * sem_ramp                  # scalar
        shift    = keep * strength * e_proj               # (B, z_dim)

        fused = z + shift
        return fused.unsqueeze(1)                         # (B,1,z_dim)
    
class Generator(nn.Module):
    def __init__(
        self,
        z_dim=256,
        c_dim=0,
        w_dim=0,
        img_resolution=256,
        img_channels=3,
        ngf=128,
        cond=0,
        mapping_kwargs={},
        synthesis_kwargs={}
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        # Mapping and Synthesis Networks
        # self.mapping = DummyMapping()  # to fit the StyleGAN API
        self.mapping = IJEPAAddFusion(
            z_dim=z_dim,
            ijepa_dim=2048,
            proj_hid1=1024,
            proj_hid2=512,
            proj_hid3=256,
            lr_multiplier=0.01,
            sem_mixing_prob=0.9
        )
        Synthesis = FastganSynthesisCond if cond else FastganSynthesis
        self.synthesis = Synthesis(ngf=ngf, z_dim=z_dim, nc=img_channels, img_resolution=img_resolution, **synthesis_kwargs)

    def forward(self, z, c, **kwargs):
        w = self.mapping(z, c, **kwargs)
        img = self.synthesis(w, c, **kwargs)
        return img
