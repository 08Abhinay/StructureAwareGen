import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


### single layers

# -----------------------------------------------------------
# Global constant: dimensionality of the IJepa embedding
_SEM_DIM = 2048          # <- set to whatever your encoder outputs
# -----------------------------------------------------------

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def embedding(*args, **kwargs):
    return spectral_norm(nn.Embedding(*args, **kwargs))


def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))


def NormLayer(c, mode='batch'):
    if mode == 'group':
        return nn.GroupNorm(c//2, c)
    elif mode == 'batch':
        return nn.BatchNorm2d(c)


### Activations


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


### Upblocks


class InitLayer(nn.Module):
    def __init__(self, nz, channel, sz=4):
        super().__init__()

        self.init = nn.Sequential(
            convTranspose2d(nz, channel*2, sz, 1, 0, bias=False),
            NormLayer(channel*2),
            GLU(),
        )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlockSmall(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        NormLayer(out_planes*2), GLU())
    return block



class UpBlockSmallCond(nn.Module):
    def __init__(self, in_planes, out_planes, z_dim):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)

        which_bn = functools.partial(CCBN,
                                     which_linear=linear,
                                     input_size=z_dim,
                                     sem_dim=_SEM_DIM)
        self.bn  = which_bn(2 * out_planes)
        self.act = GLU()

    def forward(self, x, c, *, e_ijepa=None):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x, c, e_ijepa=e_ijepa)
        x = self.act(x)
        return x
    
    
# --- FiLM --------------------------------------------------------------
class FiLM(nn.Module):
    """Per-feature FiLM: y = (1 + gamma) * x + β."""
    def __init__(self, channels, z_dim):
        super().__init__()
        self.fc = linear(z_dim, 2 * channels)   # spectral-norm wrapped

        # small init so early training ≈ identity
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, z):
        # x : (B, C, H, W)    z : (B, z_dim)
        gamma, beta = self.fc(z).chunk(2, dim=1)       # (B, C) each
        gamma = gamma.unsqueeze(2).unsqueeze(3)        # (B, C, 1, 1)
        beta  =  beta .unsqueeze(2).unsqueeze(3)
        return (1 + gamma) * x + beta
# ----------------------------------------------------------------------

# --------------------------------------------------------------
class UpBlockBigFiLM(nn.Module):
    def __init__(self, in_planes, out_planes, z_dim):
        super().__init__()
        self.up     = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1  = conv2d(in_planes,  out_planes*2, 3, 1, 1, bias=False)
        self.conv2  = conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False)

        self.noise  = NoiseInjection()
        self.bn1    = NormLayer(out_planes*2)
        self.bn2    = NormLayer(out_planes*2)
        self.act    = GLU()

        self.film1  = FiLM(out_planes, z_dim)   # expect C = out_planes
        self.film2  = FiLM(out_planes, z_dim)

    def forward(self, x, z):
        # block 1
        x = self.up(x)
        x = self.noise(self.conv1(x))
        x = self.bn1(x)
        x = self.act(x)           # GLU halves channels → C = out_planes
        x = self.film1(x, z)

        # block 2
        x = self.noise(self.conv2(x))
        x = self.bn2(x)
        x = self.act(x)
        x = self.film2(x, z)
        return x


class UpBlockSmallFiLM(nn.Module):
    def __init__(self, in_planes, out_planes, z_dim):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)
        self.bn   = NormLayer(out_planes*2)
        self.act  = GLU()
        self.film = FiLM(out_planes, z_dim)

    def forward(self, x, z):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)        # after GLU → C = out_planes
        x = self.film(x, z)
        return x
# --------------------------------------------------------------




def UpBlockBig(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        NormLayer(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        NormLayer(out_planes*2), GLU()
        )
    return block


# --------------------------------------------------------------
class UpBlockBigCond(nn.Module):
    def __init__(self, in_planes, out_planes, z_dim):
        super().__init__()
        self.up     = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1  = conv2d(in_planes,  out_planes*2, 3, 1, 1, bias=False)
        self.conv2  = conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False)

        which_bn = functools.partial(CCBN,
                                     which_linear=linear,
                                     input_size=z_dim,
                                     sem_dim=_SEM_DIM)
        self.bn1  = which_bn(2 * out_planes)
        self.bn2  = which_bn(2 * out_planes)

        self.act   = GLU()
        self.noise = NoiseInjection()

    def forward(self, x, c, *, e_ijepa=None):
        # ── block 1 ─────────────────────────────────────────────
        x = self.up(x)
        x = self.conv1(x)
        x = self.noise(x)
        x = self.bn1(x, c, e_ijepa=e_ijepa)
        x = self.act(x)

        # ── block 2 ─────────────────────────────────────────────
        x = self.conv2(x)
        x = self.noise(x)
        x = self.bn2(x, c, e_ijepa=e_ijepa)
        x = self.act(x)
        return x



class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            conv2d(ch_in, ch_out, 4, 1, 0, bias=False),
            Swish(),
            conv2d(ch_out, ch_out, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


### Downblocks


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = conv2d(in_channels, in_channels, kernel_size=kernel_size,
            groups=in_channels, bias=bias, padding=1)
        self.pointwise = conv2d(in_channels, out_channels,
            kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, separable=False):
        super().__init__()
        if not separable:
            self.main = nn.Sequential(
                conv2d(in_planes, out_planes, 4, 2, 1),
                NormLayer(out_planes),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.main = nn.Sequential(
                SeparableConv2d(in_planes, out_planes, 3),
                NormLayer(out_planes),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2, 2),
            )

    def forward(self, feat):
        return self.main(feat)


class DownBlockPatch(nn.Module):
    def __init__(self, in_planes, out_planes, separable=False):
        super().__init__()
        self.main = nn.Sequential(
            DownBlock(in_planes, out_planes, separable),
            conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
            NormLayer(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)

import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

class DownBlockCond(nn.Module):
    def __init__(self, in_planes, out_planes, embedding_dim, sem_dim=_SEM_DIM, separable=False):
        super().__init__()
        if not separable:
            self.conv = conv2d(in_planes, out_planes, 4, 2, 1)
        else:
            self.conv = nn.Sequential(
                SeparableConv2d(in_planes, out_planes, 3),
                nn.AvgPool2d(2, 2),
            )
        # conditional batch-norm that also takes e_ijepa:
        self.bn  = CCBN(out_planes, embedding_dim, which_linear=linear, sem_dim=sem_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, c_embed, *, e_ijepa=None, sem_ramp=1.0):
        x = self.conv(x)
        # e_ijepa already faded by sem_ramp upstream (see below)
        x = self.bn(x, c_embed, e_ijepa=e_ijepa)
        return self.act(x)


class DownBlockPatchCond(nn.Module):
    def __init__(self, in_planes, out_planes, embedding_dim, sem_dim=_SEM_DIM, separable=False):
        super().__init__()
        self.down = DownBlockCond(in_planes, out_planes, embedding_dim, sem_dim, separable)
        self.conv1 = conv2d(out_planes, out_planes, 1, 1, 0, bias=False)
        self.bn2   = CCBN(out_planes, embedding_dim, which_linear=linear, sem_dim=sem_dim)
        self.act2  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, c_embed, *, e_ijepa=None, sem_ramp=1.0):
        x = self.down(x, c_embed, e_ijepa=e_ijepa, sem_ramp=sem_ramp)
        x = self.conv1(x)
        x = self.bn2(x, c_embed, e_ijepa=e_ijepa)
        return self.act2(x)

### CSM


class ResidualConvUnit(nn.Module):
    def __init__(self, cin, activation, bn):
        super().__init__()
        self.conv = nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=True)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.skip_add.add(self.conv(x), x)


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, lowest=False):
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output = self.skip_add.add(output, xs[1])

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


### Misc


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


import torch
import torch.nn as nn
import torch.nn.functional as F

class CCBN(nn.Module):
    """
    Conditional BatchNorm extended with I-JEPA conditioning.

    Parameters
    ----------
    output_size : int
        Number of feature channels to normalize.
    input_size : int
        Dimensionality of the class-conditioning vector `y`.
    which_linear : callable
        Factory for a linear layer (e.g. spectral-norm wrapped nn.Linear).
    sem_dim : int
        Dimensionality of the I-JEPA embedding `e_ijepa`.
    eps : float
        Epsilon for batch-norm.
    momentum : float
        Momentum for running statistics.
    """
    def __init__(self, output_size, input_size, which_linear, sem_dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.output_size = output_size
        self.input_size  = input_size
        self.eps         = eps
        self.momentum    = momentum

        # class-conditioned gain & bias
        self.gain       = which_linear(input_size,  output_size)
        self.bias       = which_linear(input_size,  output_size)
        # semantic (I-JEPA) gain & bias
        self.gain_sem   = which_linear(sem_dim,      output_size)
        self.bias_sem   = which_linear(sem_dim,      output_size)

        # running statistics for batch-norm
        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var',  torch.ones(output_size))

    def forward(self, x, y, *, e_ijepa=None):
        """
        x        : Tensor, shape (B, C, H, W)
        y        : Tensor, shape (B, input_size) — class embedding
        e_ijepa  : Tensor or None, shape (B, sem_dim)
        """
        # class-conditioned parameters
        gain_c = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias_c =  self.bias(y).view(y.size(0), -1, 1, 1)

        # semantic-conditioned parameters (zero if not provided)
        if e_ijepa is not None:
            gain_s = self.gain_sem(e_ijepa).view(e_ijepa.size(0), -1, 1, 1)
            bias_s = self.bias_sem(e_ijepa).view(e_ijepa.size(0), -1, 1, 1)
        else:
            
            gain_s = torch.zeros_like(gain_c)
            bias_s = torch.zeros_like(bias_c)

        # apply batch-norm without learned affine
        out = F.batch_norm(
            x, self.stored_mean, self.stored_var,
            weight=None, bias=None,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps
        )
        # combine class and semantic modulations
        return out * (gain_c + gain_s) + (bias_c + bias_s)


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, size, mode='bilinear', align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            size=self.size,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x
