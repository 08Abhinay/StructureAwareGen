import math

import torch


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise():
        noise = torch.randn((1, *shape[1:]), device=device)
        return noise.repeat(shape[0], *((1,) * (len(shape) - 1)))

    def fresh_noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else fresh_noise()


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if repeat_only:
        return timesteps.float().unsqueeze(-1).repeat(1, dim)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
