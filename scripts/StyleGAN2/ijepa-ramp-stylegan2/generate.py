# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
from training.ijepa_encoder import build_ijepa_encoder

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--ijepa_checkpoint', help='IJEPA checkpoint for embedding conditioning', type=str)
@click.option('--ijepa_ref', help='Reference image used to compute IJEPA embedding', type=str, metavar='FILE')
@click.option('--ijepa_image', help='IJEPA encoder input resolution', type=int, default=256, show_default=True)
@click.option('--ijepa_input_channel', help='IJEPA encoder input channels', type=int)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    ijepa_checkpoint: Optional[str],
    ijepa_ref: Optional[str],
    ijepa_image: int,
    ijepa_input_channel: Optional[int],
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    ijepa_enabled = hasattr(G, "mapping") and hasattr(G.mapping, "proj_ijepa")
    e_ijepa = None
    if ijepa_checkpoint is not None or ijepa_ref is not None:
        if ijepa_checkpoint is None or ijepa_ref is None:
            ctx.fail('Both --ijepa_checkpoint and --ijepa_ref are required for IJEPA conditioning.')
        ijepa_in_ch = ijepa_input_channel if ijepa_input_channel is not None else G.img_channels
        ijepa_enc, _ = build_ijepa_encoder(
            ijepa_checkpoint,
            device=device,
            in_channels_override=ijepa_in_ch,
            img_size=ijepa_image,
        )
        ijepa_enc.eval().requires_grad_(False)
        ref_img = PIL.Image.open(ijepa_ref)
        if ijepa_in_ch == 1:
            ref_img = ref_img.convert('L')
        else:
            ref_img = ref_img.convert('RGB')
        ref_arr = np.array(ref_img)
        if ref_arr.ndim == 2:
            ref_arr = ref_arr[None, :, :]
        else:
            ref_arr = ref_arr.transpose(2, 0, 1)
        ref_tensor = torch.from_numpy(ref_arr).unsqueeze(0).to(device).float()
        ref_tensor = ref_tensor / 127.5 - 1.0
        if ref_tensor.shape[1] < ijepa_in_ch:
            ref_tensor = ref_tensor.repeat(1, ijepa_in_ch // ref_tensor.shape[1], 1, 1)
        elif ref_tensor.shape[1] > ijepa_in_ch:
            ref_tensor = ref_tensor.mean(1, keepdim=True)
        with torch.no_grad():
            e_ijepa = ijepa_enc(ref_tensor)
    elif ijepa_enabled:
        ctx.fail('IJEPA-enabled generator detected; pass --ijepa_checkpoint and --ijepa_ref to use embeddings.')

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-w')
        if e_ijepa is not None:
            print('warn: IJEPA embedding is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        # img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        if e_ijepa is not None:
            img = G(z, label, e_ijepa=e_ijepa, sem_ramp=1.0, truncation_psi=truncation_psi, noise_mode=noise_mode)
        else:
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
