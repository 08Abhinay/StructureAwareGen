# SEG-RDM

Standalone Representation Diffusion Model (RDM) training extracted from the local RCG codebase.
This version removes all pretrained encoder dependencies and trains directly on representation vectors.

## Quick Start (random reps)

```bash
python -m seg_rdm.main_rdm \
  --config seg_rdm/configs/default_rdm.yaml \
  --batch_size 128 \
  --epochs 1 \
  --blr 1e-6 \
  --weight_decay 0.01 \
  --output_dir /tmp/seg_rdm_out \
  --rep_source random
```

## Train with cached reps (.npz)

```bash
python -m seg_rdm.main_rdm \
  --config seg_rdm/configs/default_rdm.yaml \
  --batch_size 128 \
  --epochs 1 \
  --blr 1e-6 \
  --weight_decay 0.01 \
  --output_dir /tmp/seg_rdm_out \
  --rep_source npz \
  --npz_path /path/to/reps.npz \
  --npz_key reps
```

Expected NPZ array shape: `[N, rep_dim]`.

## Notebook usage

```python
import sys
from seg_rdm.main_rdm import main, get_args_parser

sys.argv = [
    "-m", "seg_rdm.main_rdm",
    "--config", "seg_rdm/configs/default_rdm.yaml",
    "--rep_source", "random",
    "--output_dir", "/tmp/seg_rdm_out",
    "--epochs", "1",
]
args = get_args_parser().parse_args()
main(args)
```

## Sampling

```python
import torch
from omegaconf import OmegaConf
from seg_rdm.utils.misc import instantiate_from_config
from seg_rdm.diffusion.ddim import DDIMSampler

cfg = OmegaConf.load("seg_rdm/configs/default_rdm.yaml")
model = instantiate_from_config(cfg.model).cuda().eval()

sampler = DDIMSampler(model)
samples, _ = sampler.sample(
    S=50,
    batch_size=8,
    shape=(cfg.rep_dim, 1, 1),
    conditioning=None,
)

# samples shape: [B, rep_dim, 1, 1]
reps = samples[:, :, 0, 0].cpu()
```

## Smoke test

```bash
python -m seg_rdm.main_rdm --config seg_rdm/configs/default_rdm.yaml --smoke_test
```

This runs one forward+loss step and one sampling call on random reps.

## Notes

- This code preserves the original DDPM/DDIM math and the SimpleMLP denoiser structure.
- All pretrained encoder code paths are removed; training consumes representation vectors directly.
- Dependencies are minimal: torch, numpy, omegaconf.
