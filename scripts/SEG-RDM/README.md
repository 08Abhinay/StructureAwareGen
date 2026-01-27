# SEG-RDM

Standalone port of the RDM training pipeline from the local RCG repo. Uses OmegaConf configs with target/params blocks and preserves the original DDPM/DDIM math, conditioning flow, and pretrained encoder path.

Testing

## Quick start

1) Install minimal deps

```bash
pip install -r requirements_min.txt
```

2) Run single-GPU training

```bash
python -m rdm.main_rdm \
  --config rdm/configs/rdm_default.yaml \
  --batch_size 128 \
  --input_size 256 \
  --epochs 1 \
  --blr 1e-6 \
  --weight_decay 0.01 \
  --output_dir /tmp/seg_rdm_out \
  --data_path /path/to/imagenet_or_dummy
```

3) Run torchrun (DDP) if desired

```bash
torchrun --nproc_per_node=1 -m rdm.main_rdm \
  --config rdm/configs/rdm_default.yaml \
  --batch_size 128 \
  --input_size 256 \
  --epochs 1 \
  --blr 1e-6 \
  --weight_decay 0.01 \
  --output_dir /tmp/seg_rdm_out \
  --data_path /path/to/imagenet_or_dummy
```

## Pretrained encoder checkpoints

The default config expects a MoCo v3 ViT-B checkpoint at:

```
pretrained_enc_ckpts/mocov3/vitb.pth.tar
```

Use an absolute path if you store checkpoints elsewhere by editing `rdm/configs/rdm_default.yaml` or the config you pass in.

## Smoke test

A minimal smoke test is provided in code:

```bash
python - <<'PY'
from rdm.main_rdm import smoke_test
loss, loss_dict, samples = smoke_test('rdm/configs/rdm_default.yaml', device='cuda')
print(loss)
print(list(loss_dict.keys()))
print(samples.shape)
PY
```

## Notebook debug instructions

Use this pattern to run `main()` from a notebook:

```python
import sys
sys.path.append('/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM')

sys.argv = [
    'rdm.main_rdm',
    '--config', 'rdm/configs/rdm_default.yaml',
    '--batch_size', '8',
    '--input_size', '256',
    '--epochs', '1',
    '--blr', '1e-6',
    '--weight_decay', '0.01',
    '--output_dir', '/tmp/seg_rdm_out',
    '--data_path', '/path/to/imagenet_or_dummy',
]

from rdm.main_rdm import get_args_parser, main
args = get_args_parser().parse_args()
main(args)
```
