#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a100-40gb
#SBATCH -q standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH -J ijepa_Chest_rampGD_warmup_5.4_2gpu_sem_mix_0.9_FusionAlpha_0.2
#SBATCH -o /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/ijepa-ramp-stylegan2/SLRUM_OUTPUT_FILES/custon-run-training.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/ijepa-ramp-stylegan2/SLRUM_OUTPUT_FILES/custon-run-training.err

set -euo pipefail

# -----------------------
# Paths (updated to new StructureAwareGen tree)
# -----------------------
PROJECT=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/stylegan2-ada-pytorch
ENV_PREFIX=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen

DATA_ZIP=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/256/chest_xray_labelled.zip
IJEPA_CKPT=/scratch/gilbreth/abelde/Thesis/CNN-JEPA/artifacts/pretrain_lightly/ijepacnn_chestxray-h5/ijepa_chestxray-h5_resnet50_bs64/version_2/ijepa_backbone_momentum_only.pth

# Training output location (kept consistent with your new path style)
OUTDIR=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/ijepa-ramp-stylegan2/outputs/Chest/sem_mixing_prob_0.9/ijepa_Chest_rampGD_warmup_5.4_2gpu_sem_mix_0.9_FusionAlpha_0.2/training-runs

mkdir -p "$OUTDIR"
mkdir -p /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/ijepa-ramp-stylegan2/SLRUM_OUTPUT_FILES

# -----------------------
# Critical fix: SHORT tmp paths (avoids AF_UNIX path too long)
# -----------------------
export TMPDIR=/tmp/$SLURM_JOB_ID
export TEMP=/tmp/$SLURM_JOB_ID
export TMP=/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR"

export TORCH_EXTENSIONS_DIR=/tmp/$SLURM_JOB_ID/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"

export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/cache
mkdir -p "$XDG_CACHE_HOME"

# Optional: reduce thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONWARNINGS="ignore:conv2d_gradfix not supported:UserWarning"


# Hugging Face / Transformers cache in scratch (safe to keep even if not used)
export HF_HOME=/scratch/gilbreth/abelde/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HUB_CACHE"

# -----------------------
# Load CUDA + activate env
# -----------------------
module load cuda/12.6.0
source /apps/external/anaconda/2025.06/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

cd "$PROJECT"

# -----------------------
# Hyperparams (matches the “new” naming you’re using)
# -----------------------
SEM_MIX=0.9
FUSION_ALPHA=0.2

# -----------------------
# Sanity prints
# -----------------------
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('torch', torch.__version__, 'torch.version.cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"
nvidia-smi || true

# -----------------------
# Train (NO apptainer, use your env python)
# -----------------------
srun --gpus=2 python /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/ijepa-ramp-stylegan2/train.py \
  --outdir="$OUTDIR" \
  --data="$DATA_ZIP" \
  --gpus=2 \
  --cond=1 \
  --ijepa_checkpoint "$IJEPA_CKPT" \
  --ijepa_lambda 1.0 \
  --ijepa_image 256 \
  --ijepa_input_channel 3 \
  --extra_dim 2048 \
  --ijepa_warmup_kimg 5.4 \
  --sem_mixing_prob "$SEM_MIX" \
  --fusion_alpha "$FUSION_ALPHA"
