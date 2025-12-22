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
#SBATCH -o /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/ijepa-ramp-stylegan2/SLRUM_OUTPUT_FILES/custom-testing.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/ijepa-ramp-stylegan2/SLRUM_OUTPUT_FILES/custom-testing.err

set -euo pipefail

# -----------------------
# Paths
# -----------------------
PROJECT=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/stylegan2-ada-pytorch
ENV_PREFIX=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen

NETWORK_PKL=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/ijepa-ramp-stylegan2/outputs/Chest/sem_mixing_prob_0.9/ijepa_Chest_rampGD_warmup_5.4_4gpu_sem_mix_0.9_FusionAlpha_0.2/training-runs/00012-chest_xray_labelled-cond-auto4-resumecustom/network-snapshot-001814.pkl
DATA_ZIP=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/256/chest_xray_labelled.zip

OUTDIR=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/ijepa-ramp-stylegan2/SLRUM_OUTPUT_FILES
mkdir -p "$OUTDIR"

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

# Hugging Face / Transformers cache in scratch
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
# Sanity prints
# -----------------------
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('torch', torch.__version__, 'torch.version.cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"
nvidia-smi || true

# -----------------------
# Run ONE metric job (request 2 GPUs for the step)
# -----------------------
srun --gpus=2 python calc_metrics_pr.py \
  --metrics pr50k3_full_cond \
  --network "$NETWORK_PKL" \
  --data "$DATA_ZIP"
