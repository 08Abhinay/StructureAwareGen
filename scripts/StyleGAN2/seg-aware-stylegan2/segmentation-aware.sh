#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a30
#SBATCH -q standby
#SBATCH --job-name=stylegan2_ijepa_chest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=80G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/seg-aware-stylegan2/SLRUM_OUTPUT_FILES/stylegan2_ijepa_chest-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/seg-aware-stylegan2/SLRUM_OUTPUT_FILES/stylegan2_ijepa_chest-%j.err

set -e
set -o pipefail

# ---- Paths ----
REPO_ROOT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/seg-aware-stylegan2"
DATA_PATH="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/256/chest_xray_labelled.zip"
OUT_BASE="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/seg-aware-stylegan2/outputs/Chest"
IJEPA_CKPT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/pretrained_enc_ckpts/ijepa/IN1K-vit.h.14-300e.pth.tar"

# ---- Pre-computed embedding directories ----
SAM_NPZ_DIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/256/sam_embeddings"
IJEPA_NPZ_DIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/256/ijepa_embeddings"

# ---- RDM mixed training (Stage-1 checkpoint) ----
# Set RDM_CKPT to enable mixed training; leave empty to disable
RDM_CKPT=""  # e.g. "/scratch/gilbreth/abelde/Thesis/.../seg_rdm_epoch_200.pt"
RDM_MIX_PROB="0.3"
RDM_WARMUP_KIMG="10000"

# ---- Hyperparams ----
SEM_MIX="0.9"
FUSION_ALPHA="0.2"
LAMBDA_SEG_ALIGN="0.1"
LAMBDA_SEG_DIVERSITY="0.05"

# ---- Resources ----
GPUS=1   # must match #SBATCH --gpus-per-node

OUTDIR="${OUT_BASE}/sem_mixing_prob_${SEM_MIX}/ijepa_Chest_rampGD_warmup_5.4_${GPUS}gpu_sem_mix_${SEM_MIX}_FusionAlpha_${FUSION_ALPHA}/training-runs"

# ---- Env ----
module load anaconda
module load cuda/12.6.0 
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen

which python
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'file:', torch.__file__)"
python -c "import torch; print('is_available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print('device0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA')"

# ---- Prevent ~/.local pollution (important) ----
export PYTHONNOUSERSITE=1

# Keep PYTHONPATH controlled; add only the repo root.
unset PYTHONPATH || true
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${REPO_ROOT}/SLRUM_OUTPUT_FILES"
mkdir -p "${OUTDIR}"

cd "${REPO_ROOT}"

# ---- Run ----
# If you later set GPUS>1, change #SBATCH --gpus-per-node too.
# Some StyleGAN2 forks use torch.multiprocessing internally when --gpus > 1.
python train.py \
  --outdir "${OUTDIR}" \
  --data "${DATA_PATH}" \
  --gpus "${GPUS}" \
  --cond 1 \
  --ijepa_checkpoint "${IJEPA_CKPT}" \
  --ijepa_lambda 1.0 \
  --ijepa_image 256 \
  --ijepa_input_channel 3 \
  --ijepa_dim 1280 \
  --ijepa_warmup_kimg 5.4 \
  --sem_mixing_prob "${SEM_MIX}" \
  --fusion_alpha "${FUSION_ALPHA}" \
  --use-seg-embeddings \
  --sam-npz-dir "${SAM_NPZ_DIR}" \
  --ijepa-npz-dir "${IJEPA_NPZ_DIR}" \
  --lambda-seg-align "${LAMBDA_SEG_ALIGN}" \
  --lambda-seg-diversity "${LAMBDA_SEG_DIVERSITY}" \
  ${RDM_CKPT:+--rdm-checkpoint "${RDM_CKPT}"} \
  ${RDM_CKPT:+--rdm-mix-prob "${RDM_MIX_PROB}"} \
  ${RDM_CKPT:+--rdm-warmup-kimg "${RDM_WARMUP_KIMG}"} \
  --resume noresume


# python3 /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/seg-aware-stylegan2/versions.py