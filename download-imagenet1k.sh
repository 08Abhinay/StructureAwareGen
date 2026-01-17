#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p training
#SBATCH -q training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --constraint=J
#SBATCH -J imagenet_hf_download-job
#SBATCH -o /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SLRUM_OUTPUT_FILES/imagenet_download-hf.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SLRUM_OUTPUT_FILES/imagenet_download-hf.err

module load anaconda

# ✅ Use whatever env you prefer; this assumes you made one called 'imagenet_dl'
# If not, either create it once:
#   conda create -n imagenet_dl python=3.10 -y
#   conda activate imagenet_dl
#   pip install datasets huggingface_hub pillow
# and then use it here:
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen

# ==========================
# Cache dirs on /scratch
# ==========================
export HF_HOME=/scratch/gilbreth/abelde/hf_cache
export HF_DATASETS_CACHE=/scratch/gilbreth/abelde/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/gilbreth/abelde/hf_cache/transformers

# Generic cache root (huggingface_hub also respects this)
export XDG_CACHE_HOME=/scratch/gilbreth/abelde/.cache

# (Optional, but safe to keep for your other work)
export TORCH_HOME=/scratch/gilbreth/abelde/torch_cache
export TORCH_CACHE=/scratch/gilbreth/abelde/torch_cache

# Create the dirs so HF doesn’t crash trying to write
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME"


# If your env is elsewhere, replace the line above with:
# conda activate /scratch/gilbreth/abelde/your_env_name

# Run the download script
python /scratch/gilbreth/abelde/Thesis/StructureAwareGen/download_imagenet_hf.py
