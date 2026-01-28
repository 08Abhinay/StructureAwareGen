#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a30
#SBATCH -q standby
#SBATCH --job-name=rdm_ijepa_h14_2g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=80G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/rdm_ijepa_h14_2g-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/rdm_ijepa_h14_2g-%j.err

set -e
set -o pipefail

module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen

which python

cd /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM
export PYTHONPATH="$PWD:$PYTHONPATH"
export PATH="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen/bin:$PATH"

mkdir -p /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/rdm_out/ijepa_h14

srun torchrun --standalone --nproc_per_node=2 -m rdm.main_rdm \
  --config /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/configs/rdm_default.yaml \
  --input_size 256 \
  --blr 1e-6 \
  --weight_decay 0.01 \
  --epochs 200 \
  --batch_size 16 \
  --accum_iter 1 \
  --output_dir /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/rdm_out/ijepa_h14 \
  --log_dir /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/rdm_out/ijepa_h14 \
  --data_path /scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf