#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a100-80gb
#SBATCH -q normal
#SBATCH --job-name=RDM-quality
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=24G
#SBATCH --time=24:00:00

#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/quality_check-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/quality_check-%j.err

PROJECT_ROOT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen"
RDM_ROOT="${PROJECT_ROOT}/scripts/SEG-RDM/rdm"

module load anaconda
conda activate "${PROJECT_ROOT}/SegmentationAwareGen"

export PYTHONNOUSERSITE=1

cd "${RDM_ROOT}"
# python multi_seg_check.py
python sample_quality_check.py \
  --config /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/configs/unified_seg_rdm_hybrid_crossattn.yaml \
  --ckpt /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/rdm_out_final/hybrid_crossattn/2_nodes/checkpoint-last.pth \
  --compute_rep_fid \
  --rep_fid_n_gen 50000 \
  --rep_fid_n_real 50000 \
  --rep_fid_batch_size 16

