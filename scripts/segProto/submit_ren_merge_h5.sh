#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p training
#SBATCH -q training
#SBATCH --job-name=ren-merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=80G
#SBATCH --time=02:00:00
#SBATCH --constraint=J

#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/ren_extraction/%x-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/ren_extraction/%x-%j.err

# ============================================================
# Merge REN H5 shards into a single flat H5 file
# ============================================================
# Run after extraction completes:
#   sbatch --dependency=afterok:<EXTRACT_JOB_ID> submit_ren_merge_h5.sh
# Or standalone:
#   sbatch submit_ren_merge_h5.sh
# ============================================================

PROJECT_ROOT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/segProto"

module load anaconda
conda activate "${PROJECT_ROOT}/SegmentationAwareGen"

export PYTHONNOUSERSITE=1

SHARD_DIR="${PROJECT_ROOT}/region_ren_shards"
MERGED_H5="${PROJECT_ROOT}/h5_embeddings/region_ren_dinov2_flat.h5"

echo "============================================================"
echo "Merging REN H5 shards → ${MERGED_H5}"
echo "Date: $(date)"
echo "============================================================"

python3 "${SCRIPT_DIR}/merge_h5_shards.py" \
  --shard_dir "${SHARD_DIR}" \
  --output "${MERGED_H5}" \
  --shard_pattern "region_ren_shard_*.h5" \
  --verify

echo "============================================================"
echo "Merge completed at $(date)"
echo "============================================================"
