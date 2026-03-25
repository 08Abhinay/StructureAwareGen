#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a100-80gb
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/ren_extraction/ren_merge_h5-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/ren_extraction/ren_merge_h5-%j.err

module load anaconda
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen
export PYTHONNOUSERSITE=1

echo "============================================================"
echo "Merging REN H5 shards → /scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/region_ren_dinov2_flat.h5"
echo "Date: $(date)"
echo "============================================================"

python3 "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/merge_h5_shards.py" \
  --shard_dir "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/region_ren_shards" \
  --output "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/region_ren_dinov2_flat.h5" \
  --shard_pattern "region_ren_shard_*.h5" \
  --verify

echo "============================================================"
echo "Merge completed at $(date)"
echo "============================================================"
