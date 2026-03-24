#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p training
#SBATCH -q training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/moco_extraction/moco_merge_h5-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/moco_extraction/moco_merge_h5-%j.err

module load anaconda
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen
export PYTHONNOUSERSITE=1

echo "============================================================"
echo "Merging H5 shards → /scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/region_moco_flat.h5"
echo "Date: $(date)"
echo "============================================================"

python3 "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/merge_h5_shards.py" \
  --shard_dir "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/region_moco_shards" \
  --output "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/region_moco_flat.h5" \
  --verify

echo "============================================================"
echo "Merge completed at $(date)"
echo "============================================================"
