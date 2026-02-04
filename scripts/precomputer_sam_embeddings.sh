#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a100-40gb
#SBATCH -q standby
#SBATCH --job-name=SAM_emb_extract
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=80G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/sam_embeddings.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/sam_embeddings.err

set -e
set -o pipefail

module load anaconda
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen

which python

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))

# Get master node for coordination
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Pre-extract SAM embeddings for 40% of ImageNet (parallel across 2 nodes × 2 GPUs)
# This cache will be shared between RDM and StyleGAN2 training
echo "Starting SAM embedding extraction..."
echo "Using 2 nodes with 2 GPUs each (4 total GPUs) for 40% of ImageNet"
echo "Master node: $MASTER_ADDR:$MASTER_PORT"

# One torchrun per node (srun launches 2 tasks total, 1 per node)
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=2 \
    --node_rank=$SLURM_PROCID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/precompute_sam_embeddings.py \
      --image_dir /scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/train \
      --output_dir /scratch/gilbreth/abelde/Thesis/StructureAwareGen/sam_cache_unified \
      --checkpoint /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/checkpoints/sam_vit_b_01ec64.pth \
      --subset_fraction 0.4 \
      --seed 42 \
      --skip_existing

echo "SAM extraction finished!"
echo "Cache ready for RDM and StyleGAN2 training."