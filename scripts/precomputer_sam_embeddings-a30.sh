#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a30
#SBATCH -q standby
#SBATCH --job-name=SAM_emb_extract
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=80G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/sam_embeddings-a30.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/sam_embeddings-a30.err

set -e
set -o pipefail

module load anaconda
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen

which python

# Setup multi-node communication
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export OMP_NUM_THREADS=16

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_NODELIST: $SLURM_NODELIST"

# Pre-extract SAM embeddings for 40% of ImageNet (parallel across 2 nodes × 2 GPUs = 4 GPUs)
# This cache will be shared between RDM and StyleGAN2 training
echo "Starting SAM embedding extraction..."
echo "Using 2 nodes with 2 GPUs each (4 GPUs total) for 40% of ImageNet"

srun torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/precompute_sam_embeddings.py \
    --image_dir /scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/train \
    --output_dir /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/seg-aware-stylegan2/sam_cache_unified \
    --checkpoint /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/checkpoints/sam_vit_b_01ec64.pth \
    --subset_fraction 0.4 \
    --seed 42 \
    --skip_existing

echo "SAM extraction finished!"
echo "Cache ready for RDM and StyleGAN2 training."





