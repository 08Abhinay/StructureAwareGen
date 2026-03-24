#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p training
#SBATCH -q training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --constraint=J
#SBATCH --job-name=moco_cls_extract
#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/moco_cls_extraction/moco_cls_extract_%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/moco_cls_extraction/moco_cls_extract_%j.err

module load anaconda
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "MoCo v3 CLS Extraction"
echo "Host: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Date: $(date)"
echo "============================================================"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

srun --ntasks-per-node=1 torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/precompute_moco_cls_h5.py" \
  --image_dir "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/train" \
  --output_dir "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/moco_cls_shards" \
  --moco_checkpoint "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/pretrained_enc_ckpts/mocov3/vitl.pth.tar" \
  --merged_h5 "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/moco_cls_flat.h5" \
  --merged_json "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/moco_cls_lookup.json" \
  --batch_size 256 \
  --num_workers 8

echo "============================================================"
echo "Extraction + merge completed at $(date)"
echo "============================================================"
