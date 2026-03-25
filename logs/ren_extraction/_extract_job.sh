#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p training
#SBATCH -q training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=80G
#SBATCH --time=24:00:00
#SBATCH --constraint=J
#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/ren_extraction/ren_extract_all-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/ren_extraction/ren_extract_all-%j.err

# Load environment
module load anaconda
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 3))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "============================================================"
echo "REN DINOv2 Region + CLS Extraction"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Date: $(date)"
echo "============================================================"

cd /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto

srun torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --rdzv_id=${SLURM_JOB_ID} \
  "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/precompute_ren_embeddings_h5.py" \
  --image_dir "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/train" \
  --output_dir "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/region_ren_shards" \
  --ren_config "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/configs/ren_dinov2_vitl14.yaml" \
  --ren_checkpoint "/scratch/gilbreth/abelde/Thesis/REN/logs/ren-dinov2-vitl14/checkpoint.pth" \
  --torch_home "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/.cache/torch" \
  --image_resolution 518 \
  --grid_size 37 \
  --merge_similarity 0.975 \
  --subset_fraction 0.80 \
  --seed 42 \
  --skip_existing --use_slic

echo "============================================================"
echo "Extraction completed at $(date)"
echo "============================================================"
