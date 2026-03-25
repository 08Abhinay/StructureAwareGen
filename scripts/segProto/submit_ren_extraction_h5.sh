#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a100-40gb
#SBATCH -q standby
#SBATCH --job-name=ren-extract
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=80G
#SBATCH --time=04:00:00


#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/ren_extraction/%x-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/ren_extraction/%x-%j.err

# ============================================================
# REN DINOv2 ViT-L/14 Region + CLS Embedding Extraction → H5 Shards
# ============================================================
# 6 nodes × 2 GPUs = 12 ranks (torchrun).
# Each rank processes 1/12 of ImageNet, writes its own H5 shard.
# After extraction, run submit_ren_merge_h5.sh to merge shards.
# ============================================================

PROJECT_ROOT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/segProto"

module load anaconda
conda activate "${PROJECT_ROOT}/SegmentationAwareGen"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"

cd "${SCRIPT_DIR}"

mkdir -p "${PROJECT_ROOT}/logs/ren_extraction" \
         "${PROJECT_ROOT}/region_ren_shards"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "============================================================"
echo "REN DINOv2 Region + CLS Extraction"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Date: $(date)"
echo "============================================================"

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_id="${SLURM_JOB_ID}" \
  "${SCRIPT_DIR}/precompute_ren_embeddings_h5.py" \
  --image_dir "${PROJECT_ROOT}/dataset/imagenet-1K-hf/train" \
  --output_dir "${PROJECT_ROOT}/region_ren_shards" \
  --ren_config "${SCRIPT_DIR}/configs/ren_dinov2_vitl14.yaml" \
  --ren_checkpoint "/scratch/gilbreth/abelde/Thesis/REN/logs/ren-dinov2-vitl14/checkpoint.pth" \
  --image_resolution 518 \
  --grid_size 37 \
  --merge_similarity 0.975 \
  --subset_fraction 0.80 \
  --seed 42 \
  --skip_existing \
  --use_slic

echo "============================================================"
echo "Extraction completed at $(date)"
echo "============================================================"
echo "  Extraction: ${EXTRACT_JOB_ID} (6 nodes × 2 GPUs = 12 ranks)"
echo "  Merge: ${MERGE_JOB_ID} (runs after extraction)"
echo "Monitor: squeue -u \$USER"
echo "Logs: ${LOGDIR}"
echo "============================================================"
