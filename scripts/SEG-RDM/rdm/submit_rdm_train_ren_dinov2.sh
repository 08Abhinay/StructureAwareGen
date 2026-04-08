#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a30
#SBATCH -q standby
#SBATCH --job-name=RDM-ren-dinov2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=80G
#SBATCH --time=04:00:00

#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/RDM_ren_dinov2/%x-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/RDM_ren_dinov2/%x-%j.err

# ============================================================
# Unified RDM: REN DINOv2 regions (1024d) + DINOv2 CLS (1024d)
# ============================================================
# Same embedding space for both streams.
# UnifiedSegTransformer with d_model=1024, max_segments=100.
# ============================================================

PROJECT_ROOT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen"
SEG_RDM_ROOT="${PROJECT_ROOT}/scripts/SEG-RDM"
RDM_ROOT="${SEG_RDM_ROOT}/rdm"

RUN_DIR="${RDM_ROOT}/rdm_out_final/ren_dinov2/4x2_nodes-a30/"

module load anaconda
conda activate "${PROJECT_ROOT}/SegmentationAwareGen"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export HDF5_USE_FILE_LOCKING=FALSE

# NCCL: do NOT hardcode NCCL_SOCKET_IFNAME — interface names differ between node types
# NCCL auto-detects the correct interface; IB/RoCE will be used for data transport
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

cd "${SEG_RDM_ROOT}"
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir -p "${RDM_ROOT}/SLRUM_OUTPUT_FILES/RDM_ren_dinov2"
mkdir -p "${RUN_DIR}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

RESUME_ARGS=()
if ls "${RUN_DIR}"/checkpoint-*.pth 1> /dev/null 2>&1; then
  LAST_CKPT=$(ls -t "${RUN_DIR}"/checkpoint-*.pth | head -n 1)
  RESUME_ARGS=(--resume "$LAST_CKPT")
fi

TORCHRUN_CMD=(
  torchrun
  --nnodes="$SLURM_NNODES"
  --nproc_per_node=2
  --rdzv_backend=c10d
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT"
  --rdzv_id="$SLURM_JOB_ID"
  -m rdm.main_rdm
  --config "${RDM_ROOT}/configs/unified_seg_rdm_ren_dinov2.yaml"
  --input_size 256
  --blr 1.25e-7
  --min_lr 1e-6
  --cosine_lr
  --warmup_epochs 5
  --weight_decay 0.01
  --epochs 300
  --batch_size 128
  --accum_iter 1
  --num_workers 8
  --output_dir "${RUN_DIR}"
  --log_dir "${RUN_DIR}"
  --data_path "${PROJECT_ROOT}/dataset/imagenet-1K-hf"
  --use_seg_dataset
  --emb_source dinov2
  --h5_path "${PROJECT_ROOT}/h5_embeddings/region_ren_dinov2_flat.h5"
  --image_h5_path "${PROJECT_ROOT}/h5_embeddings/imagenet_train_images.h5"
  --max_segments 100
)

if [ ${#RESUME_ARGS[@]} -gt 0 ]; then
  TORCHRUN_CMD+=("${RESUME_ARGS[@]}")
fi

srun "${TORCHRUN_CMD[@]}"
