#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a30
#SBATCH -q standby
#SBATCH --job-name=RDM-xattn
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=80G
#SBATCH --time=04:00:00


#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/RDM_hybrid_crossattn/a30%x-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/RDM_hybrid_crossattn/a30%x-%j.err

# ============================================================
# Cross-Attention RDM: I-JEPA regions (256d) + MoCo CLS (256d)
# ============================================================
# Same data as hybrid self-attention run, but uses
# CrossAttentionSegTransformer instead of UnifiedSegTransformer.
# Separate input projections + cross-attention between global/segment
# streams, since they live in different embedding spaces.
# ============================================================

PROJECT_ROOT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen"
SEG_RDM_ROOT="${PROJECT_ROOT}/scripts/SEG-RDM"
RDM_ROOT="${SEG_RDM_ROOT}/rdm"

RUN_DIR="${RDM_ROOT}/rdm_out_final/hybrid_crossattn/6_nodes/"

module load anaconda
conda activate "${PROJECT_ROOT}/SegmentationAwareGen"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

cd "${SEG_RDM_ROOT}"
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir -p "${RDM_ROOT}/SLRUM_OUTPUT_FILES/RDM_hybrid_crossattn"
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
  --node_rank="$SLURM_PROCID"
  --rdzv_backend=c10d
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT"
  --rdzv_id="$SLURM_JOB_ID"
  -m rdm.main_rdm
  --config "${RDM_ROOT}/configs/unified_seg_rdm_hybrid_crossattn.yaml"
  --input_size 256
  --blr 1.25e-7
  --min_lr 1e-6
  --cosine_lr
  --warmup_epochs 5
  --weight_decay 0.01
  --epochs 300
  --batch_size 32
  --accum_iter 1
  --num_workers 4
  --output_dir "${RUN_DIR}"
  --log_dir "${RUN_DIR}"
  --data_path "${PROJECT_ROOT}/dataset/imagenet-1K-hf"
  --use_seg_dataset
  --h5_path "${PROJECT_ROOT}/h5_embeddings/region_emb_flat.h5"
  --ijepa_h5_path "${PROJECT_ROOT}/h5_embeddings/moco_cls_flat.h5"
  --ijepa_lookup_json "${PROJECT_ROOT}/h5_embeddings/moco_cls_lookup.json"
  --cls_from_external_h5
  --max_segments 100
)

if [ ${#RESUME_ARGS[@]} -gt 0 ]; then
  TORCHRUN_CMD+=("${RESUME_ARGS[@]}")
fi

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
  "${TORCHRUN_CMD[@]}"
