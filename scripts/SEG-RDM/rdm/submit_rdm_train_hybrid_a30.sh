#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a30
#SBATCH -q standby
#SBATCH --job-name=RDM-hybrid-a30
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=80G
#SBATCH --time=4:00:00

#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/RDM_hybrid_moco_cls/%x-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/RDM_hybrid_moco_cls/%x-%j.err

# ============================================================
# Hybrid RDM Training: I-JEPA regions (256d) + MoCo CLS (256d)
# Quick test run on a30/standby: 4 nodes x 2 GPUs = 8 GPUs total
# ============================================================
# Regions:  h5_embeddings/region_emb_flat.h5    (I-JEPA patch tokens, 256d)
# Global:   h5_embeddings/moco_cls_flat.h5      (MoCo CLS through MLP head, 256d, z-score)
# Lookup:   h5_embeddings/moco_cls_lookup.json
#
# Key flag: --cls_from_external_h5  routes MoCo CLS as 'cls_emb'
#           (bypasses projection/z-score in ddpm.py since already normalised)
# ============================================================

PROJECT_ROOT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen"
SEG_RDM_ROOT="${PROJECT_ROOT}/scripts/SEG-RDM"
RDM_ROOT="${SEG_RDM_ROOT}/rdm"

RUN_DIR="${RDM_ROOT}/rdm_out_final/hybrid_moco_cls/4_nodes_a30/"

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

mkdir -p "${RDM_ROOT}/SLRUM_OUTPUT_FILES/RDM_hybrid_moco_cls"
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
  --config "${RDM_ROOT}/configs/unified_seg_rdm_hybrid.yaml"
  --input_size 256
  --blr 1.25e-7
  --min_lr 1e-6
  --cosine_lr
  --warmup_epochs 5
  --weight_decay 0.01
  --epochs 300
  --batch_size 128
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
