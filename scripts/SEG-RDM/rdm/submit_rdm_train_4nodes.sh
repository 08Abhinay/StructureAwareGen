#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a30
#SBATCH -q standby
#SBATCH --job-name=ijepa_and_seg_aware
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=80G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/%x-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/%x-%j.err


PROJECT_ROOT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen"
SEG_RDM_ROOT="${PROJECT_ROOT}/scripts/SEG-RDM"
RDM_ROOT="${SEG_RDM_ROOT}/rdm"

# Change this one path only when switching experiment output location.
RUN_DIR="${RDM_ROOT}/rdm_out_final/2_nodes/a30/batch_64/ijepa_and_seg_aware"

module load anaconda
conda activate "${PROJECT_ROOT}/SegmentationAwareGen"

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))

cd "${SEG_RDM_ROOT}"
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir -p "${RDM_ROOT}/SLRUM_OUTPUT_FILES"
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
  --config "${RDM_ROOT}/configs/unified_seg_rdm.yaml"
  --input_size 256
  --blr 1e-6
  --weight_decay 0.01
  --epochs 200
  --batch_size 16
  --accum_iter 4
  --output_dir "${RUN_DIR}"
  --log_dir "${RUN_DIR}"
  --data_path "${PROJECT_ROOT}/dataset/imagenet-1K-hf"
  --use_seg_dataset
  --mask_npz_dir "${PROJECT_ROOT}/sam_cache_unified"
  --ijepa_cache_dir "${PROJECT_ROOT}/ijepa_embeddings"
  --max_segments 250
)

if [ ${#RESUME_ARGS[@]} -gt 0 ]; then
  TORCHRUN_CMD+=("${RESUME_ARGS[@]}")
fi

# One torchrun per node (srun launches 2 tasks total, 1 per node)
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
  "${TORCHRUN_CMD[@]}"
