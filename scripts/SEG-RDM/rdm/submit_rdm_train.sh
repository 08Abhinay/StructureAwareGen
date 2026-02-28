#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p training
#SBATCH -q training
#SBATCH --job-name=RDM-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=80G
#SBATCH --time=24:00:00
#SBATCH --constraint=J

#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/RDM_a100-training-localemb-rerun/%x-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/RDM_a100-training-localemb-rerun/%x-%j.err


PROJECT_ROOT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen"
SEG_RDM_ROOT="${PROJECT_ROOT}/scripts/SEG-RDM"
RDM_ROOT="${SEG_RDM_ROOT}/rdm"

# Change this one path only when switching experiment output location.
RUN_DIR="${RDM_ROOT}/rdm_out_final/IJEPA_local_feat/a100-rerun/2_nodes/"

module load anaconda
conda activate "${PROJECT_ROOT}/SegmentationAwareGen"

# Prevent CPU oversubscription stalls that can trigger NCCL collective timeouts.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

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
  --nproc_per_node=4
  --node_rank="$SLURM_PROCID"
  --rdzv_backend=c10d
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT"
  --rdzv_id="$SLURM_JOB_ID"
  -m rdm.main_rdm
  --config "${RDM_ROOT}/configs/unified_seg_rdm_region_ijepa.yaml"
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
  --ijepa_h5_path "${PROJECT_ROOT}/h5_embeddings/ijepa_emb_flat.h5"
  --ijepa_lookup_json "${PROJECT_ROOT}/h5_embeddings/ijepa_lookup.json"
  --max_segments 100
)

if [ ${#RESUME_ARGS[@]} -gt 0 ]; then
  TORCHRUN_CMD+=("${RESUME_ARGS[@]}")
fi

# One torchrun per node (srun launches $SLURM_NNODES tasks total, 1 per node)
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
  "${TORCHRUN_CMD[@]}"
