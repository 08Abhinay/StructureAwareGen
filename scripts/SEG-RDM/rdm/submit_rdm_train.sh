#!/bin/bash
set -euo pipefail

# =========================
# SLURM / cluster settings
# =========================
ACCOUNT=pfw-cs
PART=a100-40gb 
QOS=standby

JOBNAME="rdm_ijepa_h14_2g"
LOGDIR=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES
NODES=1
GPUS_PER_NODE=4

# =========================
# Environment / code paths
# =========================
# Update CONDA_ENV to your training environment.
CONDA_ENV=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen
REPO_ROOT=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM
CONFIG=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/configs/rdm_default.yaml

# =========================
# Output + data
# =========================
OUTPUT_DIR=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/rdm_out/ijepa_h14
DATA_PATH=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf

# =========================
# Training hyperparams
# =========================
INPUT_SIZE=256
EPOCHS=200
BATCH_SIZE=16
ACCUM_ITER=1
BLR=1e-6
WEIGHT_DECAY=0.01

mkdir -p "$LOGDIR" "$OUTPUT_DIR"

# =========================
# SLURM job submission
# =========================
sbatch --begin=now+0hours --job-name="$JOBNAME" <<EOF
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PART}
#SBATCH -q ${QOS}

#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=80G
#SBATCH --time=04:00:00
#SBATCH --output=${LOGDIR}/${JOBNAME}-%j.out
#SBATCH --error=${LOGDIR}/${JOBNAME}-%j.err

set -euo pipefail

module load anaconda
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}

MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_NODELIST}" | head -n 1)
MASTER_PORT=\$((29500 + SLURM_JOB_ID % 1000))
NNODES=\$SLURM_NNODES
GPUS_PER_NODE=\${SLURM_GPUS_ON_NODE:-${GPUS_PER_NODE}}

cd ${REPO_ROOT}
export PYTHONPATH="\$PWD:\$PYTHONPATH"

RESUME_ARG=""
if ls "${OUTPUT_DIR}"/checkpoint-*.pth 1> /dev/null 2>&1; then
  LAST_CKPT=\$(ls -t "${OUTPUT_DIR}"/checkpoint-*.pth | head -n 1)
  echo "Found checkpoint: \$LAST_CKPT"
  RESUME_ARG="--resume \$LAST_CKPT"
fi

srun --nodes=\$NNODES --ntasks=\$NNODES --ntasks-per-node=1 \
  torchrun \
  --nnodes=\$NNODES \
  --nproc_per_node=\$GPUS_PER_NODE \
  --rdzv_backend=c10d \
  --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \
  --rdzv_id=\$SLURM_JOB_ID \
  -m rdm.main_rdm \
    --config ${CONFIG} \
    --input_size ${INPUT_SIZE} \
    --blr ${BLR} \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --accum_iter ${ACCUM_ITER} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --data_path ${DATA_PATH} \
    \$RESUME_ARG
EOF

echo "Submitted: ${JOBNAME}"
echo "  Logs:   ${LOGDIR}/${JOBNAME}-<jobid>.out/.err"
echo "  Output: ${OUTPUT_DIR}"
