#!/bin/bash
set -euo pipefail

# ============================================================
# MoCo v3 CLS-only Embedding Extraction → Flat H5
# ============================================================
# CLS token through MLP head (1024→256), z-score normalised.
# NO SAM needed — pure forward pass, very fast (~15 min on 4 GPUs).
#
# Uses torchrun for DDP: each rank writes a shard, rank 0 merges.
# ============================================================

ACCOUNT="pfw-cs"
PART="training"
QOS="training"

# Paths
BASE="/scratch/gilbreth/abelde/Thesis/StructureAwareGen"
IMAGE_DIR="${BASE}/dataset/imagenet-1K-hf/train"
OUTPUT_DIR="${BASE}/moco_cls_shards"
MERGED_H5="${BASE}/h5_embeddings/moco_cls_flat.h5"
MERGED_JSON="${BASE}/h5_embeddings/moco_cls_lookup.json"
MOCO_CKPT="${BASE}/scripts/SEG-RDM/rdm/pretrained_enc_ckpts/mocov3/vitl.pth.tar"
EXTRACT_SCRIPT="${BASE}/scripts/segProto/precompute_moco_cls_h5.py"
LOGDIR="${BASE}/logs/moco_cls_extraction"

mkdir -p "$LOGDIR" "$OUTPUT_DIR"

echo "============================================================"
echo "MoCo v3 CLS-only Extraction (through MLP head → 256d)"
echo "============================================================"
echo "Setup: 2 nodes × 4 GPUs = 8 GPUs total (torchrun multi-node)"
echo "Image dir: ${IMAGE_DIR}"
echo "Output: ${MERGED_H5}"
echo "Lookup: ${MERGED_JSON}"
echo "Logs: ${LOGDIR}"
echo "============================================================"

# ---- Write job script ----
JOB_SCRIPT="${LOGDIR}/_extract_cls_job.sh"

cat > "${JOB_SCRIPT}" << 'EOF'
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
EOF

# Add paths (expand variables now)
cat >> "${JOB_SCRIPT}" << __SBATCH__
#SBATCH --job-name=moco_cls_extract
#SBATCH --output=${LOGDIR}/moco_cls_extract_%j.out
#SBATCH --error=${LOGDIR}/moco_cls_extract_%j.err
__SBATCH__

# Runtime environment (no expansion)
cat >> "${JOB_SCRIPT}" << 'ENV'

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
ENV

# The torchrun command (expand paths now)
cat >> "${JOB_SCRIPT}" << __CMD__

srun --ntasks-per-node=1 torchrun \\
  --nproc_per_node=4 \\
  --nnodes=2 \\
  --rdzv_backend=c10d \\
  --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \\
  "${EXTRACT_SCRIPT}" \\
  --image_dir "${IMAGE_DIR}" \\
  --output_dir "${OUTPUT_DIR}" \\
  --moco_checkpoint "${MOCO_CKPT}" \\
  --merged_h5 "${MERGED_H5}" \\
  --merged_json "${MERGED_JSON}" \\
  --batch_size 256 \\
  --num_workers 8

echo "============================================================"
echo "Extraction + merge completed at \$(date)"
echo "============================================================"
__CMD__

chmod +x "${JOB_SCRIPT}"
echo "Generated: ${JOB_SCRIPT}"

# Submit
JOB_ID=$(sbatch --parsable "${JOB_SCRIPT}")
echo ""
echo "============================================================"
echo "Job submitted: ${JOB_ID}"
echo "Monitor: squeue -u \$USER"
echo "Output:  ${LOGDIR}/moco_cls_extract_${JOB_ID}.out"
echo "============================================================"
