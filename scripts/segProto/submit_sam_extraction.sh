#!/bin/bash
set -euo pipefail

# ============================================================
# SAM Embedding Extraction - Parallel Job Submission
# ============================================================
# This script submits 100 parallel jobs to extract SAM embeddings
# from ImageNet-1K, splitting the dataset into chunks.
# ============================================================

# SLURM Configuration
ACCOUNT="pfw-cs"
PART="a30"  # or "a100" if available
QOS="standby"

# Dataset splitting
CHUNK=100               # Number of parallel jobs
TOTAL_IMAGES=1281167    # ImageNet-1K train set size
CHUNK_SIZE=$((TOTAL_IMAGES / CHUNK))

# Paths (UPDATE THESE!)
IMAGE_DIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1k-hf/train"
OUTPUT_DIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/sam_embeddings"
CHECKPOINT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/checkpoints/sam_vit_b_01ec64.pth"
SCRIPT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/precompute_sam_embeddings.py"
LOGDIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/sam_extraction"

# SAM Parameters
MODEL_TYPE="vit_b"
MAX_KEEP=250
POINTS_PER_SIDE=64
PRED_IOU_THRESH=0.80
STABILITY_SCORE_THRESH=0.85
MIN_MASK_REGION_AREA=300
DEDUP_IOU_THRESH=0.90

# Setup
mkdir -p "$LOGDIR" "$OUTPUT_DIR"

# Optional: throttle submissions
SLEEP_BETWEEN_SUBMITS=0

echo "============================================================"
echo "Submitting SAM extraction jobs"
echo "============================================================"
echo "Total images: ${TOTAL_IMAGES}"
echo "Number of jobs: ${CHUNK}"
echo "Chunk size: ~${CHUNK_SIZE} images/job"
echo "Output: ${OUTPUT_DIR}"
echo "Logs: ${LOGDIR}"
echo "============================================================"

for ((job_id=0; job_id<CHUNK; job_id++)); do
  start_idx=$((job_id * CHUNK_SIZE))
  end_idx=$(((job_id + 1) * CHUNK_SIZE))
  
  # Last job gets remainder
  if (( job_id == CHUNK - 1 )); then
    end_idx=$TOTAL_IMAGES
  fi

  JOBNAME="sam_extract_${job_id}_${CHUNK}"
  
  echo "Submitting job ${job_id}/${CHUNK}: images [${start_idx}:${end_idx}]"

  sbatch --begin=now+0hours --job-name="$JOBNAME" <<EOF
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PART}
#SBATCH -q ${QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=${LOGDIR}/${JOBNAME}-%j.out
#SBATCH --error=${LOGDIR}/${JOBNAME}-%j.err

# Load environment
module load anaconda
conda activate /scratch/gilbreth/abelde/conda_env

# Job info
echo "============================================================"
echo "Job: ${JOBNAME}"
echo "Node: \$(hostname)"
echo "Date: \$(date)"
echo "Processing images: [${start_idx}:${end_idx}]"
echo "============================================================"

# Run SAM extraction
python3 ${SCRIPT} \\
  --image_dir "${IMAGE_DIR}" \\
  --output_dir "${OUTPUT_DIR}" \\
  --checkpoint "${CHECKPOINT}" \\
  --model_type "${MODEL_TYPE}" \\
  --start_index ${start_idx} \\
  --end_index ${end_idx} \\
  --max_keep ${MAX_KEEP} \\
  --points_per_side ${POINTS_PER_SIDE} \\
  --pred_iou_thresh ${PRED_IOU_THRESH} \\
  --stability_score_thresh ${STABILITY_SCORE_THRESH} \\
  --min_mask_region_area ${MIN_MASK_REGION_AREA} \\
  --dedup_iou_thresh ${DEDUP_IOU_THRESH} \\
  --skip_existing \\
  --device cuda:0

echo "============================================================"
echo "Job ${JOBNAME} completed at \$(date)"
echo "============================================================"
EOF

  if (( SLEEP_BETWEEN_SUBMITS > 0 )); then
    sleep "${SLEEP_BETWEEN_SUBMITS}"
  fi
done

echo ""
echo "============================================================"
echo "✅ Submitted ${CHUNK} SAM extraction jobs!"
echo "============================================================"
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: ${LOGDIR}"
echo "Output will be in: ${OUTPUT_DIR}/masks_npz/"
echo "============================================================"
echo ""
echo "Expected completion: ~24 hours with 100 parallel jobs"
echo "Each job processes ~${CHUNK_SIZE} images at ~10s/image"
echo "============================================================"
