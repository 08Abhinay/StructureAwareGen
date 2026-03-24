#!/bin/bash
set -euo pipefail

# ============================================================
# SAM Embedding Extraction - Parallel Job Submission
# ============================================================
# This script submits parallel jobs for either:
#   1) legacy SAM encoder embeddings (precompute_sam_embeddings.py), or
#   2) region embeddings (precompute_region_embeddings.py).
# The dataset is split into deterministic index chunks.
# ============================================================

# SLURM Configuration
ACCOUNT="pfw-cs"
PART="a30"  # or "a100" if available
QOS="standby"

# Dataset splitting
CHUNK=100               # Number of parallel jobs
TOTAL_IMAGES=1281167    # ImageNet-1K train set size
CHUNK_SIZE=$((TOTAL_IMAGES / CHUNK))

# Extraction mode: "sam" (legacy) or "region" (ViT pooled region embeddings)
EXTRACT_MODE="region"

# Paths (UPDATE THESE!)
IMAGE_DIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1k-hf/train"
OUTPUT_DIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/region_emb_extract-dinov2-vitl14"
SAM_CHECKPOINT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/checkpoints/sam_vit_b_01ec64.pth"
SAM_SCRIPT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/precompute_sam_embeddings.py"
REGION_SCRIPT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/precompute_region_embeddings.py"
IJEPA_CHECKPOINT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/pretrained_enc_ckpts/ijepa/IN1K-vit.h.14-300e.pth.tar"
LOGDIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/sam_extraction"

# Shared SAM/AMG Parameters
MODEL_TYPE="vit_b"
POINTS_PER_SIDE=32
PRED_IOU_THRESH=0.82
STABILITY_SCORE_THRESH=0.85
MIN_MASK_REGION_AREA=300
BOX_NMS_THRESH=0.70
CROP_N_LAYERS=0
CROP_OVERLAP_RATIO=0.35
CROP_N_POINTS_DOWNSCALE=2

# Legacy SAM mode parameters
MAX_KEEP_SAM=250
DEDUP_IOU_THRESH_SAM=0.90

# Region mode parameters (parity defaults)
REGION_BACKBONE="dinov2_vitl14"
MAX_KEEP_REGION=100
DEDUP_IOU_THRESH_REGION=0.65
MIN_QUALITY_SCORE=0.75
MIN_AREA_FRAC=0.001
MAX_AREA_FRAC=0.85
MEAN_SUBTRACT=false

REGION_MEAN_SUBTRACT_FLAG=""
if [[ "${MEAN_SUBTRACT}" == "true" ]]; then
  REGION_MEAN_SUBTRACT_FLAG="--mean_subtract"
fi

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
echo "Mode: ${EXTRACT_MODE}"
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

if [[ "${EXTRACT_MODE}" == "region" ]]; then
  python3 ${REGION_SCRIPT} \\
    --image_dir "${IMAGE_DIR}" \\
    --output_dir "${OUTPUT_DIR}" \\
    --sam_checkpoint "${SAM_CHECKPOINT}" \\
    --sam_model_type "${MODEL_TYPE}" \\
    --backbone "${REGION_BACKBONE}" \\
    --ijepa_checkpoint "${IJEPA_CHECKPOINT}" \\
    --start_index ${start_idx} \\
    --end_index ${end_idx} \\
    --points_per_side ${POINTS_PER_SIDE} \\
    --pred_iou_thresh ${PRED_IOU_THRESH} \\
    --stability_score_thresh ${STABILITY_SCORE_THRESH} \\
    --box_nms_thresh ${BOX_NMS_THRESH} \\
    --crop_n_layers ${CROP_N_LAYERS} \\
    --crop_overlap_ratio ${CROP_OVERLAP_RATIO} \\
    --crop_n_points_downscale ${CROP_N_POINTS_DOWNSCALE} \\
    --min_mask_region_area ${MIN_MASK_REGION_AREA} \\
    --max_keep ${MAX_KEEP_REGION} \\
    --dedup_iou_thresh ${DEDUP_IOU_THRESH_REGION} \\
    --min_quality_score ${MIN_QUALITY_SCORE} \\
    --min_area_frac ${MIN_AREA_FRAC} \\
    --max_area_frac ${MAX_AREA_FRAC} \\
    --skip_existing ${REGION_MEAN_SUBTRACT_FLAG}
else
  python3 ${SAM_SCRIPT} \\
    --image_dir "${IMAGE_DIR}" \\
    --output_dir "${OUTPUT_DIR}" \\
    --checkpoint "${SAM_CHECKPOINT}" \\
    --model_type "${MODEL_TYPE}" \\
    --start_index ${start_idx} \\
    --end_index ${end_idx} \\
    --max_keep ${MAX_KEEP_SAM} \\
    --points_per_side ${POINTS_PER_SIDE} \\
    --pred_iou_thresh ${PRED_IOU_THRESH} \\
    --stability_score_thresh ${STABILITY_SCORE_THRESH} \\
    --min_mask_region_area ${MIN_MASK_REGION_AREA} \\
    --dedup_iou_thresh ${DEDUP_IOU_THRESH_SAM} \\
    --skip_existing \\
    --device cuda:0
fi

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
