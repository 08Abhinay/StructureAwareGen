#!/bin/bash
set -euo pipefail

# ============================================================
# MoCo v3 Region + CLS Embedding Extraction → H5 Shards
# ============================================================
# Single SLURM job: 2 nodes × 4 GPUs = 8 ranks (via torchrun).
# Matches the proven pattern from submit_region_extraction.sh:
#   ntasks-per-node=1, srun torchrun --nproc_per_node=4
# Each rank processes 1/8 of ImageNet, writes its own H5 shard.
# After extraction, a merge job combines shards into one file.
#
# Approach: write job scripts to files, then sbatch them.
# This avoids heredoc expansion issues with long paths.
# ============================================================

# SLURM Configuration
ACCOUNT="pfw-cs"
PART="training"
QOS="training"

# Paths
IMAGE_DIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/train"
OUTPUT_DIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/region_moco_shards"
MERGED_H5="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_embeddings/region_moco_flat.h5"
SAM_CHECKPOINT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/checkpoints/sam_vit_b_01ec64.pth"
EXTRACT_SCRIPT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/precompute_region_embeddings_h5.py"
MERGE_SCRIPT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/merge_h5_shards.py"

# MoCo v3 backbone
BACKBONE="mocov3_vit_large"
MOCO_CHECKPOINT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/pretrained_enc_ckpts/mocov3/vitl.pth.tar"

LOGDIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/moco_extraction"

# SAM AMG Parameters
MODEL_TYPE="vit_b"
POINTS_PER_SIDE=32
PRED_IOU_THRESH=0.82
STABILITY_SCORE_THRESH=0.85
MIN_MASK_REGION_AREA=300
BOX_NMS_THRESH=0.70
CROP_N_LAYERS=0
CROP_OVERLAP_RATIO=0.35
CROP_N_POINTS_DOWNSCALE=2

# Region parameters
MAX_KEEP=100
DEDUP_IOU_THRESH=0.65
MIN_QUALITY_SCORE=0.75
MIN_AREA_FRAC=0.001
MAX_AREA_FRAC=0.85
MEAN_SUBTRACT=false

# Subset / runtime
SUBSET_FRACTION=0.40
SEED=42

MEAN_SUBTRACT_FLAG=""
if [[ "${MEAN_SUBTRACT}" == "true" ]]; then
  MEAN_SUBTRACT_FLAG="--mean_subtract"
fi

# Setup
mkdir -p "$LOGDIR" "$OUTPUT_DIR"

echo "============================================================"
echo "MoCo v3 Region + CLS Extraction"
echo "============================================================"
echo "Setup: 2 nodes × 4 GPUs = 8 ranks (torchrun)"
echo "Subset fraction: ${SUBSET_FRACTION}"
echo "Backbone: ${BACKBONE}"
echo "Output shards: ${OUTPUT_DIR}"
echo "Final merged: ${MERGED_H5}"
echo "Logs: ${LOGDIR}"
echo "============================================================"

# ---- Write extraction job script to file ----
JOBNAME="moco_extract_all"
EXTRACT_JOB_SCRIPT="${LOGDIR}/_extract_job.sh"

cat > "${EXTRACT_JOB_SCRIPT}" <<'__HEADER__'
#!/bin/bash
__HEADER__

# SBATCH directives (expand paths now)
cat >> "${EXTRACT_JOB_SCRIPT}" <<__SBATCH__
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PART}
#SBATCH -q ${QOS}
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=${LOGDIR}/${JOBNAME}-%j.out
#SBATCH --error=${LOGDIR}/${JOBNAME}-%j.err
#SBATCH --constraint=J
__SBATCH__

# Runtime environment (quoted heredoc — no expansion)
cat >> "${EXTRACT_JOB_SCRIPT}" <<'__ENV__'

# Load environment
module load anaconda
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 3))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "============================================================"
echo "MoCo v3 Region + CLS Extraction"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Date: $(date)"
echo "============================================================"
__ENV__

# The srun command — expand all config paths NOW, keep runtime vars literal
cat >> "${EXTRACT_JOB_SCRIPT}" <<__CMD__

srun torchrun \\
  --nnodes=2 \\
  --nproc_per_node=4 \\
  --rdzv_backend=c10d \\
  --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \\
  --rdzv_id=\${SLURM_JOB_ID} \\
  "${EXTRACT_SCRIPT}" \\
  --image_dir "${IMAGE_DIR}" \\
  --output_dir "${OUTPUT_DIR}" \\
  --sam_checkpoint "${SAM_CHECKPOINT}" \\
  --sam_model_type "${MODEL_TYPE}" \\
  --backbone "${BACKBONE}" \\
  --moco_checkpoint "${MOCO_CHECKPOINT}" \\
  --subset_fraction ${SUBSET_FRACTION} \\
  --seed ${SEED} \\
  --skip_existing \\
  --points_per_side ${POINTS_PER_SIDE} \\
  --pred_iou_thresh ${PRED_IOU_THRESH} \\
  --stability_score_thresh ${STABILITY_SCORE_THRESH} \\
  --box_nms_thresh ${BOX_NMS_THRESH} \\
  --crop_n_layers ${CROP_N_LAYERS} \\
  --crop_overlap_ratio ${CROP_OVERLAP_RATIO} \\
  --crop_n_points_downscale ${CROP_N_POINTS_DOWNSCALE} \\
  --min_mask_region_area ${MIN_MASK_REGION_AREA} \\
  --max_keep ${MAX_KEEP} \\
  --dedup_iou_thresh ${DEDUP_IOU_THRESH} \\
  --min_quality_score ${MIN_QUALITY_SCORE} \\
  --min_area_frac ${MIN_AREA_FRAC} \\
  --max_area_frac ${MAX_AREA_FRAC} ${MEAN_SUBTRACT_FLAG}

echo "============================================================"
echo "Extraction completed at \$(date)"
echo "============================================================"
__CMD__

chmod +x "${EXTRACT_JOB_SCRIPT}"
echo "Generated: ${EXTRACT_JOB_SCRIPT}"

# Verify the generated script looks correct
echo "--- Verifying srun line in generated script ---"
grep -n "srun torchrun\|EXTRACT_SCRIPT\|precompute" "${EXTRACT_JOB_SCRIPT}" || true

# Submit extraction job
EXTRACT_JOB_ID=$(sbatch --parsable --job-name="${JOBNAME}" "${EXTRACT_JOB_SCRIPT}")
echo "Extraction job submitted: ${EXTRACT_JOB_ID}"

# ---- Write merge job script to file ----
MERGE_JOBNAME="moco_merge_h5"
MERGE_JOB_SCRIPT="${LOGDIR}/_merge_job.sh"

cat > "${MERGE_JOB_SCRIPT}" <<__MERGE__
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PART}
#SBATCH -q ${QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=${LOGDIR}/${MERGE_JOBNAME}-%j.out
#SBATCH --error=${LOGDIR}/${MERGE_JOBNAME}-%j.err

module load anaconda
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen
export PYTHONNOUSERSITE=1

echo "============================================================"
echo "Merging H5 shards → ${MERGED_H5}"
echo "Date: \$(date)"
echo "============================================================"

python3 "${MERGE_SCRIPT}" \\
  --shard_dir "${OUTPUT_DIR}" \\
  --output "${MERGED_H5}" \\
  --verify

echo "============================================================"
echo "Merge completed at \$(date)"
echo "============================================================"
__MERGE__

chmod +x "${MERGE_JOB_SCRIPT}"
echo "Generated: ${MERGE_JOB_SCRIPT}"

# Submit merge job (depends on extraction)
MERGE_JOB_ID=$(sbatch --parsable --dependency=afterok:${EXTRACT_JOB_ID} --job-name="${MERGE_JOBNAME}" "${MERGE_JOB_SCRIPT}")
echo "Merge job submitted: ${MERGE_JOB_ID} (depends on ${EXTRACT_JOB_ID})"

echo ""
echo "============================================================"
echo "Jobs submitted!"
echo "============================================================"
echo "  Extraction: ${EXTRACT_JOB_ID} (2 nodes × 4 GPUs = 8 ranks)"
echo "  Merge: ${MERGE_JOB_ID} (runs after extraction)"
echo "Monitor: squeue -u \$USER"
echo "Logs: ${LOGDIR}"
echo "============================================================"
