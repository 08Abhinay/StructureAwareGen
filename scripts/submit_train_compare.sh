#!/bin/bash
# ===================================================================
# Comparison SLURM scripts for training SEG-RDM with different
# embedding sources: SAM (baseline) vs I-JEPA vs DINOv2.
#
# Usage:
#   sbatch scripts/submit_train_compare.sh sam
#   sbatch scripts/submit_train_compare.sh ijepa
#   sbatch scripts/submit_train_compare.sh dinov2
#
# Or run all three:
#   for src in sam ijepa dinov2; do sbatch scripts/submit_train_compare.sh $src; done
# ===================================================================

#SBATCH -A pfw-cs
#SBATCH -p a30
#SBATCH -q standby
#SBATCH --job-name=seg_rdm_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=80G
#SBATCH --time=04:00:00

#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/train_%x_%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/train_%x_%j.err

set -e
set -o pipefail

module load anaconda
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen

which python

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))

# ---------- PARSE EMBEDDING SOURCE ----------
EMB_SOURCE=${1:-"ijepa"}

BASE=/scratch/gilbreth/abelde/Thesis/StructureAwareGen
RDM_DIR=$BASE/scripts/SEG-RDM

case $EMB_SOURCE in
    sam)
        CONFIG=$RDM_DIR/rdm/configs/unified_seg_rdm.yaml
        MASK_NPZ_DIR=$BASE/sam_cache_unified
        OUTPUT_DIR=$RDM_DIR/rdm/output_dir/train_sam_embs
        ;;
    ijepa)
        CONFIG=$RDM_DIR/rdm/configs/unified_seg_rdm_ijepa_embs.yaml
        MASK_NPZ_DIR=$BASE/sam_cache_ijepa
        OUTPUT_DIR=$RDM_DIR/rdm/output_dir/train_ijepa_embs
        ;;
    dinov2)
        CONFIG=$RDM_DIR/rdm/configs/unified_seg_rdm_ijepa_embs.yaml
        MASK_NPZ_DIR=$BASE/sam_cache_dinov2
        OUTPUT_DIR=$RDM_DIR/rdm/output_dir/train_dinov2_embs
        ;;
    dino)
        CONFIG=$RDM_DIR/rdm/configs/unified_seg_rdm_ijepa_embs.yaml
        MASK_NPZ_DIR=$BASE/sam_cache_dino
        OUTPUT_DIR=$RDM_DIR/rdm/output_dir/train_dino_embs
        ;;
    *)
        echo "Unknown emb source: $EMB_SOURCE. Choose from: sam, ijepa, dinov2, dino"
        exit 1
        ;;
esac

mkdir -p "$OUTPUT_DIR"

# ---------- COORDINATION ----------
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "=== SEG-RDM Training ==="
echo "Embedding source: $EMB_SOURCE"
echo "Config:           $CONFIG"
echo "Mask NPZ dir:     $MASK_NPZ_DIR"
echo "Output dir:       $OUTPUT_DIR"
echo "========================"

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    $RDM_DIR/rdm/main_rdm.py \
      --config "$CONFIG" \
      --data_path $BASE/dataset/imagenet-1K-hf \
      --output_dir "$OUTPUT_DIR" \
      --use_seg_dataset \
      --mask_npz_dir "$MASK_NPZ_DIR" \
      --emb_source "$EMB_SOURCE" \
      --max_segments 250 \
      --batch_size 32 \
      --epochs 400 \
      --blr 1e-4 \
      --cosine_lr \
      --warmup_epochs 5 \
      --num_workers 8

echo "Training finished: $EMB_SOURCE"
echo "Checkpoints in: $OUTPUT_DIR"
