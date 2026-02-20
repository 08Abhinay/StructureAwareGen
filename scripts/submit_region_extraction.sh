#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a30
#SBATCH -q standby
#SBATCH --job-name=region_emb_extract
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=80G
#SBATCH --time=04:00:00

#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/region_emb_extract.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/region_emb_extract.err

set -e
set -o pipefail

module load anaconda
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen

which python

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))

# ---------- CONFIGURE HERE ----------
# Choose backbone: ijepa_vit_h14 | dinov2_vitl14 | dino_vitb8
BACKBONE=${BACKBONE:-"ijepa_vit_h14"}

# Output directory is named after backbone
case $BACKBONE in
    ijepa_vit_h14)
        OUTPUT_DIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/sam_cache_ijepa"
        EXTRA_ARGS="--ijepa_checkpoint /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/pretrained_enc_ckpts/ijepa/IN1K-vit.h.14-300e.pth.tar"
        ;;
    dinov2_vitl14)
        OUTPUT_DIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/sam_cache_dinov2"
        EXTRA_ARGS=""
        ;;
    dino_vitb8)
        OUTPUT_DIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/sam_cache_dino"
        EXTRA_ARGS=""
        ;;
    *)
        echo "Unknown backbone: $BACKBONE"
        exit 1
        ;;
esac
# -------------------------------------

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "=== Region Embedding Extraction ==="
echo "Backbone:    $BACKBONE"
echo "Output:      $OUTPUT_DIR"
echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $SLURM_NNODES, GPUs/node: $SLURM_GPUS_PER_NODE"
echo "===================================="

srun torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/precompute_region_embeddings.py \
      --image_dir /scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/train \
      --output_dir "$OUTPUT_DIR" \
      --sam_checkpoint /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/checkpoints/sam_vit_b_01ec64.pth \
      --backbone "$BACKBONE" \
      $EXTRA_ARGS \
      --subset_fraction 0.40 \
      --seed 42 \
      --skip_existing \
      --points_per_side 32 \
      --crop_n_layers 0 \
      --max_keep 100 \
      --pred_iou_thresh 0.82 \
      --dedup_iou_thresh 0.50

echo "Region embedding extraction finished!"
echo "Backbone: $BACKBONE -> $OUTPUT_DIR"
