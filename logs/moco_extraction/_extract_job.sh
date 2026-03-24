#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p training
#SBATCH -q training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/moco_extraction/moco_extract_all-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/logs/moco_extraction/moco_extract_all-%j.err
#SBATCH --constraint=J

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

srun torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --rdzv_id=${SLURM_JOB_ID} \
  "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/precompute_region_embeddings_h5.py" \
  --image_dir "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/train" \
  --output_dir "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/region_moco_shards" \
  --sam_checkpoint "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/checkpoints/sam_vit_b_01ec64.pth" \
  --sam_model_type "vit_b" \
  --backbone "mocov3_vit_large" \
  --moco_checkpoint "/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/pretrained_enc_ckpts/mocov3/vitl.pth.tar" \
  --subset_fraction 0.40 \
  --seed 42 \
  --skip_existing \
  --points_per_side 32 \
  --pred_iou_thresh 0.82 \
  --stability_score_thresh 0.85 \
  --box_nms_thresh 0.70 \
  --crop_n_layers 0 \
  --crop_overlap_ratio 0.35 \
  --crop_n_points_downscale 2 \
  --min_mask_region_area 300 \
  --max_keep 100 \
  --dedup_iou_thresh 0.65 \
  --min_quality_score 0.75 \
  --min_area_frac 0.001 \
  --max_area_frac 0.85 

echo "============================================================"
echo "Extraction completed at $(date)"
echo "============================================================"
