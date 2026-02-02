#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p training
#SBATCH -q training
#SBATCH --job-name=Fetch_sam_emb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=80G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/sam_embeddings.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/sam_embeddings.err

set -e
set -o pipefail

module load anaconda
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen

which python

python /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/precompute_sam_embeddings.py \
     --image_dir /scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/train \
     --output_dir /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/output_dir/sam_embeddings \
     --checkpoint /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/segProto/checkpoints/sam_vit_b_01ec64.pth \
     --model_type vit_b \
     --max_keep 250 

echo "Finished!"
# echo "Finished. Going to the next script for verifying!"
#echo "Verifying data alignment now..."
# python scripts/verify_data_alignment.py \
#      --image_dir /scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/train \
#      --sam_npz_dir ... --ijepa_npz_dir ...





