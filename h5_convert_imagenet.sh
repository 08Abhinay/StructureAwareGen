#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a100-80gb
#SBATCH -q normal
#SBATCH --job-name=h5-imagenet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=80G
#SBATCH --time=24:00:00

#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/SLRUM_OUTPUT_FILES/h5_imagenet/%x-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/SLRUM_OUTPUT_FILES/h5_imagenet/%x-%j.err

PROJECT_ROOT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen"

module load anaconda
conda activate "${PROJECT_ROOT}/SegmentationAwareGen"

mkdir -p "${PROJECT_ROOT}/SLRUM_OUTPUT_FILES/h5_imagenet"

python "${PROJECT_ROOT}/h5_convert_imagenet.py" \
    --image_dir "${PROJECT_ROOT}/dataset/imagenet-1K-hf/train" \
    --dst_h5    "${PROJECT_ROOT}/h5_embeddings/imagenet_train_images.h5" \
    --verify
