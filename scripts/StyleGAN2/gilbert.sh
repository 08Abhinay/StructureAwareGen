#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a30
#SBATCH -q standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1    
#SBATCH --cpus-per-task=5
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/seg-aware-stylegan2/SLRUM_OUTPUT_FILES/imagenet_debug_dataset.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/seg-aware-stylegan2/SLRUM_OUTPUT_FILES/imagenet_debug_dataset.err

set -e
set -o pipefail

module load anaconda
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen
cd /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/seg-aware-stylegan2

# pip3 install -r requirements.txt

# Run the training script
# python3 train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1 --dry-run
# python3 dataset_tool.py --source=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/ChestXray/images \
#                         --dest=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/ChestXray.zip

# python dataset_tool.py --source=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/Brain_cancer/Training \
#                         --dest=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/256/Brain_cancer_labelled.zip \
#                         --width=256 --height=256 --resize-filter=box

# python dataset_tool.py --source=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/Lung_cancer/Train_cases \
#                         --dest=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/256/Lung_cancer_labelled.zip \
#                         --width=256 --height=256 --resize-filter=box

# python dataset_tool.py --source=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/chest_xray_Pnem_Normal/train \
#                         --dest=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/256/chest_xray_labelled.zip \
#                         --width=256 --height=256 --resize-filter=box

# python dataset_tool.py --source=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/Scenary \
#                         --dest=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/256/Scenary_labelled.zip \
#                         --width=256 --height=256 --resize-filter=box

# python dataset_tool.py --source=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/lsun_1000 \
#                         --dest=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/256/lsun_unlabelled.zip \
#                         --width=256 --height=256 --resize-filter=box  

# python dataset_tool.py --source=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/val2017 \
#                         --dest=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/256/EyePACS_AIROGS_labelled.zip \
#                         --width=256 --height=256 --resize-filter=box


# On HPC, create a debug subset with 20 images
cd /scratch/gilbreth/abelde/Thesis/StructureAwareGen

# Create a small subset directory
mkdir -p dataset/imagenet_debug_subset
cd dataset/imagenet_debug_subset

# Copy just 20 images from any class
cp ../imagenet-1K-hf/train/0/*.JPEG . 2>/dev/null | head -20

# Go back and create the StyleGAN2 dataset
cd /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/StyleGAN2/seg-aware-stylegan2


python dataset_tool.py \
    --source=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet_debug_subset \
    --dest=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/256/imagenet_debug_subset.zip \
    --width=256 --height=256 --resize-filter=box