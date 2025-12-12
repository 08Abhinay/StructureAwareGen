#!/bin/bash
#SBATCH -A debug
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=80G

# Activate your virtual environment
source /scratch/gilbreth/abelde/NLP_Research/venv/bin/activate


cd /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch

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

python dataset_tool.py --source=/scratch/gilbreth/abelde/Thesis/dataset/EyePACS_AIROGS/release-crop/train \
                        --dest=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/datasets/256/EyePACS_AIROGS_labelled.zip \
                        --width=256 --height=256 --resize-filter=box   