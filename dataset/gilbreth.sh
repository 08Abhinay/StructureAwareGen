#!/bin/bash
#SBATCH -A debug
#SBATCH -J download_and_train
#SBATCH -o /scratch/gilbreth/abelde/Thesis/dataset/SLURM_OUTPUT_FILES/download_lsuncat.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/dataset/SLURM_OUTPUT_FILES/download_lsuncat.err

#SBATCH --nodes=1
#SBATCH --ntasks=1            
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=00:30:00

# 1) Activate your Python environment
source /scratch/gilbreth/abelde/Thesis/CNN-JEPA/jepa-cnn/bin/activate

# 3) Download LSUN Cat into your dataset folder
cd /scratch/gilbreth/abelde/Thesis/dataset

python download_lsun.py \
  --input_dir /scratch/gilbreth/abelde/Thesis/dataset/lsun/cat \
  --train_dir /scratch/gilbreth/abelde/Thesis/dataset/lsun/cat_train \
  --test_dir  /scratch/gilbreth/abelde/Thesis/dataset/lsun/cat_test \
  --train_ratio 0.8 \
  --seed 42
