#!/bin/bash
#SBATCH -A pfw-cs-k
#SBATCH -J DDPM_Brain_c
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/DDPM/SLRUM_OUTPUT_FILES/Brain_cancer_labelled_v1.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/DDPM/SLRUM_OUTPUT_FILES/Brain_cancer_labelled_v1.err

#SBATCH --nodes=1
#SBATCH --ntasks=1            # ONE task on that node
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100-80GB  
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=01-00:00:00

source /scratch/gilbreth/abelde/Thesis/scripts/DDPM/venv/bin/activate
cd /scratch/gilbreth/abelde/Thesis/scripts/DDPM

python3 train_ddpm.py




