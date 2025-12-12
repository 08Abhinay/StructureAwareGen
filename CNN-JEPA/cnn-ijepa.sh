#!/bin/bash
#SBATCH -A pfw-cs-k
#SBATCH -J req_files
#SBATCH -o /scratch/gilbreth/abelde/Thesis/CNN-JEPA/SLURM_OUTPUT_FILES/ijepa-Brain_cancer_labelled_256.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/CNN-JEPA/SLURM_OUTPUT_FILES/ijepa-Brain_cancer_labelled_256.err

#SBATCH --nodes=1
#SBATCH --ntasks=1            
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=03-00:00:00


source /scratch/gilbreth/abelde/Thesis/CNN-JEPA/jepa-cnn/bin/activate
cd /scratch/gilbreth/abelde/Thesis/CNN-JEPA

python -m pretrain.train_ijepacnn --config-name ijepacnn_imagenet100.yaml

