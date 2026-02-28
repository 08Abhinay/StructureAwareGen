#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a100-80gb
#SBATCH -q normal
#SBATCH --job-name=h5-convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=80G
#SBATCH --time=10:00:00


#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/h5convert/%x-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/h5convert/%x-%j.err


PROJECT_ROOT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen"


module load anaconda
conda activate "${PROJECT_ROOT}/SegmentationAwareGen"

python /scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_convert.py