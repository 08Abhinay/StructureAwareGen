#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p training
#SBATCH -q training
#SBATCH --job-name=h5-convert-ijepa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=80G
#SBATCH --time=24:00:00


#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/h5convert-ijepa/%x-%j.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/h5convert-ijepa/%x-%j.err


PROJECT_ROOT="/scratch/gilbreth/abelde/Thesis/StructureAwareGen"


module load anaconda
conda activate "${PROJECT_ROOT}/SegmentationAwareGen"

python /scratch/gilbreth/abelde/Thesis/StructureAwareGen/h5_convert_ijepa.py