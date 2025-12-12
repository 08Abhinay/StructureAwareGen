#!/bin/bash
#SBATCH -A debug
#SBATCH -J metrics
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/SLURM_OUTPUT_FILES/pr_cond_chest_metric_calc.err
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/SLURM_OUTPUT_FILES/pr-cond_chest_metric_calc.out

#SBATCH --nodes=1
#SBATCH --ntasks=1            
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=00:30:00



export PROJECT=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch
export TMPDIR=$PROJECT/tmp/torch_tmp
export APPTAINER_TMPDIR=$PROJECT/tmp/apptainer_tmp
export APPTAINER_CACHEDIR=$PROJECT/tmp/apptainer_cache
mkdir -p "$TMPDIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
cd "$PROJECT"

srun --gres=gpu:2 \
  apptainer exec --nv --cleanenv \
      -B $PROJECT:$PROJECT \
      $PROJECT/SIF_FILES/stylegan2ada-devel.sif \
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/calc_metrics_pr.py --metrics=pr50k3_full_cond \
                                                   --network=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/extended-results/Brain/sem_0.5/alpha_0.6/training-runs/00001-Brain_cancer_labelled-cond-auto4-resumecustom/network-snapshot-000403.pkl \
                                                   --data=/scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

srun --gres=gpu:2 \
  apptainer exec --nv --cleanenv \
      -B $PROJECT:$PROJECT \
      $PROJECT/SIF_FILES/stylegan2ada-devel.sif \
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/calc_metrics.py --metrics=kid50k_full,is50k \
                                                   --network=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/extended-results/Brain/sem_0.5/alpha_0.6/training-runs/00001-Brain_cancer_labelled-cond-auto4-resumecustom/network-snapshot-000403.pkl \
                                                   --data=/scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip