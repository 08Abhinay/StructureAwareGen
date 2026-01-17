#!/bin/bash
#SBATCH -A pfw-cs-k
#SBATCH -J ijepa-diff-stylegan-ramp_Brain_metric_calc
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/SLRUM_OUTPUT_FILES/ijepa-diff-stylegan-ramp_Brain_metric_calc.err
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/SLRUM_OUTPUT_FILES/ijepa-diff-stylegan-ramp_Brain_metric_calc.out

#SBATCH --nodes=1
#SBATCH --ntasks=1            
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=04:00:00



export PROJECT=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch
export TMPDIR=$PROJECT/tmp/torch_tmp
export APPTAINER_TMPDIR=$PROJECT/tmp/apptainer_tmp
export APPTAINER_CACHEDIR=$PROJECT/tmp/apptainer_cache
mkdir -p "$TMPDIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
cd "$PROJECT"

srun --gres=gpu:1 \
  apptainer exec --nv --cleanenv \
      -B $PROJECT:$PROJECT \
      $PROJECT/SIF_FILES/stylegan2ada-devel.sif \
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/calc_metrics.py --metrics=is50k,kid50k_full \
                                                   --network=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/extended-results/Brain/sem_0.5/alpha_0.5/training-runs/00001-Brain_cancer_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \
                                                   --data=/scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

srun --gres=gpu:1 \
  apptainer exec --nv --cleanenv \
      -B $PROJECT:$PROJECT \
      $PROJECT/SIF_FILES/stylegan2ada-devel.sif \
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/calc_metrics_pr.py --metrics=pr50k3_full_cond \
                                                   --network=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/extended-results/Brain/sem_0.5/alpha_0.5/training-runs/00001-Brain_cancer_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \
                                                   --data=/scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

   