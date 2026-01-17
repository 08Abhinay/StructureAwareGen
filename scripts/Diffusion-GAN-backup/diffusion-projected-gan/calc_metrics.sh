#!/bin/bash
#SBATCH -A standby
#SBATCH -J Baseline_ProjGAN_Brain_metric_calc
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/SLURM_OUTPUT_FILES/Baseline_Brain_metric_calc.err
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/SLURM_OUTPUT_FILES/Baseline_Brain_metric_calc.out

#SBATCH --nodes=1
#SBATCH --ntasks=1            
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00



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
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/calc_metrics.py --metrics=pr50k3_full,is50k,kid50k_full \
                                                   --network=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/outputs/final/fastgan/Brain_labelled_256_4gpu/training-runs/00000-fastgan-Brain_cancer_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \
                                                   --data=/scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

   