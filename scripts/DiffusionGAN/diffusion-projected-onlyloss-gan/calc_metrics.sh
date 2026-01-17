#!/bin/bash
#SBATCH -A standby
#SBATCH -J ProjLossonly_ProjGAN_Chest_metric_calc
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/SLURM_OUTPUT_FILES/pr_Chest_metric_calc.err
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/SLURM_OUTPUT_FILES/pr_Chest_metric_calc.out

#SBATCH --nodes=1
#SBATCH --ntasks=1            
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
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
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/calc_metrics_pr.py --metrics=pr50k3_full_cond \
                                                   --network=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/outputs/final/fastgan/chest_xray_labelled_256_4gpu_sem_mix_0.6/training-runs/00000-fastgan-chest_xray_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \
                                                   --data=/scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

   