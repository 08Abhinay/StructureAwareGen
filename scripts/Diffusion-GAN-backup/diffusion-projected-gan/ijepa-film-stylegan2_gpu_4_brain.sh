#!/bin/bash
#SBATCH -A standby
#SBATCH -J Brain_projbasic
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/SLURM_OUTPUT_FILES/fastgan/Brain_256_4gpu.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/SLURM_OUTPUT_FILES/fastgan/Brain_256_4gpu.err

#SBATCH --nodes=1
#SBATCH --ntasks=1            
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=04:00:00



export PROJECT=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch
export TMPDIR=$PROJECT/tmp/torch_tmp
export APPTAINER_TMPDIR=$PROJECT/tmp/apptainer_tmp
export APPTAINER_CACHEDIR=$PROJECT/tmp/apptainer_cache
mkdir -p "$TMPDIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
cd "$PROJECT"

srun --gres=gpu:4 \
  apptainer exec --nv --cleanenv \
      -B $PROJECT:$PROJECT \
      $PROJECT/SIF_FILES/stylegan2ada-devel.sif \
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/train.py \
        --outdir=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/outputs/final/fastgan/Brain_labelled_256_4gpu/training-runs \
        --data=/scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip \
        --gpus=4 \
        --cond=1 \
        --cfg fastgan \
        --batch=64 \
        --resume /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/outputs/final/fastgan/Brain_labelled_256_4gpu/training-runs/00000-fastgan-Brain_cancer_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl

    