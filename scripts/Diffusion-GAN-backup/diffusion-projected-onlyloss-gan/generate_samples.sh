#!/bin/bash
#SBATCH -A debug
#SBATCH -J ProjLossonly_ProjGAN_Chest_metric_calc
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/SLURM_OUTPUT_FILES/generate_samples.err
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/SLURM_OUTPUT_FILES/generate_samples.out

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
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/gen_images.py \
                            --outdir /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/outputs/Brain_samples_seed_1-30-class-3 \
                            --seeds=1-30 \
                            --class 0 \
                            --network=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/outputs/final/fastgan/Brain_C_labelled_256_4gpu/training-runs/00000-fastgan-Brain_cancer_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \

   