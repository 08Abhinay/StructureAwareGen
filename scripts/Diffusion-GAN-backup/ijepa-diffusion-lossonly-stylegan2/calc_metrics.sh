#!/bin/bash
#SBATCH -A debug
#SBATCH -J prlossonly-cond-brain
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/ijepa-diffusion-lossonly-stylegan2/SLURM_OUTPUT_FILES/prlossonly-cond-brain.err
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/ijepa-diffusion-lossonly-stylegan2/SLURM_OUTPUT_FILES/prlossonly-cond-brain.out

#SBATCH --nodes=1
#SBATCH --ntasks=1            
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
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
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/ijepa-diffusion-lossonly-stylegan2/calc_metrics_pr.py --metrics=pr50k3_full_cond \
                                                   --network=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/ijepa-diffusion-lossonly-stylegan2/outputs/final/ijepa_diffgan_Brain_labelled_256/training-runs/00001-Brain_cancer_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \
                                                   --data=/scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

   