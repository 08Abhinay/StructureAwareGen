#!/bin/bash
#SBATCH -A standby
#SBATCH -J SD-Brain_labelled-requeued 
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/SLURM_OUTPUT_FILES/Brain_labelled_SGDiff.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/SLURM_OUTPUT_FILES/Brain_labelled_SGDiff.err

#SBATCH --nodes=1
#SBATCH --ntasks=1            # ONE task on that node
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00


export PROJECT=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch
export TMPDIR=$PROJECT/tmp/torch_tmp
export APPTAINER_TMPDIR=$PROJECT/tmp/apptainer_tmp
export APPTAINER_CACHEDIR=$PROJECT/tmp/apptainer_cache
mkdir -p "$TMPDIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
cd "$PROJECT"

# Run the job (no host LD_PRELOAD thanks to --cleanenv)
# srun --gres=gpu:2 \
#   apptainer exec --nv --cleanenv \
#       -B $PROJECT:$PROJECT \
#       $PROJECT/SIF_FILES/stylegan2ada-devel.sif \
#       /opt/conda/bin/python train.py \
#         --outdir=$PROJECT/outputs/outputs_cifar_v4/training-runs \
#         --data=$PROJECT/datasets/cifar10.zip \
#         --gpus=2
srun --gres=gpu:4 \
  apptainer exec --nv --cleanenv \
      -B $PROJECT:$PROJECT \
      $PROJECT/SIF_FILES/stylegan2ada-devel.sif \
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/train.py \
        --outdir=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/outputs/final/ChestXray_labelled_256/training-runs \
        --data=/scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip \
        --gpus=4 \
        --cond=1 \
        --resume /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/outputs/final/ChestXray_labelled_256/training-runs/00000-chest_xray_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05/network-snapshot.pkl
        

   