#!/bin/bash
#SBATCH -A standby
#SBATCH -J ijepa_Brain_rampGD_warmup_5.4_4gpu_sem_mix_0.9_gamma
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/SLRUM_OUTPUT_FILES/ijepa_Brain_rampGD_warmup_5.4_4gpu_sem_mix_0.9_gamma.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/SLRUM_OUTPUT_FILES/ijepa_Brain_rampGD_warmup_5.4_4gpu_sem_mix_0.9_gamma.err

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
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/train.py \
        --outdir=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/outputs/final/ijepa_Brain_C_rampGD_warmup_6.4_4gpu_sem_mix_0.9_gamma/training-runs \
        --data=/scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip \
        --gpus=4 \
        --cond=1 \
        --ijepa_checkpoint /scratch/gilbreth/abelde/Thesis/CNN-JEPA/artifacts/pretrain_lightly/ijepacnn_Brain_cancer-h5/ijepa_Brain_cancer-h5_resnet50_bs64/version_2/Brain_c_ijepa_backbone_momentum_only.pth \
        --ijepa_lambda 1.0 \
        --ijepa_image 256 \
        --ijepa_input_channel 3 \
        --extra_dim 2048 \
        --ijepa_warmup_kimg 6.4 \
        --sem_mixing_prob 0.9 \
        --resume /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/outputs/final/ijepa_Brain_C_rampGD_warmup_6.4_4gpu_sem_mix_0.9_gamma/training-runs/00001-Brain_cancer_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/network-snapshot.pkl

   
        

   