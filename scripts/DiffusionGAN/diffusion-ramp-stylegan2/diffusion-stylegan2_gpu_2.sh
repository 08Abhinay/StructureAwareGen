#!/bin/bash
#SBATCH -A debug
#SBATCH -J Diffusion_GAN_ijepa_ChestXray_rampGD_warmup_5.4_4gpu_sem_mix_0.9_lr_mul_0.01_gamma_auto_persample_lambda_1
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/SLURM_OUTPUT_FILES/Diffusion_GAN_ijepa_ChestXray_rampGD_warmup_5.4_2gpu_sem_mix_0.9_lr_mul_0.01_gamma_auto_persample_lambda_1.err
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/SLURM_OUTPUT_FILES/Diffusion_GAN_ijepa_ChestXray_rampGD_warmup_5.4_2gpu_sem_mix_0.9_lr_mul_0.01_gamma_auto_persample_lambda_1.out

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
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/train.py \
        --outdir=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/outputs/final/Diffusion_GAN_ijepa_ChestXray_rampGD_warmup_5.4_4gpu_sem_mix_0.9_lr_mul_0.01_gamma_auto_persample_lambda_1/training-runs \
        --data=/scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip \
        --gpus=2 \
        --cond=1 \
        --ijepa_checkpoint /scratch/gilbreth/abelde/Thesis/CNN-JEPA/artifacts/pretrain_lightly/ijepacnn_chestxray-h5/ijepa_chestxray-h5_resnet50_bs64/version_2/ijepa_backbone_momentum_only.pth \
        --ijepa_lambda 1.0 \
        --ijepa_image 256 \
        --ijepa_input_channel 3 \
        --extra_dim 2048 \
        --ijepa_warmup_kimg 5.4 \
        --sem_mixing_prob 0.9 \
        --resume /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/outputs/final/Diffusion_GAN_ijepa_ChestXray_rampGD_warmup_5.4_4gpu_sem_mix_0.9_lr_mul_0.01_gamma_auto_persample_lambda_1/training-runs/00000-chest_xray_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05/best_model.pkl
  

        

   