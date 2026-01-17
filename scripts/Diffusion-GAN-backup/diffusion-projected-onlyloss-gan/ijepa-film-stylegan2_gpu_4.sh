#!/bin/bash
#SBATCH -A standby
#SBATCH -J Dproj-fastgan_lossonly_Chest_Xray_labelled_4gpu_0.6_sem_mix
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/SLURM_OUTPUT_FILES/fastgan/Chest_Xray_labelled_256_4gpusem_mix_0.6.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/SLURM_OUTPUT_FILES/fastgan/Chest_Xray_labelled_256_4gpusem_mix_0.6.err

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
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/train.py \
        --outdir=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/outputs/final/fastgan/chest_xray_labelled_256_4gpu_sem_mix_0.6/training-runs \
        --data=/scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip \
        --gpus=4 \
        --cond=1 \
        --cfg fastgan \
        --batch=64 \
        --ijepa_checkpoint /scratch/gilbreth/abelde/Thesis/CNN-JEPA/artifacts/pretrain_lightly/ijepacnn_chestxray-h5/ijepa_chestxray-h5_resnet50_bs64/version_2/ijepa_backbone_momentum_only.pth \
        --ijepa_lambda 1.0 \
        --ijepa_image 256 \
        --ijepa_input_channel 3 \
        --extra_dim 2048 \
        --ijepa_warmup_kimg 5.4 \
        --sem_mixing_prob 0.6 \
        --resume /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/outputs/final/fastgan/chest_xray_labelled_256_4gpu_sem_mix_0.6/training-runs/00000-fastgan-chest_xray_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/network-snapshot.pkl
        
        


    