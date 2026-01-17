#!/bin/bash
#SBATCH -A standby
#SBATCH -J ISD-Chest_labelled
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/ijepa-diffusion-lossonly-stylegan2/SLURM_OUTPUT_FILES/Chestlabelled_ijepa_ISGDiff.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/ijepa-diffusion-lossonly-stylegan2/SLURM_OUTPUT_FILES/Chestlabelled_ijepa_ISGDiff.err

#SBATCH --nodes=1
#SBATCH --ntasks=1            # ONE task on that node
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
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/ijepa-diffusion-lossonly-stylegan2/train.py \
        --outdir=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/ijepa-diffusion-lossonly-stylegan2/outputs/final/ijepa_diffgan_ChestXray_labelled_256/training-runs \
        --data=/scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip \
        --gpus=4 \
        --cond=1 \
        --ijepa_checkpoint /scratch/gilbreth/abelde/Thesis/CNN-JEPA/artifacts/pretrain_lightly/ijepacnn_chestxray-h5/ijepa_chestxray-h5_resnet50_bs64/version_2/ijepa_backbone_momentum_only.pth \
        --ijepa_lambda 1.0 \
        --ijepa_image 256 \
        --ijepa_input_channel 3 \
        --ijepa_warmup_kimg 5.4 \
        --resume /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/ijepa-diffusion-lossonly-stylegan2/outputs/final/ijepa_diffgan_ChestXray_labelled_256/training-runs/00001-chest_xray_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl

        
   