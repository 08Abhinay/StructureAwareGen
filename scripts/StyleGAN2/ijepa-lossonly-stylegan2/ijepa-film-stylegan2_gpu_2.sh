#!/bin/bash
#SBATCH -A standby
#SBATCH -J ijepa_ChestXray_rampGD_warmup_5.4_2gpu_sem_mix_0.7
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/SLRUM_OUTPUT_FILES/ijepa_ChestXray_rampGD_warmup_5.4_2gpu_sem_mix_0.7.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/SLRUM_OUTPUT_FILES/ijepa_ChestXray_rampGD_warmup_5.4_2gpu_sem_mix_0.7.err

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
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/train.py \
        --outdir=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/outputs/final/ijepa_ChestXray_rampGD_warmup_5.4_2gpu_sem_mix_0.7/training-runs \
        --data=/scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip \
        --gpus=2 \
        --cond=1 \
        --ijepa_checkpoint /scratch/gilbreth/abelde/Thesis/CNN-JEPA/artifacts/pretrain_lightly/ijepacnn_chestxray-h5/ijepa_chestxray-h5_resnet50_bs64/version_2/ijepa_backbone_momentum_only.pth \
        --ijepa_lambda 1.0 \
        --ijepa_image 256 \
        --ijepa_input_channel 3 \
        --extra_dim 2048 \
        --ijepa_warmup_kimg 5.4 \
        --sem_mixing_prob 0.7
        
        

  

        

   