#!/bin/bash
#SBATCH -A standby
#SBATCH -J lossonly_SG_Chest_warmup_5.4_4gpu_sem_mixing_0.5
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-lossonly-stylegan2/SLRUM_OUTPUT_FILES/ijepa_SG_lossonly_Brain_c_warmup_6.4_4gpu_sem_mixing_0.9.err
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-lossonly-stylegan2/SLRUM_OUTPUT_FILES/ijepa_SG_lossonly_Brain_c_warmup_6.4_4gpu_sem_mixing_0.9.out

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


SEM_MIX=0.5
RESUME_PATH=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/outputs/Chest/sem_mixing_prob_0.5/ijepa_Chest_rampGD_warmup_5.4_4gpu_sem_mix_0.5_FusionAlpha_0.2/training-runs/00001-chest_xray_labelled-cond-auto4-resumecustom/network-snapshot-001814.pkl

srun --gres=gpu:4 \
  apptainer exec --nv --cleanenv \
      -B $PROJECT:$PROJECT \
      $PROJECT/SIF_FILES/stylegan2ada-devel.sif \
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-lossonly-stylegan2/train.py \
        --outdir=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-lossonly-stylegan2/outputs/Chest/sem_mixing_prob_${SEM_MIX}/ijepa_Chest_rampGD_warmup_5.4_4gpu_sem_mix_${SEM_MIX}/training-runs \
        --data=/scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip \
        --gpus=4 \
        --cond=1 \
        --ijepa_checkpoint /scratch/gilbreth/abelde/Thesis/CNN-JEPA/artifacts/pretrain_lightly/ijepacnn_chestxray-h5/ijepa_chestxray-h5_resnet50_bs64/version_2/ijepa_backbone_momentum_only.pth \
        --ijepa_lambda 1.0 \
        --ijepa_image 256 \
        --ijepa_input_channel 3 \
        --ijepa_warmup_kimg 5.4 \
        --resume /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-lossonly-stylegan2/outputs/Chest/sem_mixing_prob_0.5/ijepa_Chest_rampGD_warmup_5.4_4gpu_sem_mix_0.5/training-runs/00010-chest_xray_labelled-cond-auto4-resumecustom/network-snapshot-0016.pkl
        

   