#!/bin/bash
#SBATCH -A standby
#SBATCH -J ijepa_SG_Brain_C_rampGD_warmup_6.4_4gpu_sem_mixing_0.9_FusionAlpha_0.2
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/SLRUM_OUTPUT_FILES/ijepa_SG_Brain_C_rampGD_warmup_6.4_4gpu_sem_mixing_0.9_FusionAlpha_0.2.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/SLRUM_OUTPUT_FILES/ijepa_SG_Brain_C_rampGD_warmup_6.4_4gpu_sem_mixing_0.9_FusionAlpha_0.2.err

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

SEM_MIX=0.9
FUSION_ALPHA=0.2
RESUME_PATH=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/outputs/Brain/sem_mixing_prob_0.9/ijepa_Brain_C_rampGD_warmup_6.4_4gpu_sem_mix_0.9_gamma_FusionAlpha_0.2/training-runs/00001-Brain_cancer_labelled-cond-auto4-resumecustom/network-snapshot-001814.pkl

srun --gres=gpu:4 \
  apptainer exec --nv --cleanenv \
      -B $PROJECT:$PROJECT \
      $PROJECT/SIF_FILES/stylegan2ada-devel.sif \
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/train.py \
        --outdir=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/outputs/Brain/sem_mixing_prob_${SEM_MIX}/ijepa_Brain_C_rampGD_warmup_6.4_4gpu_sem_mix_${SEM_MIX}_FusionAlpha_${FUSION_ALPHA}/training-runs \
        --data=/scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip \
        --gpus=4 \
        --cond=1 \
        --ijepa_checkpoint /scratch/gilbreth/abelde/Thesis/CNN-JEPA/artifacts/pretrain_lightly/ijepacnn_Brain_cancer-h5/ijepa_Brain_cancer-h5_resnet50_bs64/version_2/Brain_c_ijepa_backbone_momentum_only.pth \
        --ijepa_lambda 1.0 \
        --ijepa_image 256 \
        --ijepa_input_channel 3 \
        --extra_dim 2048 \
        --ijepa_warmup_kimg 5.4 \
        --sem_mixing_prob $SEM_MIX \
        --fusion_alpha $FUSION_ALPHA \
        --resume "${RESUME_PATH}"

        