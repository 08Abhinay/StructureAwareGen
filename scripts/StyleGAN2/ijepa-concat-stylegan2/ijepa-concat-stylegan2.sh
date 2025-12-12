#!/bin/bash
#SBATCH -A standby
#SBATCH -J ijepa_concat_sg2_ChestXray
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-concat-stylegan2/SLRUM_OUTPUT_FILES/ijepa_gan_ChestXray_labelled_256.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-concat-stylegan2/SLRUM_OUTPUT_FILES/ijepa_gan_ChestXray_labelled_256.err

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
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-concat-stylegan2/train.py \
        --outdir=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-concat-stylegan2/outputs/final/ijepa_gan_ChestXray_labelled_256/training-runs \
        --data=/scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip \
        --gpus=4 \
        --cond=1 \
        --ijepa_checkpoint /scratch/gilbreth/abelde/Thesis/ijepa-main/logs/vith14-256-vitsmall-chest_xray_Pnem_Normal/jepa-chest_xray_Pnem_Normal-256-vitsmall-latest.pth.tar \
        --ijepa_lambda 2.0 \
        --ijepa_image 256 \
        --ijepa_input_channel 1 \
        --ijepa_dim 384 \
        --resume /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-concat-stylegan2/outputs/final/ijepa_gan_ChestXray_labelled_256/training-runs/00000-chest_xray_labelled-cond-auto4/network-snapshot-001411.pkl
        

        

   