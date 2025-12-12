#!/bin/bash
#SBATCH -A standby
#SBATCH -J ijepa_sg2_Brain_C_labelled
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-FiLM-stylegan2/SLRUM_OUTPUT_FILES/ijepa_gan_Brain_c_labelled_256.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-FiLM-stylegan2/SLRUM_OUTPUT_FILES/ijepa_gan_Brain_c_labelled_256.err

#SBATCH --nodes=1
#SBATCH --ntasks=1             
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
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
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-FiLM-stylegan2/train.py \
        --outdir=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-FiLM-stylegan2/outputs/final/ijepa_gan_Brain_c_labelled_256_temp/training-runs \
        --data=/scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip \
        --gpus=2 \
        --cond=1 \
        --ijepa_checkpoint /scratch/gilbreth/abelde/Thesis/ijepa-main/logs/vith14-256-vitsmall-Brain_cancer/jepa-Brain_cancer-res256-vitsmall-latest.pth.tar \
        --ijepa_lambda 1.0 \
        --ijepa_image 256 \
        --ijepa_input_channel 3 \
        --extra_dim 384 \
        --film_layers 0 \
        --ijepa_warmup_kimg 50

        

   