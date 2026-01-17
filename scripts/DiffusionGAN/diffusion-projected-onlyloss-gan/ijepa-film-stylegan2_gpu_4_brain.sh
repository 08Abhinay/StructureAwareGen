#!/bin/bash
#SBATCH -A standby
#SBATCH -J Brain_proj-lossonly
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/SLURM_OUTPUT_FILES/fastgan/Brain_proj-lossonly.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/SLURM_OUTPUT_FILES/fastgan/Brain_proj-lossonly.err

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
        --outdir=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/outputs/final/Brain_C_labelled_256_4gpu/training-runs \
        --data=/scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip \
        --gpus=4 \
        --cond=1 \
        --cfg fastgan \
        --batch=64 \
        --ijepa_checkpoint /scratch/gilbreth/abelde/Thesis/CNN-JEPA/artifacts/pretrain_lightly/ijepacnn_Brain_cancer-h5/ijepa_Brain_cancer-h5_resnet50_bs64/version_2/Brain_c_ijepa_backbone_momentum_only.pth \
        --ijepa_lambda 1.0 \
        --ijepa_image 256 \
        --ijepa_input_channel 3 \
        --extra_dim 2048 \
        --ijepa_warmup_kimg 6.4 \
        --sem_mixing_prob 0.6 \
        --resume /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/outputs/final/Brain_C_labelled_256_4gpu/training-runs/00000-fastgan-Brain_cancer_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/metric-fid50k_full.jsonl
        

    