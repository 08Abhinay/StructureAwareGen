#!/bin/bash
# submit_all_metric_jobs.sh  -----------------------------------------------
# Fire off one SLURM batch *per model family* (StyleGAN2-baseline, Diff-SG2,
# Proj-FastGAN, …).  Each block below is the body you were previously
# running serially — now wrapped in its own sbatch submission.
# --------------------------------------------------------------------------
set -euo pipefail

# --------------------------------------------------------------------------
# Common environment & paths (unchanged)
# --------------------------------------------------------------------------
PROJECT=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch
SIF=${PROJECT}/SIF_FILES/stylegan2ada-devel.sif
export TMPDIR=$PROJECT/tmp/torch_tmp
export APPTAINER_TMPDIR=$PROJECT/tmp/apptainer_tmp
export APPTAINER_CACHEDIR=$PROJECT/tmp/apptainer_cache
mkdir -p "$TMPDIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" \
        "$PROJECT/SLURM_OUTPUT_FILES"

# --------------------------------------------------------------------------
# Helper: submit a job whose SBATCH body is provided via heredoc.
# Args: $1 = job-tag / short name  (used in –J, –o, –e)
# --------------------------------------------------------------------------
submit_job () {
  local JOBTAG="$1"; shift
  sbatch <<EOF
#!/bin/bash
#SBATCH -A standby
#SBATCH -J ${JOBTAG}
#SBATCH -e ${PROJECT}/SLURM_OUTPUT_FILES/${JOBTAG}.err
#SBATCH -o ${PROJECT}/SLURM_OUTPUT_FILES/${JOBTAG}.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00

export PROJECT=${PROJECT}
export TMPDIR=\$PROJECT/tmp/torch_tmp
export APPTAINER_TMPDIR=\$PROJECT/tmp/apptainer_tmp
export APPTAINER_CACHEDIR=\$PROJECT/tmp/apptainer_cache
cd \$PROJECT

$*
EOF
}

############################################################################
# 1) StyleGAN2 baseline metrics
############################################################################
submit_job "SG2_base_metrics" "
#StyleGAN2 Baseline --------------------------------------------------------
# Chest
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/outputs/final/chest_xray_labelled_256_4gpu/training-runs/00004-chest_xray_labelled-cond-auto4-resumecustom/network-snapshot-001612.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/outputs/final/chest_xray_labelled_256_4gpu/training-runs/00004-chest_xray_labelled-cond-auto4-resumecustom/network-snapshot-001612.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

# Brain
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/outputs/final/Brain_cancer_labelled_256/training-runs/00001-Brain_cancer_labelled-cond-auto4-resumecustom/network-snapshot-000201.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch/outputs/final/Brain_cancer_labelled_256/training-runs/00001-Brain_cancer_labelled-cond-auto4-resumecustom/network-snapshot-000201.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip
"

############################################################################
# 2) Diffusion-StyleGAN2 baseline metrics
############################################################################
submit_job "DiffSG2_base_metrics" "
#Diffusion StyleGAN2 Baseline ---------------------------------------------
# Chest
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/outputs/final/ChestXray_labelled_256/training-runs/00001-chest_xray_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/outputs/final/ChestXray_labelled_256/training-runs/00001-chest_xray_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

# Brain
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/outputs/final/Brain_cancer_labelled_256/training-runs/00001-Brain_cancer_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/outputs/final/Brain_cancer_labelled_256/training-runs/00001-Brain_cancer_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip
"

############################################################################
# 3) Projected-FastGAN baseline metrics
############################################################################
submit_job "ProjFG_base_metrics" "
#Projected FastGAN Baseline -----------------------------------------------
# Chest
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/outputs/final/fastgan/chest_xray_labelled_256_4gpu/training-runs/00000-fastgan-chest_xray_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/outputs/final/fastgan/chest_xray_labelled_256_4gpu/training-runs/00000-fastgan-chest_xray_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

# Brain
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/outputs/final/fastgan/Brain_labelled_256_4gpu/training-runs/00000-fastgan-Brain_cancer_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/outputs/final/fastgan/Brain_labelled_256_4gpu/training-runs/00000-fastgan-Brain_cancer_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip
"

############################################################################
# 4) StyleGAN2 LOSS-only metrics
############################################################################
submit_job "SG2_loss_metrics" "
#StyleGAN2 loss-only -------------------------------------------------------
# Chest
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-lossonly-stylegan2/outputs/Chest/sem_mixing_prob_0.5/ijepa_Chest_rampGD_warmup_5.4_4gpu_sem_mix_0.5/training-runs/00009-chest_xray_labelled-cond-auto4-resumecustom/network-snapshot-000201.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-lossonly-stylegan2/outputs/Chest/sem_mixing_prob_0.5/ijepa_Chest_rampGD_warmup_5.4_4gpu_sem_mix_0.5/training-runs/00009-chest_xray_labelled-cond-auto4-resumecustom/network-snapshot-000201.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

# Brain
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-stylegan2/outputs/final/ijepa_gan_Brain_cancer_labelled_256/training-runs/00001-Brain_cancer_labelled-cond-auto4-resumecustom/network-snapshot-000806.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-stylegan2/outputs/final/ijepa_gan_Brain_cancer_labelled_256/training-runs/00001-Brain_cancer_labelled-cond-auto4-resumecustom/network-snapshot-000806.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip
"

############################################################################
# 5) Diffusion-StyleGAN2 LOSS-only metrics
############################################################################
submit_job "DiffSG2_loss_metrics" "
#Diffusion StyleGAN2 loss-only --------------------------------------------
# Chest
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/ijepa-diffusion-lossonly-stylegan2/outputs/final/ijepa_diffgan_ChestXray_labelled_256/training-runs/00001-chest_xray_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/ijepa-diffusion-lossonly-stylegan2/outputs/final/ijepa_diffgan_ChestXray_labelled_256/training-runs/00001-chest_xray_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

# Brain
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/ijepa-diffusion-lossonly-stylegan2/outputs/final/ijepa_diffgan_Brain_labelled_256/training-runs/00001-Brain_cancer_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-stylegan2/calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/ijepa-diffusion-lossonly-stylegan2/outputs/final/ijepa_diffgan_Brain_labelled_256/training-runs/00001-Brain_cancer_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip
"

############################################################################
# 6) Projected-FastGAN LOSS-only metrics
############################################################################
submit_job "ProjFG_loss_metrics" "
#Projected FastGAN loss-only ----------------------------------------------
# Chest
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/outputs/final/fastgan/chest_xray_labelled_256_4gpu_sem_mix_0.6/training-runs/00000-fastgan-chest_xray_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/outputs/final/fastgan/chest_xray_labelled_256_4gpu_sem_mix_0.6/training-runs/00000-fastgan-chest_xray_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

# Brain
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/outputs/final/fastgan/Brain_C_labelled_256_4gpu/training-runs/00000-fastgan-Brain_cancer_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-gan/calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-onlyloss-gan/outputs/final/fastgan/Brain_C_labelled_256_4gpu/training-runs/00000-fastgan-Brain_cancer_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip
"

############################################################################
# 7) StyleGAN2 RAMP metrics
############################################################################
submit_job "SG2_ramp_metrics" "
#StyleGAN2 RAMP ------------------------------------------------------------
# Chest
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/outputs/Chest/sem_mixing_prob_0.9/ijepa_Chest_rampGD_warmup_5.4_4gpu_sem_mix_0.9_FusionAlpha_0.2/training-runs/00011-chest_xray_labelled-cond-auto4-resumecustom/network-snapshot-001612.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/outputs/Chest/sem_mixing_prob_0.9/ijepa_Chest_rampGD_warmup_5.4_4gpu_sem_mix_0.9_FusionAlpha_0.2/training-runs/00011-chest_xray_labelled-cond-auto4-resumecustom/network-snapshot-001612.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

# Brain
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/outputs/Brain/sem_mixing_prob_0.9/ijepa_Brain_C_rampGD_warmup_6.4_4gpu_sem_mix_0.9_gamma_FusionAlpha_0.2/training-runs/00001-Brain_cancer_labelled-cond-auto4-resumecustom/network-snapshot-000403.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/outputs/Brain/sem_mixing_prob_0.9/ijepa_Brain_C_rampGD_warmup_6.4_4gpu_sem_mix_0.9_gamma_FusionAlpha_0.2/training-runs/00001-Brain_cancer_labelled-cond-auto4-resumecustom/network-snapshot-000403.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip
"

############################################################################
# 8) Diffusion-StyleGAN2 RAMP metrics
############################################################################
submit_job "DiffSG2_ramp_metrics" "
#Diffusion StyleGAN2 RAMP --------------------------------------------------
# Chest
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/outputs/final/Diffusion_GAN_ijepa_ChestXray_rampGD_warmup_5.4_4gpu_sem_mix_0.9_lr_mul_0.01_gamma_auto_persample_lambda_1/training-runs/00001-chest_xray_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/outputs/final/Diffusion_GAN_ijepa_ChestXray_rampGD_warmup_5.4_4gpu_sem_mix_0.9_lr_mul_0.01_gamma_auto_persample_lambda_1/training-runs/00001-chest_xray_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

# Brain
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/outputs/final/ijepa_Brain_C_rampGD_warmup_6.4_4gpu_sem_mix_0.9_gamma/training-runs/00001-Brain_cancer_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/outputs/final/ijepa_Brain_C_rampGD_warmup_6.4_4gpu_sem_mix_0.9_gamma/training-runs/00001-Brain_cancer_labelled-cond-mirror-auto4-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip
"

############################################################################
# 9) Projected-FastGAN RAMP metrics
############################################################################
submit_job "ProjFG_ramp_metrics" "
#Projected FastGAN RAMP ----------------------------------------------------
# Chest
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/outputs/final/fastgan/chest_xray_labelled_256_4gpu_sem_mix_0.6/training-runs/00000-fastgan-chest_xray_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/outputs/final/fastgan/chest_xray_labelled_256_4gpu_sem_mix_0.6/training-runs/00000-fastgan-chest_xray_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip

# Brain
srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/calc_metrics_pr.py --metrics pr50k3_full_cond \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/outputs/final/fastgan/Brain_labelled_256_4gpu_sem_mix_0.6/training-runs/00000-fastgan-Brain_cancer_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

srun --gres=gpu:2 apptainer exec --nv --cleanenv -B \$PROJECT:\$PROJECT ${SIF} \\
  /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/calc_metrics.py --metrics kid50k_full,is50k \\
  --network /scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/outputs/final/fastgan/Brain_labelled_256_4gpu_sem_mix_0.6/training-runs/00000-fastgan-Brain_cancer_labelled-gpus4-batch64-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/best_model.pkl \\
  --data     /scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip
"

echo "All metric-calculation jobs have been submitted."
