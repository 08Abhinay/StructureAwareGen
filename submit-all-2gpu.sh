#!/bin/bash
# submit_extended_results.sh
# ---------------------------------------------------------------------------
# Launch all (model × dataset × semMix × fusionAlpha) ramp experiments
#   – Chest :  ijepa_warmup_kimg = 5.4
#   – Brain :  ijepa_warmup_kimg = 6.4
#   – Resume : value read from RESUME matrix (or omitted)
# ---------------------------------------------------------------------------
# Safety settings
set -euo pipefail

# -------------------------- CONSTANT PATHS ---------------------------------
declare -A ROOT TAG TRAIN

ROOT[SG2]=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2
TAG[SG2]=ijepa_SG2ADA
TRAIN[SG2]=${ROOT[SG2]}/train.py

ROOT[DSG2]=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2
TAG[DSG2]=ijepa_DSG2
TRAIN[DSG2]=${ROOT[DSG2]}/train.py

ROOT[DProjFG]=/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan
TAG[DProjFG]=ijepa_DProjFG
TRAIN[DProjFG]=${ROOT[DProjFG]}/train.py

# -------------------------- IJEPA CHECKPOINTS ------------------------------
IJ_CHEST=/scratch/gilbreth/abelde/Thesis/CNN-JEPA/artifacts/pretrain_lightly/ijepacnn_chestxray-h5/ijepa_chestxray-h5_resnet50_bs64/version_2/ijepa_backbone_momentum_only.pth
IJ_BRAIN=/scratch/gilbreth/abelde/Thesis/CNN-JEPA/artifacts/pretrain_lightly/ijepacnn_Brain_cancer-h5/ijepa_Brain_cancer-h5_resnet50_bs64/version_2/Brain_c_ijepa_backbone_momentum_only.pth

# -------------------------- DATASETS ---------------------------------------
ZIP_CHEST=/scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip
ZIP_BRAIN=/scratch/gilbreth/abelde/Thesis/dataset/256/Brain_cancer_labelled.zip

# -------------------------- SINGULARITY ------------------------------------
PROJECT=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/stylegan2-ada-pytorch
SIF=${PROJECT}/SIF_FILES/stylegan2ada-devel.sif

# -------------------------- SWEEP VALUES -----------------------------------
SEM_MIX_LIST=(0.5 0.9)
ALPHA_LIST=(0.5 0.6)
DATASETS=(Chest Brain)

# -------------------------- RESUME MATRIX -----------------------------------
# Key pattern  :  <MODEL>_<DATASET>_sm<SEM>_fa<ALPHA>
#                (decimals kept, eg sm0.5)
# Value        :  "",  noresume,  auto4,  /abs/path/to/snapshot.pkl
declare -A RESUME

##### Chest – SG2 ############################################################
RESUME[SG2_Chest_sm0.5_fa0.5]="/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/extended-results-2gpu/Chest/sem_0.5/alpha_0.5/training-runs/00000-chest_xray_labelled-cond-auto2/network-snapshot-000400.pkl"
RESUME[SG2_Chest_sm0.5_fa0.6]="/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/extended-results-2gpu/Chest/sem_0.5/alpha_0.6/training-runs/00001-chest_xray_labelled-cond-auto2/network-snapshot-000400.pkl"
RESUME[SG2_Chest_sm0.9_fa0.5]="/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/extended-results-2gpu/Chest/sem_0.9/alpha_0.5/training-runs/00001-chest_xray_labelled-cond-auto2/network-snapshot-000800.pkl"
RESUME[SG2_Chest_sm0.9_fa0.6]="noresume"

##### Chest – DSG2 ###########################################################
RESUME[DSG2_Chest_sm0.5_fa0.5]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/extended-results-2gpu/Chest/sem_0.5/alpha_0.5/training-runs/00000-chest_xray_labelled-cond-mirror-auto2-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05/network-snapshot.pkl"
RESUME[DSG2_Chest_sm0.5_fa0.6]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/extended-results-2gpu/Chest/sem_0.5/alpha_0.6/training-runs/00001-chest_xray_labelled-cond-mirror-auto2-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/network-snapshot.pkl"
RESUME[DSG2_Chest_sm0.9_fa0.5]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/extended-results-2gpu/Chest/sem_0.9/alpha_0.5/training-runs/00001-chest_xray_labelled-cond-mirror-auto2-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/network-snapshot.pkl"
RESUME[DSG2_Chest_sm0.9_fa0.6]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/extended-results-2gpu/Chest/sem_0.9/alpha_0.6/training-runs/00001-chest_xray_labelled-cond-mirror-auto2-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/network-snapshot.pkl"

##### Chest – DProjFG ########################################################
RESUME[DProjFG_Chest_sm0.5_fa0.5]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/extended-results-2gpu/Chest/sem_0.5/alpha_0.5/training-runs/00000-fastgan-chest_xray_labelled-gpus2-batch32-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/network-snapshot.pkl"
RESUME[DProjFG_Chest_sm0.5_fa0.6]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/extended-results-2gpu/Chest/sem_0.5/alpha_0.6/training-runs/00000-fastgan-chest_xray_labelled-gpus2-batch32-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/network-snapshot.pkl"
RESUME[DProjFG_Chest_sm0.9_fa0.5]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/extended-results-2gpu/Chest/sem_0.9/alpha_0.5/training-runs/00000-fastgan-chest_xray_labelled-gpus2-batch32-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/network-snapshot.pkl"
RESUME[DProjFG_Chest_sm0.9_fa0.6]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/extended-results-2gpu/Chest/sem_0.9/alpha_0.6/training-runs/00000-fastgan-chest_xray_labelled-gpus2-batch32-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/network-snapshot.pkl"

##### Brain – SG2 ############################################################
RESUME[SG2_Brain_sm0.5_fa0.5]="/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/extended-results-2gpu/Brain/sem_0.5/alpha_0.5/training-runs/00000-Brain_cancer_labelled-cond-auto2/network-snapshot-000800.pkl"
RESUME[SG2_Brain_sm0.5_fa0.6]="/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/extended-results-2gpu/Brain/sem_0.5/alpha_0.6/training-runs/00001-Brain_cancer_labelled-cond-auto2/network-snapshot-000800.pkl"
RESUME[SG2_Brain_sm0.9_fa0.5]="/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/extended-results-2gpu/Brain/sem_0.9/alpha_0.5/training-runs/00000-Brain_cancer_labelled-cond-auto2/network-snapshot-000800.pkl"
RESUME[SG2_Brain_sm0.9_fa0.6]="/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-ramp-stylegan2/extended-results-2gpu/Brain/sem_0.9/alpha_0.6/training-runs/00000-Brain_cancer_labelled-cond-auto2/network-snapshot-000800.pkl"

RESUME[DSG2_Brain_sm0.5_fa0.5]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/extended-results-2gpu/Brain/sem_0.5/alpha_0.5/training-runs/00000-Brain_cancer_labelled-cond-mirror-auto2-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05/network-snapshot.pkl"
RESUME[DSG2_Brain_sm0.5_fa0.6]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/extended-results-2gpu/Brain/sem_0.5/alpha_0.6/training-runs/00001-Brain_cancer_labelled-cond-mirror-auto2-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05-resumecustom/network-snapshot.pkl"
RESUME[DSG2_Brain_sm0.9_fa0.5]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/extended-results-2gpu/Brain/sem_0.9/alpha_0.5/training-runs/00000-Brain_cancer_labelled-cond-mirror-auto2-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05/network-snapshot.pkl"
RESUME[DSG2_Brain_sm0.9_fa0.6]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-ramp-stylegan2/extended-results-2gpu/Chest/sem_0.9/alpha_0.6/training-runs/00000-chest_xray_labelled-cond-mirror-auto2-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05/network-snapshot.pkl"
##### Brain – DProjFG ########################################################
RESUME[DProjFG_Brain_sm0.5_fa0.5]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/extended-results-2gpu/Brain/sem_0.5/alpha_0.5/training-runs/00000-fastgan-Brain_cancer_labelled-gpus2-batch32-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/network-snapshot.pkl"
RESUME[DProjFG_Brain_sm0.5_fa0.6]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/extended-results-2gpu/Brain/sem_0.5/alpha_0.6/training-runs/00000-fastgan-Brain_cancer_labelled-gpus2-batch32-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/network-snapshot.pkl"
RESUME[DProjFG_Brain_sm0.9_fa0.5]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/extended-results-2gpu/Brain/sem_0.9/alpha_0.5/training-runs/00000-fastgan-Brain_cancer_labelled-gpus2-batch32-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/network-snapshot.pkl"
RESUME[DProjFG_Brain_sm0.9_fa0.6]="/scratch/gilbreth/abelde/Thesis/scripts/Diffusion-GAN/diffusion-projected-ramp-gan/extended-results-2gpu/Brain/sem_0.9/alpha_0.6/training-runs/00000-fastgan-Brain_cancer_labelled-gpus2-batch32-d_pos-first-noise_sd-0.5-target0.6-ada_kimg100/network-snapshot.pkl"

# ----------------------------------------------------------------------------
#                          SUBMIT EACH COMBINATION
# ----------------------------------------------------------------------------
for KEY in DProjFG SG2 DSG2; do
  ROOT_DIR=${ROOT[$KEY]}
  TRAIN_PY=${TRAIN[$KEY]}
  MODEL_TAG=${TAG[$KEY]}

  mkdir -p "${ROOT_DIR}/extended-results-2gpu/SLURM_OUTPUT_FILES"

  for DATA in "${DATASETS[@]}"; do
    if [[ $DATA == "Chest" ]]; then
      ZIP=$ZIP_CHEST;  IJ=$IJ_CHEST;  WARMUP=5.4
    else
      ZIP=$ZIP_BRAIN;  IJ=$IJ_BRAIN;  WARMUP=6.4
    fi

    for SEM in "${SEM_MIX_LIST[@]}"; do
      for ALPHA in "${ALPHA_LIST[@]}"; do

        # ---------- resume flag ------------
        RES_KEY="${KEY}_${DATA}_sm${SEM}_fa${ALPHA}"
        RES_VALUE="${RESUME[$RES_KEY]:-}"     # empty if key not present

        RESUME_OPT=""
        if [[ -n "$RES_VALUE" && "$RES_VALUE" != "noresume" ]]; then
          RESUME_OPT="--resume $RES_VALUE"
        fi
        # -----------------------------------

        # ---------- cfg flag ------------
        if [[ $KEY == "DProjFG" ]]; then
          CFG_OPT="--cfg fastgan"
        else
          CFG_OPT="--cfg auto"
        fi
        # --------------------------------

        # ---------- batch flag ------------
        if [[ $KEY == "DProjFG" ]]; then
          BATCH_OPT="--batch 32"
        else
          BATCH_OPT=""
        fi
        # --------------------------------

        JOBNAME=2gpu_${MODEL_TAG}_${DATA}_sem${SEM}_alpha${ALPHA}
        SAVE_DIR=${ROOT_DIR}/extended-results-2gpu/${DATA}/sem_${SEM}/alpha_${ALPHA}/training-runs

        sbatch --job-name="$JOBNAME" <<EOF
#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a100-40gb
#SBATCH -q standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4    # request 2 GPUs per node:contentReference[oaicite:2]{index=2}
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=${ROOT_DIR}/extended-results-2gpu/SLURM_OUTPUT_FILES/${JOBNAME}.out
#SBATCH --error=${ROOT_DIR}/extended-results-2gpu/SLURM_OUTPUT_FILES/${JOBNAME}.err

export PROJECT=${PROJECT}
export TMPDIR=\$PROJECT/tmp/torch_tmp
export APPTAINER_TMPDIR=\$PROJECT/tmp/apptainer_tmp
export APPTAINER_CACHEDIR=\$PROJECT/tmp/apptainer_cache
mkdir -p "\$TMPDIR" "\$APPTAINER_TMPDIR" "\$APPTAINER_CACHEDIR"
cd \$PROJECT

srun \
  apptainer exec --nv --cleanenv \
    -B \$PROJECT:\$PROJECT \
    ${SIF} \
    /opt/conda/bin/python ${TRAIN_PY} \
      --outdir=${SAVE_DIR} \
      --data=${ZIP} \
      --gpus=4 \
      --cond=1 \
      --ijepa_checkpoint ${IJ} \
      --ijepa_lambda 1.0 \
      --ijepa_image 256 \
      --ijepa_input_channel 3 \
      --extra_dim 2048 \
      --ijepa_warmup_kimg ${WARMUP} \
      --sem_mixing_prob ${SEM} \
      --fusion_alpha ${ALPHA} \
      ${CFG_OPT} \
      ${BATCH_OPT} \
      ${RESUME_OPT}
EOF

      done
    done
  done
done
