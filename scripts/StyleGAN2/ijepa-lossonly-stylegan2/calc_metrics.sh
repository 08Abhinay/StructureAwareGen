#!/bin/bash
#SBATCH -A debug
#SBATCH -J pr_cond_chest_metric_calc
#SBATCH -e /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-lossonly-stylegan2/SLURM_OUTPUT_FILES/pr_cond_chest_metric_calc.err
#SBATCH -o /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-lossonly-stylegan2/SLURM_OUTPUT_FILES/pr-cond_chest_metric_calc.out

#SBATCH --nodes=1
#SBATCH --ntasks=1            
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=00:30:00



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
      /opt/conda/bin/python /scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-lossonly-stylegan2/calc_metrics_pr.py --metrics=pr50k3_full_cond \
                                                   --network=/scratch/gilbreth/abelde/Thesis/scripts/StyleGAN2/ijepa-lossonly-stylegan2/outputs/Chest/sem_mixing_prob_0.5/ijepa_Chest_rampGD_warmup_5.4_4gpu_sem_mix_0.5/training-runs/00009-chest_xray_labelled-cond-auto4-resumecustom/network-snapshot-001814.pkl \
                                                   --data=/scratch/gilbreth/abelde/Thesis/dataset/256/chest_xray_labelled.zip
   