#!/bin/bash
#SBATCH -A standby
#SBATCH -J ijepa_ddp
#SBATCH --nodes=4
#SBATCH --gpus-per-node=2          # 2 DDP ranks per node
#SBATCH --ntasks-per-node=2        # one task == one DDP rank
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o /scratch/gilbreth/abelde/Thesis/ijepa/SLURM_OUTPUT_FILES/AIRROGS_vitsmall_res256.out
#SBATCH -e /scratch/gilbreth/abelde/Thesis/ijepa/SLURM_OUTPUT_FILES/AIRROGS_vitsmall_res256.err

module purge
module load cuda/12.2 cudnn
source /scratch/gilbreth/abelde/Thesis/ijepa-main/ijepa-env/bin/activate

# -------- rendezvous information ---------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500                  # pick any free port
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# -----------------------------------------------------------------

CONFIG=/scratch/gilbreth/abelde/Thesis/ijepa/configs/in1k_vith14_ep300_finetuning.yaml

torchrun \
  --nnodes          $SLURM_NNODES \
  --nproc_per_node  $SLURM_GPUS_PER_NODE \
  --node_rank       $SLURM_NODEID \
  --rdzv_id         "ijepa_${SLURM_JOB_ID}" \
  --rdzv_backend    c10d \
  --rdzv_endpoint   ${MASTER_ADDR}:${MASTER_PORT} \
  /scratch/gilbreth/abelde/Thesis/ijepa/main_finetuning.py \
  --fname $CONFIG
