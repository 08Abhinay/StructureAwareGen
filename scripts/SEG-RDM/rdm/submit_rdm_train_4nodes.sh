#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH -p a100-40gb
#SBATCH -q standby
#SBATCH --job-name=ijepa_and_seg_aware
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=80G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/ijepa_and_seg_aware-batch_32/.out
#SBATCH --error=/scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES/ijepa_and_seg_aware-batch_32/.err

module load anaconda
conda activate /scratch/gilbreth/abelde/Thesis/StructureAwareGen/SegmentationAwareGen

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))

cd /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir -p /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/SLRUM_OUTPUT_FILES
mkdir -p /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/rdm_out/4_nodes/batch_128/ijepa_h14

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

RESUME_ARG=""
if ls /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/rdm_out/4_nodes/batch_128/ijepa_h14/checkpoint-*.pth 1> /dev/null 2>&1; then
  LAST_CKPT=$(ls -t /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/rdm_out/4_nodes/batch_128/ijepa_h14/checkpoint-*.pth | head -n 1)
  RESUME_ARG="--resume $LAST_CKPT"
fi

# One torchrun per node (srun launches 4 tasks total, 1 per node)
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=2 \
    --node_rank=$SLURM_PROCID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    -m rdm.main_rdm \
      --config /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/configs/rdm_default.yaml \
      --input_size 256 \
      --blr 1e-6 \
      --weight_decay 0.01 \
      --epochs 200 \
      --batch_size 32 \
      --accum_iter 1 \
      --output_dir /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/rdm_out/4_nodes/batch_32/ijepa_and_seg_aware \
      --log_dir /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm \
      --data_path /scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf \
      --resume /scratch/gilbreth/abelde/Thesis/StructureAwareGen/scripts/SEG-RDM/rdm/rdm_out/4_nodes/batch_128/ijepa_h14/checkpoint-last.pth
