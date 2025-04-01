#!/bin/bash
#SBATCH --job-name=aig-transformer
#SBATCH --partition=gpu_a100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

cd ..
# Create log directory if it doesn't exist
mkdir -p slurm_logs

module purge
module load 2024
module load Anaconda3/2024.06-1

# Activate your environment
source activate aig-ml

# Print environment info
echo "Job started at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Run the script with SLURM-specific parameters
python cluster_run.py \
    --num_nodes=$SLURM_JOB_NUM_NODES \
    --devices=$SLURM_GPUS_PER_NODE \
    --strategy=ddp \
    --precision=16 \
    --batch_size=32 \
    --accumulate_grad_batches=4 \
    --num_epochs=100 \
    --learning_rate=1e-3 \
    --weight_decay=1e-5 \
    --num_workers=$SLURM_CPUS_PER_TASK \
    --pin_memory \
    --output_dir="logs/aig_transformer_${SLURM_JOB_ID}" \
    --data_path="/path/to/your/complete_tt_graphs.pkl" \
    --seed=42

# Print job completion message
echo "Job finished at $(date)"