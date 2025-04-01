#!/bin/bash
#SBATCH --job-name=aig-transformer
#SBATCH --partition=gpu_a100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/%j.out

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
srun python cluster_run.py \
    --batch_size=32 \
    --num_epochs=100 \
    --learning_rate=1e-3 \
    --weight_decay=1e-5 \
    --output_dir="logs/aig_transformer_${SLURM_JOB_ID}" \
    --data_path="complete_tt_graphs.pkl" \
    --seed=42

# Print job completion message
echo "Job finished at $(date)"