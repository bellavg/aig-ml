#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=edge_mask
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=48:00:00
#SBATCH --output=edge_mask_%A.out

# Load required modules
module purge
module load 2024
module load Anaconda3/2024.06-1

# Activate your environment
source activate aig-ml

# Navigate to your project directory
cd ..

# Create a timestamp for unique run identification
TIMESTAMP=$(date +"%Y%m%d_%H")
RUN_ID="edge_mask_${TIMESTAMP}"

# Set parameters from the JSON input
MASK_MODE="edge_feature"


echo "Starting progressive mask probability training with optimized parameters..."
echo "Run ID: $RUN_ID"
echo "Master directory: $MASTER_DIR"

# Log start of experiment
echo "Mask mode: $MASK_MODE"

# Array of mask probabilities to try
MASK_PROBS=(0.2 0.4 0.6 0.8)

# Train models with progressive masking
for MASK_PROB in "${MASK_PROBS[@]}"; do
    # Format mask probability for directory name (replace . with _)
    MASK_PROB_DIR="${MASK_PROB/./}"

    # Set run name
    RUN_NAME="${MASK_MODE}_mp${MASK_PROB_DIR}"

    echo "==================================================="
    echo "Starting training with mask probability: $MASK_PROB"
    echo "Run name: $RUN_NAME"

    # Build command
    CMD="srun python main.py \
      --mask_prob $MASK_PROB \
      --mask_mode $MASK_MODE \
      "

    # Execute the command
    echo "Executing: $CMD"
    eval "$CMD"


    echo "Completed edge feature training with mask probability: $MASK_PROB"
    echo "==================================================="
done


