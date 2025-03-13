#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=node_mask
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=48:00:00
#SBATCH --output=node_mask_%A.out

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
RUN_ID="node_mask_${TIMESTAMP}"

# Set parameters from the JSON input
NUM_GRAPHS=5000
SEED=42
NUM_LAYERS=6
NUM_HEADS=8
HIDDEN_DIM=512
DROPOUT=0.01
LEARNING_RATE=0.002
BATCH_SIZE=64
EPOCHS=100
MASK_MODE="node_feature"

# Create master directory for this run
MASTER_DIR="./results/${RUN_ID}"
mkdir -p "$MASTER_DIR"
mkdir -p "${MASTER_DIR}/models"

echo "Starting progressive mask probability training with optimized parameters..."
echo "Run ID: $RUN_ID"
echo "Master directory: $MASTER_DIR"

# Log start of experiment
echo "Configuration:"
echo "Number of graphs: $NUM_GRAPHS"
echo "Batch size: $BATCH_SIZE"
echo "Epochs per mask level: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Hidden dim: $HIDDEN_DIM"
echo "Num layers: $NUM_LAYERS"
echo "Num heads: $NUM_HEADS"
echo "Dropout: $DROPOUT"
echo "Seed: $SEED"
echo "Mask mode: $MASK_MODE"

# Array of mask probabilities to try
MASK_PROBS=(0.2 0.4 0.6 0.8)
PRETRAINED_MODEL=""

# Train models with progressive masking
for MASK_PROB in "${MASK_PROBS[@]}"; do
    # Format mask probability for directory name (replace . with _)
    MASK_PROB_DIR="${MASK_PROB/./}"

    # Set run name
    RUN_NAME="${MASK_MODE}_mp${MASK_PROB_DIR}"
    EXP_DIR="${MASTER_DIR}/${RUN_NAME}"

    echo "==================================================="
    echo "Starting training with mask probability: $MASK_PROB"
    echo "Run name: $RUN_NAME"
    echo "Using pretrained model: $PRETRAINED_MODEL"

    # Create directory for this run
    mkdir -p "$EXP_DIR"

    # Build command
    CMD="python main.py \
      --num_graphs $NUM_GRAPHS \
      --mask_prob $MASK_PROB \
      --mask_mode $MASK_MODE \
      --batch_size $BATCH_SIZE \
      --num_epochs $EPOCHS \
      --lr $LEARNING_RATE \
      --hidden_dim $HIDDEN_DIM \
      --num_layers $NUM_LAYERS \
      --num_heads $NUM_HEADS \
      --dropout $DROPOUT \
      --seed $SEED \
      "

    # Add pretrained model parameter if not first run
    if [ -n "$PRETRAINED_MODEL" ]; then
        CMD="$CMD --pretrained_model \"$PRETRAINED_MODEL\""
    fi

    # Execute the command
    echo "Executing: $CMD"
    eval "$CMD"

    # Update pretrained model path for next iteration
    PRETRAINED_MODEL="$./models/${RUN_NAME}_best.pt"

    echo "Completed training with mask probability: $MASK_PROB"
    echo "Best model saved at: $PRETRAINED_MODEL"
    echo "==================================================="
done


