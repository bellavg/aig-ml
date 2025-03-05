#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=progressive_mask_optimized
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=48:00:00
#SBATCH --output=progressive_mask_optimized_%A.out

# Load required modules
module purge
module load 2024
module load Anaconda3/2024.06-1

# Activate your environment
source activate aig-ml

# Navigate to your project directory
cd ..

# Create a timestamp for unique run identification
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="progressive_mask_optimized_${TIMESTAMP}"

# Set parameters from the JSON input
NUM_GRAPHS=5000
SEED=42
NUM_LAYERS=4
NUM_HEADS=16
HIDDEN_DIM=256
DROPOUT=0.0
LEARNING_RATE=0.004
BATCH_SIZE=128
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
echo "Configuration:" > "${MASTER_DIR}/config.txt"
echo "Number of graphs: $NUM_GRAPHS" >> "${MASTER_DIR}/config.txt"
echo "Batch size: $BATCH_SIZE" >> "${MASTER_DIR}/config.txt"
echo "Epochs per mask level: $EPOCHS" >> "${MASTER_DIR}/config.txt"
echo "Learning rate: $LEARNING_RATE" >> "${MASTER_DIR}/config.txt"
echo "Hidden dim: $HIDDEN_DIM" >> "${MASTER_DIR}/config.txt"
echo "Num layers: $NUM_LAYERS" >> "${MASTER_DIR}/config.txt"
echo "Num heads: $NUM_HEADS" >> "${MASTER_DIR}/config.txt"
echo "Dropout: $DROPOUT" >> "${MASTER_DIR}/config.txt"
echo "Seed: $SEED" >> "${MASTER_DIR}/config.txt"
echo "Mask mode: $MASK_MODE" >> "${MASTER_DIR}/config.txt"

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
      --exp_name \"$RUN_NAME\" \
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
      --results_dir \"$MASTER_DIR/\" \
      --save_model_path \"$MASTER_DIR/models/\""

    # Add pretrained model parameter if not first run
    if [ -n "$PRETRAINED_MODEL" ]; then
        CMD="$CMD --pretrained_model \"$PRETRAINED_MODEL\""
    fi

    # Log the command
    echo "Command: $CMD" >> "${EXP_DIR}/command.txt"

    # Execute the command
    echo "Executing: $CMD"
    eval "$CMD"

    # Update pretrained model path for next iteration
    PRETRAINED_MODEL="${MASTER_DIR}/models/${RUN_NAME}_best.pt"

    echo "Completed training with mask probability: $MASK_PROB"
    echo "Best model saved at: $PRETRAINED_MODEL"
    echo "==================================================="
done

echo "Progressive mask training completed!"
echo "All results saved to: $MASTER_DIR"

# Summarize results
echo "Summary of results:" > "${MASTER_DIR}/summary.txt"
for MASK_PROB in "${MASK_PROBS[@]}"; do
    MASK_PROB_DIR="${MASK_PROB/./}"
    RUN_NAME="${MASK_MODE}_mp${MASK_PROB_DIR}"
    METRICS_FILE="${MASTER_DIR}/${RUN_NAME}/metrics.json"

    if [ -f "$METRICS_FILE" ]; then
        echo "Mask probability: $MASK_PROB" >> "${MASTER_DIR}/summary.txt"
        echo "Best val loss: $(grep -o '\"best_val_loss\": [0-9.]*' $METRICS_FILE | cut -d' ' -f2)" >> "${MASTER_DIR}/summary.txt"
        echo "Test losses: $(grep -o '\"test_losses\": {.*}' $METRICS_FILE)" >> "${MASTER_DIR}/summary.txt"
        echo "" >> "${MASTER_DIR}/summary.txt"
    else
        echo "No results found for mask probability: $MASK_PROB" >> "${MASTER_DIR}/summary.txt"
    fi
done

echo "Summary created at: ${MASTER_DIR}/summary.txt"