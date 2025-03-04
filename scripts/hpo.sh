#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=node_hpo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=hpo_run_%A.out

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

# Set common parameters
NUM_GRAPHS=750
NUM_TRIALS=300
HPO_EPOCHS=25
MASK_PROB=0.4
SEED=42

# Create timestamp for unique study name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STUDY_NAME="hpo_node_mp${MASK_PROB/./}_${TIMESTAMP}"

echo "Starting node masking hyperparameter optimization..."
echo "Study name: $STUDY_NAME"
echo "Number of trials: $NUM_TRIALS"
echo "HPO epochs per trial: $HPO_EPOCHS"
echo "Mask probability: $MASK_PROB"
echo "Number of graphs: $NUM_GRAPHS"

# Run HPO with node masking
python hpo.py \
  --num_graphs "$NUM_GRAPHS" \
  --mask_prob "$MASK_PROB" \
  --n_trials "$NUM_TRIALS" \
  --hpo_epochs "$HPO_EPOCHS" \
  --study_name "$STUDY_NAME" \
  --seed "$SEED" \
  --optimize_batch_size

echo "HPO completed for node masking!"
echo "Check results in: ./hpo_results/$STUDY_NAME/"