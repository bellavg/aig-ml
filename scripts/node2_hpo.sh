#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=node_exist_hpo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=16:00:00
#SBATCH --output=hpo_node_existence_%A.out

# Load required modules
module purge
module load 2024
module load Anaconda3/2024.06-1

# Activate your environment
source activate aig-ml

# Navigate to your project directory
cd ..

# Set common parameters
NUM_GRAPHS=750
NUM_TRIALS=300
HPO_EPOCHS=25
MASK_PROB=0.4
SEED=42

# Create timestamp for unique study name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STUDY_NAME="hpo_node_existence_mp${MASK_PROB/./}_${TIMESTAMP}"

echo "Starting node existence masking hyperparameter optimization..."
echo "Study name: $STUDY_NAME"
echo "Number of trials: $NUM_TRIALS"
echo "HPO epochs per trial: $HPO_EPOCHS"
echo "Mask probability: $MASK_PROB"
echo "Number of graphs: $NUM_GRAPHS"

# Run HPO with node existence masking
python hpo.py \
  --num_graphs "$NUM_GRAPHS" \
  --mask_prob "$MASK_PROB" \
  --mask_mode "node_existence" \
  --n_trials "$NUM_TRIALS" \
  --hpo_epochs "$HPO_EPOCHS" \
  --study_name "$STUDY_NAME" \
  --seed "$SEED" \
  --optimize_batch_size

echo "HPO completed for node existence masking!"
echo "Check results in: ./hpo_results/$STUDY_NAME/"