#!/bin/bash
#SBATCH --job-name=aig-transformer-multi
#SBATCH --partition=gpu_a100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=48:00:00          # Increased time for multiple runs
#SBATCH --output=slurm_logs/multi_mask_%j.out

cd ..

# Create log directory if it doesn't exist
# Setup environment
module purge
module load 2024
module load Anaconda3/2024.06-1

# Activate your environment
source activate aig-ml

# Print environment info
echo "Job started at $(date)"

# Define masking ratios to test
MASK_RATIOS=(0.1 0.3 0.5 0.7)
MASK_MODE="and_gates"  # Single mask mode

# Create a master output directory
MASTER_DIR="logs/multi_mask_experiment_${SLURM_JOB_ID}"
mkdir -p ${MASTER_DIR}

# Save experiment metadata
echo "Starting multi-masking experiment with the following parameters:" > ${MASTER_DIR}/experiment_info.txt
echo "Mask ratios: ${MASK_RATIOS[*]}" >> ${MASTER_DIR}/experiment_info.txt
echo "Mask mode: ${MASK_MODE}" >> ${MASTER_DIR}/experiment_info.txt
echo "Job ID: ${SLURM_JOB_ID}" >> ${MASTER_DIR}/experiment_info.txt
echo "Start time: $(date)" >> ${MASTER_DIR}/experiment_info.txt

# Run experiments with different masking ratios
for RATIO in "${MASK_RATIOS[@]}"; do
    echo "========================================================"
    echo "Starting experiment with mask_ratio=${RATIO}, mask_mode=${MASK_MODE}"
    echo "========================================================"

    # Create experiment-specific output directory
    EXP_DIR="${MASTER_DIR}/mask${RATIO}"
        mkdir -p ${EXP_DIR}

        # Run the training script
        srun python cluster_run.py \
            --batch_size=32 \
            --num_epochs=100 \
            --learning_rate=1e-3 \
            --weight_decay=1e-5 \
            --output_dir=${EXP_DIR} \
            --data_path="complete_tt_graphs.pkl" \
            --mask_ratio=${RATIO} \
            --mask_mode=${MASK_MODE} \
            --seed=42

        # Record completion of this experiment
        echo "Completed mask_ratio=${RATIO} at $(date)" >> ${MASTER_DIR}/progress.txt

        # Optional: sync to disk to ensure logs are saved even if job fails
        sync
    done
done

# Generate a summary of results
echo "========================================================"
echo "Generating results summary"
echo "========================================================"

# Create a summary file
SUMMARY_FILE="${MASTER_DIR}/results_summary.txt"
echo "AIG Transformer Multi-Mask Experiment Results" > ${SUMMARY_FILE}
echo "Run on $(date)" >> ${SUMMARY_FILE}
echo "=======================================================" >> ${SUMMARY_FILE}

# Collect test results from each experiment
echo "Results for mask_mode=${MASK_MODE}:" >> ${SUMMARY_FILE}
echo "-------------------------------------------------------" >> ${SUMMARY_FILE}
echo "Mask Ratio | Test Loss | Binary Accuracy | L1 Loss" >> ${SUMMARY_FILE}
echo "-------------------------------------------------------" >> ${SUMMARY_FILE}

for RATIO in "${MASK_RATIOS[@]}"; do
    TEST_RESULTS="${MASTER_DIR}/mask${RATIO}/test_results.json"
        if [ -f "$TEST_RESULTS" ]; then
            # Extract metrics using jq (if available) or grep
            if command -v jq &> /dev/null; then
                TEST_LOSS=$(jq -r '.test_loss' ${TEST_RESULTS})
                BINARY_ACC=$(jq -r '.test_binary_accuracy' ${TEST_RESULTS})
                L1_LOSS=$(jq -r '.test_l1_loss' ${TEST_RESULTS})
            else
                TEST_LOSS=$(grep -o '"test_loss":[^,}]*' ${TEST_RESULTS} | cut -d':' -f2)
                BINARY_ACC=$(grep -o '"test_binary_accuracy":[^,}]*' ${TEST_RESULTS} | cut -d':' -f2)
                L1_LOSS=$(grep -o '"test_l1_loss":[^,}]*' ${TEST_RESULTS} | cut -d':' -f2)
            fi
            echo "  ${RATIO}   |   ${TEST_LOSS}   |   ${BINARY_ACC}   |   ${L1_LOSS}" >> ${SUMMARY_FILE}
        else
            echo "  ${RATIO}   |   Results file not found" >> ${SUMMARY_FILE}
        fi
    done
echo "" >> ${SUMMARY_FILE}

# Print job completion message
echo "Multi-masking experiment completed at $(date)"
echo "Results are available in ${MASTER_DIR}"
echo "========================================================"