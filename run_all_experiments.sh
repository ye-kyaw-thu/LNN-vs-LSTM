#!/bin/bash

# Create a directory for logs if it doesn't exist
LOG_DIR="./experiment_logs"
mkdir -p $LOG_DIR

# Define arrays for datasets and methods
DATASETS=("quickdraw" "iam" "n-mnist")
METHODS=("lstm" "lnn")

echo "Starting Liquid Neural Network vs LSTM Baseline Experiments..."

for ds in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        LOG_FILE="${LOG_DIR}/${ds}_${method}_${TIMESTAMP}.log"
        
        echo "------------------------------------------------"
        echo "Running Experiment: Dataset=$ds, Method=$method"
        echo "Logging to: $LOG_FILE"
        
        # Execute the python script and pipe both stdout and stderr to the log file
        # Using --epochs 30 and --units 256 as standard defaults
        time python3 lnn.py \
            --dataset "$ds" \
            --method "$method" \
            --epochs 30 \
            --units 256 \
            --batch_size 128 \
            --data_path "./data" \
            2>&1 | tee "$LOG_FILE"
            
        echo "Finished $ds with $method."
    done
done

echo "All experiments completed. Check the '$LOG_DIR' folder for logs."


