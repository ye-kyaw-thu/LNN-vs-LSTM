#!/bin/bash

# Configuration
SCRIPT_NAME="debug_iam.py"
LOG_DIR="./debug_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/iam_debug_$TIMESTAMP.log"

mkdir -p $LOG_DIR

echo "------------------------------------------------"
echo "Starting IAM Dataset Debugging..."
echo "Target: $SCRIPT_NAME"
echo "Logging to: $LOG_FILE"

# Check if the data file exists before running
DATA_PATH="./data/iam/sentences.npz"
if [ ! -f "$DATA_PATH" ]; then
    echo "WARNING: $DATA_PATH not found. The script will use dummy data for structural testing." | tee -a $LOG_FILE
fi

# Run the script
python3 $SCRIPT_NAME 2>&1 | tee -a $LOG_FILE

echo "------------------------------------------------"
echo "IAM Debugging session finished."

