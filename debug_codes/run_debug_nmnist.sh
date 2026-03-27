#!/bin/bash

# Configuration
SCRIPT_NAME="debug_nmnist.py"
LOG_DIR="./debug_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/nmnist_debug_$TIMESTAMP.log"

mkdir -p $LOG_DIR

echo "------------------------------------------------"
echo "Starting N-MNIST Dataset Debugging..."
echo "Target: $SCRIPT_NAME"
echo "Logging to: $LOG_FILE"
echo "Goal: Verify if accuracy exceeds 10% (random guess baseline)"

# Run the script
# We use stdbuf to ensure the training progress is logged in real-time
stdbuf -oL python3 $SCRIPT_NAME 2>&1 | tee -a $LOG_FILE

echo "------------------------------------------------"
echo "N-MNIST Debugging session finished. Check $LOG_FILE for loss trends."

