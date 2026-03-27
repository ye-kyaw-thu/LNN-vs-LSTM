#!/bin/bash
# (pytorch_py3.10) active

# Run Baseline LSTM on IAM-line
echo "Starting IAM Experiment: Method = LSTM"
time python lnn_iam.py \
    --dataset iam \
    --method lstm \
    --units 256 \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.0005 | tee lstm_iam_30.log

echo "Experiment Complete. Log saved to lstm_iam_30.log"

