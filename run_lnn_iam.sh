#!/bin/bash
# (pytorch_py3.10) active

# Run Liquid Neural Network (CfC) on IAM-line
echo "Starting IAM Experiment: Method = LNN (CfC)"
time python lnn_iam.py \
    --dataset iam \
    --method lnn \
    --units 256 \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.0005 | tee lnn_iam_30.log

echo "Experiment Complete. Log saved to lnn_iam_30.log"

