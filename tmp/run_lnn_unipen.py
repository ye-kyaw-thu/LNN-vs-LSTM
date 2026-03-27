#!/bin/bash
# Run LNN on UNIPEN (Ensure data is in ./data/unipen)
time python lnn_main.py --dataset unipen --method lnn --num_classes 10 --units 256 --epochs 30 \
--data_path ./data/unipen | tee lnn_up_10.log
