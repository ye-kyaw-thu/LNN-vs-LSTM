#!/bin/bash
# Run LNN on QuickDraw
time python lnn_main.py --dataset quickdraw --method lnn --num_classes 10 --units 256 --epochs 30 | tee lnn_qd_10.log

