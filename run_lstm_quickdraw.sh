#!/bin/bash
# Run LSTM on QuickDraw
time python lnn_main.py --dataset quickdraw --method lstm --num_classes 10 --units 256 --epochs 30 | tee lstm_qd_10.log

