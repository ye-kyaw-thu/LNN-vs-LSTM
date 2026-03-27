#!/bin/bash

time python ./lnn_nmnist.py --method lstm --epochs 10 | tee lstm_nmnist_10.log
