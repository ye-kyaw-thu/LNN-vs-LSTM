#!/bin/bash

time python ./lnn_mnist.py --method lstm --epochs 30 | tee lstm_mnist_30.log

