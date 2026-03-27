#!/bin/bash
# (pytorch_py3.10) active

echo "=== TESTING IAM RECOVERY ==="
python3 debug_iam_v2.py

echo -e "\n=== TESTING N-MNIST STABILITY ==="
python3 debug_nmnist_v2.py
