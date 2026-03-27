#!/bin/bash
mkdir -p ./results_stress_test

# Higher epochs for stress test to ensure convergence
EPOCHS=30

for dset in nmnist quickdraw; do
    for method in lstm lnn; do
        echo "Running: $dset with $method"
        time python stress_test.py \
            --dataset $dset \
            --method $method \
            --data ./data \
            --epochs $EPOCHS \
            --units 256 \
            --drop_rates 0.0 0.3 0.5 0.7 > ./results_stress_test/${dset}_${method}.log
    done
done
echo "Stress test complete."

