#!/bin/bash

# Ensure directory exists
mkdir -p data/iam

echo "Installing huggingface_hub if missing..."
pip install -q huggingface_hub

echo "Running Python downloader..."
python3 download_iam.py

echo "Download complete. Checking file count..."
find data/iam -type f | wc -l | xargs echo "Total files downloaded:"

