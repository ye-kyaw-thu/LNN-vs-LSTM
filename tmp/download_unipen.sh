#!/bin/bash

# 1. Setup Directories
TARGET_DIR="./data/unipen"
TEMP_DIR="./data/unipen_temp"
mkdir -p "$TARGET_DIR"
mkdir -p "$TEMP_DIR"

echo "--- Fetching UNIPEN Download Link via API ---"

# Record ID for the UNIPEN dataset
RECORD_ID="1195803"

# Use Python to get the actual download URL from Zenodo's API
FILE_URL=$(python3 -c "
import urllib.request, json
try:
    with urllib.request.urlopen(f'https://zenodo.org/api/records/$RECORD_ID') as response:
        data = json.loads(response.read().decode())
        # Look for the .tar.gz file in the files list
        for f in data['files']:
            if f['key'].endswith('.tar.gz'):
                print(f['links']['self'])
                break
except Exception as e:
    pass
")

if [ -z "$FILE_URL" ]; then
    echo "ERROR: Could not retrieve download URL from Zenodo API."
    exit 1
fi

echo "--- Downloading: $FILE_URL ---"
curl -L "$FILE_URL" -o "$TEMP_DIR/unipen.tar.gz"

# 2. Verify and Extract
if [[ ! -f "$TEMP_DIR/unipen.tar.gz" ]] || [[ $(file "$TEMP_DIR/unipen.tar.gz") != *"gzip compressed data"* ]]; then
    echo "ERROR: Download failed or file is not a valid gzip archive."
    exit 1
fi

echo "--- Extracting files ---"
tar -xzf "$TEMP_DIR/unipen.tar.gz" -C "$TEMP_DIR"

# 3. Convert to JSON
echo "--- Converting UNIPEN 1a (Digits) to JSON ---"

python3 <<EOF
import os
import json
import re

# Identify the correct folder in the extracted content
source_root = "$TEMP_DIR"
source_dir = None
for root, dirs, files in os.walk(source_root):
    if root.endswith("data/1a"):
        source_dir = root
        break

if not source_dir:
    print("Error: Could not find 'data/1a' directory.")
    exit(1)

digits_data = {str(i): [] for i in range(10)}

def parse_file(path):
    try:
        with open(path, 'r', encoding='latin-1') as f:
            content = f.read()
    except: return

    # UNIPEN segments look like .SEGMENT DIGIT 0-9
    segments = re.split(r'\.SEGMENT\s+DIGIT\s+', content)
    for seg in segments[1:]:
        lines = seg.strip().split('\n')
        label = lines[0].split()[0] # The digit
        if label not in digits_data: continue
        
        points = []
        in_coord = False
        for line in lines:
            line = line.strip()
            if line.startswith('.COORD'):
                in_coord = True
                continue
            if in_coord and (line.startswith('.') or not line):
                break
            if in_coord:
                coords = line.split()
                if len(coords) >= 2:
                    points.append([float(coords[0]), float(coords[1]), 0.0])
        
        if len(points) > 5:
            points[-1][2] = 1.0 # End of stroke
            digits_data[label].append(points)

for fname in os.listdir(source_dir):
    parse_file(os.path.join(source_dir, fname))

for digit, samples in digits_data.items():
    if samples:
        with open(os.path.join("$TARGET_DIR", f"{digit}.json"), 'w') as f:
            json.dump(samples, f)
        print(f"Digit {digit}: {len(samples)} samples saved.")
EOF

rm -rf "$TEMP_DIR"
echo "--- UNIPEN Dataset Ready ---"

