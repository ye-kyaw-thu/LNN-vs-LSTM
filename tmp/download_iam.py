import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"Skipping {filename}, already exists.")
        return
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def setup_iam():
    # Setup IAM directory
    target_dir = "./data/iam"
    os.makedirs(target_dir, exist_ok=True)

    # Pre-processed IAM Online Stroke Dataset (Sentences)
    iam_url = "https://github.com/sjvasquez/handwriting-synthesis/raw/master/data/sentences.npz"
    path = os.path.join(target_dir, "sentences.npz")
    
    print("--- Downloading IAM Handwriting Stroke Dataset ---")
    download_file(iam_url, path)
    print("\nIAM dataset ready in ./data/iam/")

if __name__ == "__main__":
    setup_iam()

