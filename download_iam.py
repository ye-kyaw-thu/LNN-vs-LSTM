import os
from huggingface_hub import snapshot_download

def download_iam_line():
    # Define the target directory
    target_path = "data/iam"
    
    print(f"--- Starting Download: Teklia/IAM-line ---")
    print(f"Target Directory: {os.path.abspath(target_path)}")

    try:
        # download_repo downloads the entire dataset repository
        # allow_patterns ensures we get the images and the metadata/labels
        snapshot_download(
            repo_id="Teklia/IAM-line",
            repo_type="dataset",
            local_dir=target_path,
            local_dir_use_symlinks=False,
            allow_patterns=["*.jpg", "*.png", "*.json", "*.txt", "data/*"]
        )
        print("\nSUCCESS: IAM-line dataset downloaded successfully.")
        
        # Check structure
        subdirs = [d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))]
        print(f"Contents of data/iam: {subdirs}")

    except Exception as e:
        print(f"\nERROR: Could not download dataset. Reason: {e}")

if __name__ == "__main__":
    download_iam_line()

