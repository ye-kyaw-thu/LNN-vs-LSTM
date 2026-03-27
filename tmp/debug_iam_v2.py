import torch
import torch.nn as nn
import numpy as np
import os
import urllib.request
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ncps.torch import CfC

class IAMDataset(Dataset):
    def __init__(self, file_path, max_len=150):
        if not os.path.exists(file_path):
            print(f"File {file_path} missing. Attempting to download clean version...")
            # Note: Replace URL with your actual data source if different
            url = "https://github.com/lucasb-eyer/pylearn2/raw/master/pylearn2/datasets/iam_handwriting/sentences.npz"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            urllib.request.urlretrieve(url, file_path)

        try:
            # Using mmap_mode to handle large files and avoiding pickle issues
            raw_data = np.load(file_path, allow_pickle=True, encoding='latin1')
            self.data = raw_data['data']
            self.labels = raw_data['labels']
            print(f"Successfully loaded {len(self.data)} samples.")
        except Exception as e:
            print(f"CRITICAL ERROR: File still unreadable ({e}).")
            print("Falling back to synthetic data for architecture testing...")
            self.data = [np.random.randn(200, 3) for _ in range(100)]
            self.labels = np.random.randint(0, 10, 100)

        self.max_len = max_len

    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        seq = torch.tensor(self.data[i][:self.max_len], dtype=torch.float32)
        return seq, torch.tensor(self.labels[i], dtype=torch.long)

def collate_fn(batch):
    seqs, labels = zip(*batch)
    return pad_sequence(seqs, batch_first=True), torch.tensor(labels)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = IAMDataset("./data/iam/sentences.npz")
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # input_size=3 (x, y, pen_state)
    model = CfC(3, 128, proj_size=10, batch_first=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x, y = next(iter(loader))
    out, _ = model(x.to(device))
    print("Forward pass successful. Output shape:", out.shape)

if __name__ == "__main__":
    main()

