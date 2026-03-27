import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ncps.torch import CfC

class IAMDataset(Dataset):
    def __init__(self, file_path, max_len=150):
        # Debugging the load error
        try:
            raw_data = np.load(file_path, allow_pickle=True, encoding='latin1')
            # npz files are dictionaries; we need to extract the arrays
            if hasattr(raw_data, 'files'):
                self.data = raw_data['data']
                self.labels = raw_data['labels']
            else:
                self.data = raw_data
                # Assuming labels are provided in the same file or derived
        except Exception as e:
            print(f"FAILED TO LOAD: {e}")
            # Create dummy data for structural debugging if file fails
            print("Creating dummy data for structural debugging...")
            self.data = [np.random.randn(200, 3) for _ in range(100)]
            self.labels = np.random.randint(0, 10, 100)

        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        # IAM typically: [delta_x, delta_y, p1]
        seq = torch.tensor(self.data[i][:self.max_len], dtype=torch.float32)
        label = torch.tensor(self.labels[i], dtype=torch.long)
        return seq, label

def collate_fn(batch):
    sequences, labels = zip(*batch)
    return pad_sequence(sequences, batch_first=True), torch.tensor(labels)

# Model and Training loop (Simplified for debugging)
def debug_iam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = IAMDataset("./data/iam/sentences.npz")
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # IAM input_size is typically 3 (dx, dy, pen_up)
    model = CfC(3, 128, proj_size=10, batch_first=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Starting IAM Debug Run...")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out, _ = model(x)
        # Use last state or mean pool
        logits = out[:, -1, :] 
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item():.4f} | Accuracy: {(logits.argmax(1) == y).float().mean():.4f}")
        break # Run 1 batch just to verify flow

if __name__ == "__main__":
    debug_iam()

