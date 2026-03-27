import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ncps.torch import CfC

class NMNISTDataset(Dataset):
    def __init__(self, data_list, labels, max_len=300):
        self.data = []
        self.labels = labels
        # N-MNIST Events: [x, y, t, p]
        for seq in data_list:
            # NORMALIZATION is key for N-MNIST
            seq = np.array(seq, dtype=np.float32)
            if len(seq) > 0:
                seq[:, 0] /= 34.0  # x-res
                seq[:, 1] /= 34.0  # y-res
                seq[:, 2] /= seq[:, 2].max() if seq[:, 2].max() > 0 else 1.0 # t
            self.data.append(torch.tensor(seq[:max_len], dtype=torch.float32))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.labels[i]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    return pad_sequence(sequences, batch_first=True), torch.tensor(labels, dtype=torch.long)

def debug_nmnist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create Dummy Data to verify model architecture (Replace with your actual loader)
    dummy_data = [np.random.rand(500, 4) for _ in range(100)]
    dummy_labels = np.random.randint(0, 10, 100)
    
    ds = NMNISTDataset(dummy_data, dummy_labels)
    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Input size 4: x, y, t, p
    model = CfC(4, 256, proj_size=10, batch_first=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    print("Starting N-MNIST Debug Run...")
    model.train()
    for epoch in range(5):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = model(x)
            # Use mean pooling for event data as it's often more stable than last-state
            logits = torch.mean(out, dim=1) 
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    debug_nmnist()

