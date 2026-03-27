import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ncps.torch import CfC
from tqdm import tqdm

# ==========================================
# 1. Dataset Logic
# ==========================================

class NMNISTDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.samples = []
        phase = "Train" if is_train else "Test"
        path = os.path.join(root_dir, "NMNIST", phase)
        for label in range(10):
            folder = os.path.join(path, str(label))
            files = glob.glob(os.path.join(folder, "*.bin"))
            for f in files: self.samples.append((f, label))

    def _load_bin(self, f_path):
        with open(f_path, 'rb') as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)
        return raw[0::5], raw[1::5], (raw[2::5] & 128) >> 7, ((raw[2::5] & 127) << 16) | (raw[3::5] << 8) | raw[4::5]

    def __getitem__(self, i):
        x, y, p, t = self._load_bin(self.samples[i][0])
        frames = np.zeros((10, 2, 34, 34), dtype=np.float32)
        if len(t) > 0:
            t_bin = (t[-1] - t[0]) / 10 if t[-1] > t[0] else 1
            for j in range(len(t)):
                idx = min(int((t[j] - t[0]) // t_bin), 9)
                if x[j] < 34 and y[j] < 34: frames[idx, p[j], y[j], x[j]] += 1.0
        return torch.from_numpy(np.log1p(frames)), self.samples[i][1]

    def __len__(self): return len(self.samples)

class QuickDrawDataset(Dataset):
    def __init__(self, root_dir, max_len=100):
        self.data, self.labels = [], []
        categories = ["airplane", "apple", "banana", "basketball", "bed", "bee", "bicycle", "bird", "book", "butterfly"]
        for idx, cat in enumerate(categories):
            path = os.path.join(root_dir, "quickdraw", f"{cat}.ndjson")
            if not os.path.exists(path): continue
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 5000: break  # Restored to 5000 for training quality
                    drawing = json.loads(line)['drawing']
                    seq = []
                    last_x, last_y = drawing[0][0][0], drawing[0][1][0]
                    for stroke in drawing:
                        x_pts, y_pts = stroke[0], stroke[1]
                        for j in range(len(x_pts)):
                            nx, ny = x_pts[j]/255.0, y_pts[j]/255.0
                            ndx, ndy = (x_pts[j]-last_x)/255.0, (y_pts[j]-last_y)/255.0
                            p = 1.0 if j == len(x_pts)-1 else 0.0
                            seq.append([nx, ny, ndx, ndy, p]) # 5 features
                            last_x, last_y = x_pts[j], y_pts[j]
                    self.data.append(torch.tensor(seq[:max_len], dtype=torch.float32))
                    self.labels.append(idx)

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.labels[i]

def collate_qd(batch):
    seqs, labels = zip(*batch)
    return pad_sequence(seqs, batch_first=True), torch.tensor(labels)

# ==========================================
# 2. Improved Model Logic
# ==========================================

class DrawingClassifier(nn.Module):
    def __init__(self, method, dataset_type, units=256):
        super().__init__()
        self.method = method
        if dataset_type == 'nmnist':
            self.backbone = nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                nn.Linear(32 * 17 * 17, 128), nn.LayerNorm(128), nn.ReLU()
            )
            inp_dim = 128
        else:
            self.backbone = nn.Sequential(nn.Linear(5, 128), nn.LayerNorm(128), nn.ReLU())
            inp_dim = 128
        
        if method == 'lnn':
            self.core = CfC(inp_dim, units, proj_size=units, batch_first=True, backbone_layers=1, backbone_units=128)
        else:
            self.core = nn.LSTM(inp_dim, units, batch_first=True, num_layers=2, dropout=0.1)
            
        self.classifier = nn.Linear(units, 10)

    def forward(self, x):
        b, t = x.shape[0], x.shape[1]
        x = self.backbone(x.view(b*t, *x.shape[2:])).view(b, t, -1)
        out, _ = self.core(x)
        # Use mean pooling as per your successful script
        pooled = torch.mean(out, dim=1)
        return self.classifier(pooled)

# ==========================================
# 3. Training & Stress Logic
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['nmnist', 'quickdraw'], required=True)
    parser.add_argument('--method', choices=['lnn', 'lstm'], required=True)
    parser.add_argument('--data', type=str, default="./data")
    parser.add_argument('--units', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--drop_rates', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.7])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: {args.dataset} | Method: {args.method}")

    if args.dataset == 'nmnist':
        train_ds = NMNISTDataset(args.data, is_train=True)
        test_ds = NMNISTDataset(args.data, is_train=False)
        loader_args = {}
    else:
        full_qd = QuickDrawDataset(args.data)
        train_size = int(0.85 * len(full_qd))
        train_ds, test_ds = torch.utils.data.random_split(full_qd, [train_size, len(full_qd)-train_size])
        loader_args = {'collate_fn': collate_qd}

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, **loader_args)
    test_loader = DataLoader(test_ds, batch_size=128, **loader_args)

    model = DrawingClassifier(args.method, args.dataset, args.units).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    for e in range(args.epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {e+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Stress Evaluation
    model.eval()
    for dr in args.drop_rates:
        corr, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                if dr > 0:
                    mask = (torch.rand(x.size(0), x.size(1), 1, device=device) > dr).float()
                    # Reshape mask for NMNIST (5D) or QuickDraw (3D)
                    for _ in range(len(x.shape) - 3): mask = mask.unsqueeze(-1)
                    x = x * mask
                corr += (model(x).argmax(1) == y.to(device)).sum().item()
                total += y.size(0)
        print(f"RESULT | Drop: {dr*100}% | Accuracy: {corr/total:.4f}")

if __name__ == "__main__":
    main()

