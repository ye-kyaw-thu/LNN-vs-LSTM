import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from ncps.torch import CfC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ==========================================
# 1. N-MNIST Data Loader (Raw .bin to Frames)
# ==========================================
class NMNISTDataset(Dataset):
    def __init__(self, root_dir, time_bins=10, is_train=True):
        self.time_bins = time_bins
        self.samples = []
        phase = "Train" if is_train else "Test"
        path = os.path.join(root_dir, phase)
        
        for label in range(10):
            folder = os.path.join(path, str(label))
            files = glob.glob(os.path.join(folder, "*.bin"))
            for f in files:
                self.samples.append((f, label))

    def _load_bin(self, file_path):
        """Reads N-MNIST .bin file format."""
        with open(file_path, 'rb') as f:
            raw_data = np.frombuffer(f.read(), dtype=np.uint8)
            
        raw_data = raw_data.astype(np.int32)
        x = raw_data[0::5]
        y = raw_data[1::5]
        p = (raw_data[2::5] & 128) >> 7
        t = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | raw_data[4::5]
        return x, y, p, t

    def _to_frames(self, x, y, p, t):
        """Aggregates asynchronous events into spatial frames."""
        frames = np.zeros((self.time_bins, 2, 34, 34), dtype=np.float32)
        if len(t) == 0: return frames
        
        t_start, t_end = t[0], t[-1]
        t_bin = (t_end - t_start) / self.time_bins if t_end > t_start else 1
        
        for i in range(len(t)):
            bin_idx = int((t[i] - t_start) // t_bin)
            bin_idx = min(bin_idx, self.time_bins - 1)
            if x[i] < 34 and y[i] < 34:
                frames[bin_idx, p[i], y[i], x[i]] += 1.0
        return np.log1p(frames)

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        f_path, label = self.samples[i]
        x, y, p, t = self._load_bin(f_path)
        frames = self._to_frames(x, y, p, t)
        return torch.from_numpy(frames), label

# ==========================================
# 2. Model Architecture (CNN Backbone + RNN)
# ==========================================
class NMNISTClassifier(nn.Module):
    def __init__(self, method='lnn', units=128):
        super().__init__()
        self.method = method
        # Lightweight CNN for 34x34 resolution
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        cnn_out_dim = 64 * 8 * 8
        if method == 'lnn':
            self.rnn = CfC(cnn_out_dim, units, proj_size=units, batch_first=True)
        else:
            self.rnn = nn.LSTM(cnn_out_dim, units, batch_first=True, num_layers=2)
        self.classifier = nn.Linear(units, 10)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        features = self.backbone(x).view(b, t, -1)
        out, _ = self.rnn(features)
        return self.classifier(torch.mean(out, dim=1))

# ==========================================
# 3. Helper for Metrics
# ==========================================
def calculate_metrics(trues, preds):
    acc = accuracy_score(trues, preds)
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average='macro', zero_division=0)
    return acc, p, r, f1

# ==========================================
# 4. Main Training Engine
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['lnn', 'lstm'], default='lnn')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--units', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data_path', type=str, default="./data/NMNIST")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = NMNISTDataset(args.data_path, is_train=True)
    test_ds = NMNISTDataset(args.data_path, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4)

    model = NMNISTClassifier(method=args.method, units=args.units).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {'t_acc': [], 'v_acc': [], 't_loss': []}

    for epoch in range(args.epochs):
        model.train()
        t_preds, t_trues, t_loss = [], [], 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_preds.extend(logits.argmax(1).cpu().numpy())
            t_trues.extend(y.cpu().numpy())

        model.eval()
        v_preds, v_trues = [], []
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x.to(device))
                v_preds.extend(logits.argmax(1).cpu().numpy())
                v_trues.extend(y.numpy())

        # Metrics Calculation
        ta, tp, tr, tf1 = calculate_metrics(t_trues, t_preds)
        va, vp, vr, vf1 = calculate_metrics(v_trues, v_preds)
        
        history['t_acc'].append(ta); history['v_acc'].append(va)
        history['t_loss'].append(t_loss/len(train_loader))
        scheduler.step()

        print(f"\n[Epoch {epoch+1}] Method: {args.method.upper()}")
        print(f"TRAIN | Acc: {ta:.4f} | F1: {tf1:.4f} | P: {tp:.4f} | R: {tr:.4f} | Loss: {t_loss/len(train_loader):.4f}")
        print(f"TEST  | Acc: {va:.4f} | F1: {vf1:.4f} | P: {vp:.4f} | R: {vr:.4f}")
        print("-" * 60)

    # Final Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['t_loss']); plt.title(f"{args.method.upper()} Training Loss")
    plt.subplot(1, 2, 2); plt.plot(history['t_acc'], label='Train'); plt.plot(history['v_acc'], label='Test')
    plt.title("Accuracy History"); plt.legend(); plt.savefig(f"nmnist_{args.method}_curves.png")

    # Confusion Matrix
    cm = confusion_matrix(v_trues, v_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'N-MNIST Confusion Matrix - {args.method.upper()}')
    plt.xlabel('Predicted'); plt.ylabel('Reference')
    plt.savefig(f"nmnist_{args.method}_cm.png")

    # Save Predictions for Manual Analysis
    with open(f"predictions_nmnist_{args.method}.txt", "w") as f:
        f.write("REF, PRED\n")
        for r, p in zip(v_trues, v_preds):
            f.write(f"{r}, {p}\n")
    print(f"Analysis logs saved to predictions_nmnist_{args.method}.txt")

if __name__ == "__main__":
    main()

