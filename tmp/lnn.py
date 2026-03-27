import os
import json
import argparse
import urllib.request
import urllib.parse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

# LNN specific import
from ncps.torch import CfC

# ==========================================
# 1. Master Category List (QuickDraw)
# ==========================================
MASTER_CATEGORIES = [
    "airplane", "apple", "banana", "basketball", "bed", "bee", "bicycle", "bird", "book", "butterfly",
    "bridge", "broom", "bucket", "bus", "calculator", "camel", "camera", "candle", "car", "carrot",
    "castle", "cat", "chair", "cheese", "clock", "cloud", "coffee cup", "compass", "computer", "cookie",
    "cow", "crab", "crocodile", "crown", "cruise ship", "cup", "diamond", "dog", "dolphin", "donut",
    "door", "dragon", "drums", "duck", "dumbbell", "ear", "elbow", "elephant", "envelope", "eye"
]

# ==========================================
# 2. Dataset Classes
# ==========================================

class QuickDrawDataset(Dataset):
    def __init__(self, data_list, max_len=100):
        self.data = []
        self.labels = []
        for idx, samples in enumerate(data_list):
            for drawing in samples:
                points = self._process_drawing(drawing)
                if len(points) > 5:
                    self.data.append(torch.tensor(points[:max_len], dtype=torch.float32))
                    self.labels.append(idx)

    def _process_drawing(self, drawing):
        seq = []
        last_x, last_y = drawing[0][0][0], drawing[0][1][0]
        for stroke in drawing:
            x_pts, y_pts = stroke[0], stroke[1]
            for i in range(len(x_pts)):
                curr_x, curr_y = x_pts[i], y_pts[i]
                nx, ny = curr_x / 255.0, curr_y / 255.0
                ndx, ndy = (curr_x - last_x) / 255.0, (curr_y - last_y) / 255.0
                p = 1.0 if i == len(x_pts) - 1 else 0.0
                seq.append([nx, ny, ndx, ndy, p])
                last_x, last_y = curr_x, curr_y
        return seq

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.labels[i]

class IAMDataset(Dataset):
    def __init__(self, file_path, max_len=200):
        self.data = []
        self.labels = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"IAM file not found at {file_path}. Please run download script.")
        
        try:
            # Use allow_pickle=True to handle the stroke object arrays
            raw_data = np.load(file_path, allow_pickle=True, encoding='latin1')
            # The 'train_strokes' key contains the coordinate sequences
            strokes = raw_data['train_strokes']
            for i, s in enumerate(strokes[:5000]): # Limit for demo
                if len(s) > 10:
                    self.data.append(torch.tensor(s[:max_len], dtype=torch.float32))
                    self.labels.append(i % 10) # 10 synthetic classes
        except Exception as e:
            print(f"Error loading IAM dataset: {e}")
            raise

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.labels[i]

class NMNISTDataset(Dataset):
    def __init__(self, root_dir, max_len=300):
        self.data = []
        self.labels = []
        test_path = os.path.join(root_dir, 'NMNIST', 'Test')
        
        for label in range(10):
            label_dir = os.path.join(test_path, str(label))
            if not os.path.exists(label_dir): continue
            
            files = sorted(os.listdir(label_dir))[:500] 
            for f in files:
                events = self._read_binary(os.path.join(label_dir, f))
                if events is not None:
                    self.data.append(torch.tensor(events[:max_len], dtype=torch.float32))
                    self.labels.append(label)

    def _read_binary(self, file_path):
        with open(file_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint8)
        if len(raw_data) % 5 != 0: return None
        
        x = raw_data[0::5]
        y = raw_data[1::5]
        ts = raw_data[2::5].astype(np.uint32) | \
             (raw_data[3::5].astype(np.uint32) << 8) | \
             ((raw_data[4::5].astype(np.uint32) & 0xFE) << 15)
        p = raw_data[4::5] & 0x01
        
        # Norm: x/34, y/34, delta_t, polarity
        events = np.column_stack([x/34.0, y/34.0, np.zeros_like(ts), p])
        events[1:, 2] = np.diff(ts) / 1000.0 # ms
        return events

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.labels[i]

# ==========================================
# 3. Helpers & Core logic
# ==========================================

def collate_fn(batch):
    sequences, labels = zip(*batch)
    return pad_sequence(sequences, batch_first=True), torch.tensor(labels)

def download_quickdraw_data(data_path, target_count):
    os.makedirs(data_path, exist_ok=True)
    loaded_data = []
    selected_names = []
    cat_idx = 0
    pbar = tqdm(total=target_count, desc="Loading QuickDraw Categories")
    while len(selected_names) < target_count and cat_idx < len(MASTER_CATEGORIES):
        name = MASTER_CATEGORIES[cat_idx]
        safe_name = name.replace(" ", "_")
        file_path = os.path.join(data_path, f"{safe_name}.ndjson")
        url_name = urllib.parse.quote(name)
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{url_name}.ndjson"
        try:
            if not os.path.exists(file_path):
                urllib.request.urlretrieve(url, file_path)
            samples = []
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 5000: break
                    item = json.loads(line)
                    if item.get('recognized', True):
                        samples.append(item['drawing'])
            loaded_data.append(samples)
            selected_names.append(name)
            pbar.update(1)
        except Exception:
            pass
        cat_idx += 1
    pbar.close()
    return loaded_data, selected_names

class DrawingClassifier(nn.Module):
    def __init__(self, method, input_size, num_classes, units=256):
        super().__init__()
        self.method = method.lower()
        self.feature_map = nn.Sequential(nn.Linear(input_size, 128), nn.LayerNorm(128), nn.ReLU())
        if self.method == 'lnn':
            self.core = CfC(128, units, proj_size=units, batch_first=True, backbone_layers=1)
        else:
            self.core = nn.LSTM(128, units, batch_first=True, num_layers=2, dropout=0.1)
        self.classifier = nn.Sequential(nn.Linear(units, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.feature_map(x)
        out, _ = self.core(x)
        return self.classifier(torch.mean(out, dim=1))

def get_metrics(trues, preds):
    acc = accuracy_score(trues, preds)
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average='macro', zero_division=0)
    return acc, p, r, f1

# ==========================================
# 4. Main Entry
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['quickdraw', 'iam', 'n-mnist'], default='quickdraw')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--method', choices=['lnn', 'lstm'], default='lnn')
    parser.add_argument('--units', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.dataset == 'iam':
        full_ds = IAMDataset(os.path.join(args.data_path, "iam", "sentences.npz"))
        input_dim, num_classes = 3, 10
    elif args.dataset == 'n-mnist':
        full_ds = NMNISTDataset(os.path.join(args.data_path, "N-MNIST"))
        input_dim, num_classes = 4, 10
    else:
        data_list, class_names = download_quickdraw_data(os.path.join(args.data_path, "quickdraw"), args.num_classes)
        full_ds = QuickDrawDataset(data_list)
        input_dim, num_classes = 5, len(class_names)

    train_size = int(0.85 * len(full_ds))
    train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, len(full_ds)-train_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    model = DrawingClassifier(args.method, input_dim, num_classes, args.units).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}

    for epoch in range(args.epochs):
        model.train()
        t_preds, t_trues, t_loss = [], [], 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            t_loss += loss.item()
            t_preds.extend(logits.argmax(1).cpu().numpy()); t_trues.extend(y.cpu().numpy())

        model.eval()
        v_preds, v_trues, v_loss = [], [], 0
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x.to(device))
                v_loss += criterion(logits, y.to(device)).item()
                v_preds.extend(logits.argmax(1).cpu().numpy()); v_trues.extend(y.numpy())

        t_acc, _, _, _ = get_metrics(t_trues, t_preds)
        v_acc, _, _, _ = get_metrics(v_trues, v_preds)
        history['train_acc'].append(t_acc); history['test_acc'].append(v_acc)
        history['train_loss'].append(t_loss/len(train_loader)); history['test_loss'].append(v_loss/len(test_loader))
        scheduler.step(v_acc)

        print(f"\n[Epoch {epoch+1}] Dataset: {args.dataset.upper()} | Method: {args.method.upper()}")
        print(f"TRAIN | Acc: {t_acc:.4f} | Loss: {t_loss/len(train_loader):.4f}")
        print(f"TEST  | Acc: {v_acc:.4f} | Loss: {v_loss/len(test_loader):.4f}")
        print("-" * 30)

if __name__ == "__main__":
    main()

