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
# 1. Master Category List
# ==========================================
MASTER_CATEGORIES = [
    "airplane", "apple", "banana", "basketball", "bed", "bee", "bicycle", "bird", "book", "butterfly",
    "bridge", "broom", "bucket", "bus", "calculator", "camel", "camera", "candle", "car", "carrot",
    "castle", "cat", "chair", "cheese", "clock", "cloud", "coffee cup", "compass", "computer", "cookie",
    "cow", "crab", "crocodile", "crown", "cruise ship", "cup", "diamond", "dog", "dolphin", "donut",
    "door", "dragon", "drums", "duck", "dumbbell", "ear", "elbow", "elephant", "envelope", "eye"
]

# ==========================================
# 2. Data Pipeline & Downloading
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

def collate_fn(batch):
    sequences, labels = zip(*batch)
    return pad_sequence(sequences, batch_first=True), torch.tensor(labels)

def download_data(data_path, target_count):
    os.makedirs(data_path, exist_ok=True)
    loaded_data = []
    selected_names = []
    
    cat_idx = 0
    pbar = tqdm(total=target_count, desc="Loading Categories")
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
        except Exception as e:
            print(f"\nSkipping {name}: {e}")
        cat_idx += 1
    pbar.close()
    return loaded_data, selected_names

# ==========================================
# 3. Model Architecture
# ==========================================

class DrawingClassifier(nn.Module):
    def __init__(self, method, input_size, num_classes, units=256):
        super().__init__()
        self.method = method.lower()
        self.feature_map = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        if self.method == 'lnn':
            self.core = CfC(128, units, proj_size=units, batch_first=True, 
                            backbone_layers=1, backbone_units=128)
        else:
            self.core = nn.LSTM(128, units, batch_first=True, num_layers=2, dropout=0.1)
            
        self.classifier = nn.Sequential(
            nn.Linear(units, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feature_map(x)
        out, _ = self.core(x)
        pooled = torch.mean(out, dim=1)
        return self.classifier(pooled)

# ==========================================
# 4. Metrics Helper
# ==========================================

def get_metrics(trues, preds):
    acc = accuracy_score(trues, preds)
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average='macro', zero_division=0)
    return acc, p, r, f1

# ==========================================
# 5. Training Engine
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--method', choices=['lnn', 'lstm'], default='lnn')
    parser.add_argument('--units', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_list, class_names = download_data(args.data_path, args.num_classes)
    full_ds = QuickDrawDataset(data_list)
    
    train_size = int(0.85 * len(full_ds))
    train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, len(full_ds)-train_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    model = DrawingClassifier(args.method, 5, len(class_names), args.units).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}

    for epoch in range(args.epochs):
        # Training
        model.train()
        t_preds, t_trues = [], []
        t_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            t_loss += loss.item()
            t_preds.extend(logits.argmax(1).cpu().numpy())
            t_trues.extend(y.cpu().numpy())

        # Validation
        model.eval()
        v_preds, v_trues = [], []
        v_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x.to(device))
                v_loss += criterion(logits, y.to(device)).item()
                v_preds.extend(logits.argmax(1).cpu().numpy())
                v_trues.extend(y.numpy())

        # Metrics Calculation
        t_acc, t_p, t_r, t_f1 = get_metrics(t_trues, t_preds)
        v_acc, v_p, v_r, v_f1 = get_metrics(v_trues, v_preds)
        
        history['train_acc'].append(t_acc)
        history['test_acc'].append(v_acc)
        history['train_loss'].append(t_loss/len(train_loader))
        history['test_loss'].append(v_loss/len(test_loader))
        
        scheduler.step(v_acc)

        print(f"\n[Epoch {epoch+1}] Method: {args.method.upper()}")
        print(f"TRAIN | Acc: {t_acc:.4f} | F1: {t_f1:.4f} | P: {t_p:.4f} | R: {t_r:.4f} | Loss: {t_loss/len(train_loader):.4f}")
        print(f"TEST  | Acc: {v_acc:.4f} | F1: {v_f1:.4f} | P: {v_p:.4f} | R: {v_r:.4f} | Loss: {v_loss/len(test_loader):.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'], label='Test')
    plt.title('Loss History'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy History'); plt.legend()
    plt.savefig('metrics_history.png')

    cm = confusion_matrix(v_trues, v_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {args.method.upper()}')
    plt.savefig('confusion_matrix.png')

if __name__ == "__main__":
    main()

