import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import torchvision.transforms as transforms
from ncps.torch import CfC
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
from Levenshtein import distance as lev_dist

# ==========================================
# 1. Improved Metrics & Logging
# ==========================================
def calculate_htr_metrics(y_true, y_pred):
    total_cer, total_wer = 0, 0
    for t, p in zip(y_true, y_pred):
        total_cer += lev_dist(t, p) / max(len(t), 1)
        t_words, p_words = t.split(), p.split()
        total_wer += lev_dist(t_words, p_words) / max(len(t_words), 1)
    return total_cer / len(y_true), total_wer / len(y_true)

def ctc_greedy_decoder(logits, idx_to_char):
    probs = torch.softmax(logits, dim=2)
    best_path = torch.argmax(probs, dim=2).transpose(0, 1)
    decoded_batch = []
    for line in best_path:
        decoded_str = []
        last_char = None
        for char_idx in line:
            idx = char_idx.item()
            if idx != 0 and idx != last_char:
                decoded_str.append(idx_to_char.get(idx, ""))
            last_char = idx
        decoded_batch.append("".join(decoded_str))
    return decoded_batch

# ==========================================
# 2. ResNet Backbone + Positional Encoding
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ==========================================
# 3. Main Model Architecture
# ==========================================
class IAMModel(nn.Module):
    def __init__(self, vocab_size, method='lnn', units=256):
        super().__init__()
        # ResNet Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64, 128, stride=2),   # 32x256
            ResBlock(128, 128),
            ResBlock(128, 256, stride=2),  # 16x128
            ResBlock(256, 256),
            ResBlock(256, 512, stride=(2, 1)), # 8x128 (Reduced height, keep width)
            ResBlock(512, 512),
        )
        
        self.pos_enc = PositionalEncoding1D(512 * 8)
        input_dim = 512 * 8 
        
        if method == 'lnn':
            self.rnn = CfC(input_dim, units, proj_size=vocab_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_dim, units, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(units * 2, vocab_size)
        self.method = method

    def forward(self, x):
        features = self.backbone(x) # [B, 512, 8, W_final]
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2).reshape(b, w, c*h)
        features = self.pos_enc(features)
        
        if self.method == 'lnn':
            out, _ = self.rnn(features)
        else:
            out, _ = self.rnn(features)
            out = self.fc(out)
        return out

# ==========================================
# 4. Data Pipeline
# ==========================================
class IAMDataset(Dataset):
    def __init__(self, parquet_path, char_to_idx=None, augment=False):
        self.df = pd.read_parquet(parquet_path)
        t_list = [transforms.Grayscale(), transforms.Resize((64, 512))]
        if augment:
            t_list.extend([
                transforms.RandomAffine(degrees=1, shear=1, scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.1)
            ])
        t_list.extend([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.transform = transforms.Compose(t_list)
        
        if char_to_idx is None:
            chars = sorted(list(set("".join(self.df['text']))))
            self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
            self.char_to_idx['<PAD>'] = 0
        else:
            self.char_to_idx = char_to_idx
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = self.transform(Image.open(io.BytesIO(row['image']['bytes'])))
        label = torch.tensor([self.char_to_idx.get(c, 0) for c in row['text']], dtype=torch.long)
        return img, label, row['text']

def collate_fn(batch):
    imgs, labels, texts = zip(*batch)
    return torch.stack(imgs), torch.nn.utils.rnn.pad_sequence(labels, batch_first=True), texts

# ==========================================
# 5. Main Training Loop
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='iam')
    parser.add_argument('--method', type=str, choices=['lnn', 'lstm'], default='lnn')
    parser.add_argument('--units', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0003)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = IAMDataset("./data/iam/data/train.parquet", augment=True)
    test_ds = IAMDataset("./data/iam/data/test.parquet", char_to_idx=train_ds.char_to_idx)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    model = IAMModel(len(train_ds.char_to_idx), method=args.method, units=args.units).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    history = {'loss': [], 'cer': []}

    for epoch in range(args.epochs):
        model.train()
        t_loss = 0
        for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs).permute(1, 0, 2).log_softmax(2)
            input_lens = torch.full((imgs.size(0),), logits.size(0), dtype=torch.long)
            target_lens = torch.tensor([len(l[l != 0]) for l in labels], dtype=torch.long)
            loss = criterion(logits, labels, input_lens, target_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for imgs, _, texts in test_loader:
                logits = model(imgs.to(device)).permute(1, 0, 2)
                preds = ctc_greedy_decoder(logits, train_ds.idx_to_char)
                all_preds.extend(preds)
                all_trues.extend(texts)

        cer, wer = calculate_htr_metrics(all_trues, all_preds)
        scheduler.step()
        history['loss'].append(t_loss/len(train_loader))
        history['cer'].append(cer)

        print(f"\n[Epoch {epoch+1}] | Loss: {t_loss/len(train_loader):.4f} | CER: {cer:.4f} | WER: {wer:.4f}")
        
    # Save Curves
    plt.plot(history['loss'], label='Loss'); plt.title(f"{args.method} Loss"); plt.savefig(f"{args.method}_iam_loss.png"); plt.close()
    plt.plot(history['cer'], label='CER'); plt.title(f"{args.method} CER"); plt.savefig(f"{args.method}_iam_cer.png"); plt.close()

    # Save Sentences for Analysis
    with open(f"predictions_{args.method}.txt", "w") as f:
        for r, p in zip(all_trues, all_preds):
            f.write(f"REF: {r}\nPRED: {p}\n{'-'*20}\n")
    print(f"Results saved to predictions_{args.method}.txt")

if __name__ == "__main__":
    main()

