import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import numpy as np
from ncps.torch import CfC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==========================================
# 1. Model Architecture (CNN + LNN/LSTM)
# ==========================================
class MNISTClassifier(nn.Module):
    def __init__(self, method='lnn', units=128):
        super().__init__()
        self.method = method
        
        # Spatial Feature Extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # After 2 MaxPools, 28x28 becomes 7x7. Flattened = 64 * 7 * 7
        cnn_out_dim = 64 * 7 * 7
        
        if method == 'lnn':
            self.rnn = CfC(cnn_out_dim, units, proj_size=units, batch_first=True)
        else:
            self.rnn = nn.LSTM(cnn_out_dim, units, batch_first=True, num_layers=2)
            
        self.classifier = nn.Linear(units, 10)

    def forward(self, x):
        features = self.backbone(x) # [B, Flattened]
        features = features.unsqueeze(1) # Add time dimension: [B, 1, Flattened]
        
        out, _ = self.rnn(features)
        pooled = out[:, -1, :] # Take the last hidden state
        return self.classifier(pooled)

# ==========================================
# 2. Main Experiment Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['lnn', 'lstm'], default='lnn')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--units', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data_path', type=str, default="./data/mnist")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # download=True will download the data to args.data_path if it doesn't exist
    train_ds = torchvision.datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MNISTClassifier(method=args.method, units=args.units).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'test_acc': []}

    print(f"--- Starting {args.method.upper()} on Standard MNIST ---")
    print(f"Data directory: {args.data_path}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x.to(device))
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_trues.extend(y.numpy())

        acc = accuracy_score(all_trues, all_preds)
        history['train_loss'].append(total_loss/len(train_loader))
        history['test_acc'].append(acc)

        print(f"\n[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f}")
        print("-" * 50)

    # Visualization
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(history['train_loss']); plt.title(f"MNIST {args.method} Loss")
    plt.subplot(1, 2, 2); plt.plot(history['test_acc']); plt.title(f"MNIST {args.method} Accuracy")
    plt.savefig(f"mnist_{args.method}_curves.png")

    # Save Predictions
    with open(f"predictions_mnist_{args.method}.txt", "w") as f:
        f.write("True_Label, Predicted_Label\n")
        for t, p in zip(all_trues, all_preds):
            f.write(f"{t}, {p}\n")
    print(f"Results saved to predictions_mnist_{args.method}.txt")

if __name__ == "__main__":
    main()

