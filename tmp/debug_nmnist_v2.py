import torch
import torch.nn as nn
from ncps.torch import CfC

# Optimized Model for Event Data
class NMNIST_LNN(nn.Module):
    def __init__(self, input_size=4, units=256, num_classes=10):
        super().__init__()
        # Use CfC with 'backbone_layers' for better feature extraction
        self.rnn = CfC(input_size, units, proj_size=num_classes, batch_first=True)

    def forward(self, x):
        # x shape: [batch, seq_len, 4]
        out, _ = self.rnn(x)
        # Instead of taking the last state, use MEAN POOLING over time
        # This is critical for event-based data which is very sparse
        return torch.mean(out, dim=1)

def train_step():
    device = torch.device("cuda")
    model = NMNIST_LNN().to(device)
    # Lower LR and weight decay for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print("Running Stabilized N-MNIST Training...")
    for epoch in range(1, 6):
        # Synthetic event data for debug: [Batch 16, Seq 300, Feat 4]
        x = torch.randn(16, 300, 4).to(device)
        y = torch.randint(0, 10, (16,)).to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        # Gradient Clipping to prevent spikes (look at your Epoch 3 spike)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_step()

