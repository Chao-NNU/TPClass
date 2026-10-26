import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# =================================================================
# ⚙️ Hyperparameters & Configuration Paths
# =================================================================
DATA_DIR = "/data2/chao/transport/train_ready_dataset"
BATCH_SIZE = 1024
EPOCHS = 100            # Extended epochs to thoroughly extract value from feature space
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse

# =================================================================
# 📦 1. High-Performance In-Memory Dataset
# =================================================================
class TransportDataset(Dataset):
    def __init__(self, data_path, label_path, name="Dataset"):
        print(f"🚚 Loading large-scale feature matrix into memory [{name}]...")
        self.X = torch.from_numpy(np.load(data_path)).float()
        self.Y = torch.from_numpy(np.load(label_path)).float()
        print(f"✅ [{name}] Ingestion complete. Samples: {self.X.shape[0]}, Dimensions: {self.X.shape[1]}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# =================================================================
# 🧬 2. Residual Multilayer Perceptron Architecture (ResMLP-Classifier)
# =================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        return x + self.dropout(self.relu(self.bn(self.fc(x))))

class TransportClassifier(nn.Module):
    def __init__(self, input_dim=1283):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.res_block1 = ResidualBlock(512)
        self.res_block2 = ResidualBlock(512)
        self.output_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.res_block1(out)
        out = self.res_block2(out)
        return self.output_layer(out)

# =================================================================
# 🏃‍♂️ 3. Core Training & Validation Closed-Loop Engine
# =================================================================
def train(data_dir=None, output_path=None, batch_size=None, epochs=None, lr=None):
    """Train Phase 1 MLP on pre-extracted 1283-dim features.

    Args:
        data_dir: directory containing X_train.npy, Y_train.npy, X_val.npy, Y_val.npy
                   (default: hardcoded DATA_DIR)
        output_path: where to save transport_classifier_best.pth
                     (default: <data_dir>/transport_classifier_best.pth)
        batch_size, epochs, lr: override defaults
    """
    global DATA_DIR
    if data_dir is not None:
        DATA_DIR = data_dir
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "transport_classifier_best.pth")
    _bs = batch_size or BATCH_SIZE
    _ep = epochs or EPOCHS
    _lr = lr or LEARNING_RATE

    print(f"🔥 Target compute device locked on: {DEVICE}")
    print(f"  DATA_DIR  = {DATA_DIR}")
    print(f"  output    = {output_path}")
    print(f"  batch={_bs} epochs={_ep} lr={_lr}")

    train_dataset = TransportDataset(
        data_path=os.path.join(DATA_DIR, "X_train.npy"),
        label_path=os.path.join(DATA_DIR, "Y_train.npy"),
        name="Train Set"
    )
    val_dataset = TransportDataset(
        data_path=os.path.join(DATA_DIR, "X_val.npy"),
        label_path=os.path.join(DATA_DIR, "Y_val.npy"),
        name="Val Set"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=_bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=_bs, shuffle=False, num_workers=4, pin_memory=True)

    # Auto-infer input_dim from X_train (supports 1283 for ESM-2 650M and 2563 for ESM-2 3B)
    input_dim = int(train_dataset.X.shape[1])
    print(f"  [Model] Auto-inferred input_dim = {input_dim} (from X_train shape)")
    model = TransportClassifier(input_dim=input_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_ep)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0

    print("\n🚀 Initiating dual-track training pipeline...")
    for epoch in range(1, _ep + 1):
        # ----------------- 🛠️ TRAINING PHASE -----------------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{_ep}] Train")
        for step, (batch_x, batch_y) in enumerate(pbar):
            batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * batch_x.size(0)
            predicted = (outputs > 0.0).float()
            train_total += batch_y.size(0)
            batch_y_unsqueezed = batch_y.unsqueeze(1)
            train_correct += (predicted == batch_y_unsqueezed).sum().item()
            
            if step % 20 == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{(train_correct/train_total)*100:.2f}%"})
        
        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100
        
        # ----------------- 🔍 VALIDATION PHASE -----------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                
                val_loss += loss.item() * batch_x.size(0)
                predicted = (outputs > 0.0).float()
                val_total += batch_y.size(0)
                batch_y_unsqueezed = batch_y.unsqueeze(1)
                val_correct += (predicted == batch_y_unsqueezed).sum().item()
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        
        print(f"📈 Epoch [{epoch}/{_ep}] Metrics -> "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # Save historical best checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint_path = output_path
            torch.save(model.state_dict(), checkpoint_path)
            print(f"🔥 [Best Model] Validation accuracy reached a new peak: {best_val_acc:.2f}%! Weights saved to: {checkpoint_path}")

        # Routine safety fallback preservation every 20 epochs
        if epoch % 20 == 0 or epoch == _ep:
            final_path = f"{os.path.splitext(output_path)[0]}_epoch_{epoch}.pth"
            torch.save(model.state_dict(), final_path)
            print(f"💾 [Checkpoint] Periodic checkpoint state safely written to: {final_path}")
            
        print("-" * 80)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None,
                    help=f"Dir with X_train.npy, Y_train.npy, X_val.npy, Y_val.npy (default: {DATA_DIR})")
    ap.add_argument("--output_path", default=None,
                    help="Path to save transport_classifier_best.pth (default: <data_dir>/transport_classifier_best.pth)")
    ap.add_argument("--batch_size", type=int, default=None, help=f"Override BATCH_SIZE (default {BATCH_SIZE})")
    ap.add_argument("--epochs",     type=int, default=None, help=f"Override EPOCHS (default {EPOCHS})")
    ap.add_argument("--lr",         type=float, default=None, help=f"Override LEARNING_RATE (default {LEARNING_RATE})")
    args = ap.parse_args()
    train(data_dir=args.data_dir, output_path=args.output_path,
          batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)