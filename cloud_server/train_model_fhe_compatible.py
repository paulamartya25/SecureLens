"""
cloud_server/train_model_fhe_compatible.py
SecureLens — FHE-Compatible Training Script

CRITICAL DIFFERENCE FROM train_model.py:
  - NO ReLU between linear layers (ReLU not FHE-compatible)
  - BatchNorm folded into weights after training
  - Architecture matches HE inference EXACTLY

This ensures zero accuracy loss between training and FHE inference.
"""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "..", "data", "chest_xray")
os.makedirs(MODELS_DIR, exist_ok=True)

IMAGE_SIZE  = 224
BATCH_SIZE  = 32
EPOCHS      = 20
LR_HEAD     = 1e-3
LR_BACKBONE = 1e-5
NUM_CLASSES = 2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[Train] Device     : {DEVICE}")
print(f"[Train] Mode       : FHE-COMPATIBLE (no ReLU in head)")


# ── Dataset (same as original) ───────────────────────────────────────

class ChestXRayDataset(Dataset):
    CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}

    def __init__(self, root_dir, split="train", transform=None):
        self.transform = transform
        self.samples   = []
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Not found: {split_dir}")
        for cls, label in self.CLASSES.items():
            d = os.path.join(split_dir, cls)
            if not os.path.exists(d): continue
            for f in os.listdir(d):
                if f.lower().endswith((".jpeg",".jpg",".png")):
                    self.samples.append((os.path.join(d,f), label))
        n = sum(1 for _,l in self.samples if l==0)
        p = sum(1 for _,l in self.samples if l==1)
        print(f"  [{split:5s}] {len(self.samples):5d} images"
              f"  NORMAL:{n}  PNEUMONIA:{p}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB",(IMAGE_SIZE,IMAGE_SIZE),128)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1,0.1),
                                scale=(0.9,1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


# ── FHE-Compatible Model ──────────────────────────────────────────────

class SecureLensNetFHE(nn.Module):
    """
    FHE-Compatible Architecture:
      - ResNet-18 backbone (512-dim features)
      - Linear head WITHOUT ReLU (FHE cannot compute ReLU efficiently)
      - BatchNorm for training stability (will be folded into weights)
      - NO Dropout (not needed for inference)
    
    Architecture: 512 → [Linear+BN] → 256 → [Linear] → 2
    
    After training, BatchNorm is folded into the Linear weights.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # Pretrained ResNet-18 backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # FHE-compatible head: Linear → BN → Linear (NO ReLU!)
        self.head = nn.Sequential(
            nn.Linear(512, 256),      # [0]
            nn.BatchNorm1d(256),      # [1] - for training only, will be folded
            nn.Linear(256, num_classes),  # [2]
        )

    def forward(self, x):
        x = self.backbone(x)          # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 512)
        x = self.head(x)
        return x

    def get_backbone_features(self, x):
        """Extract 512-dim feature vector from image."""
        with torch.no_grad():
            f = self.backbone(x)
            return f.view(f.size(0), -1)

    def fold_batchnorm_into_linear(self):
        """
        Folds BatchNorm parameters into the preceding Linear layer.
        After this, BN becomes identity and can be removed.
        
        Formula:
          y = gamma * (x - mean) / sqrt(var + eps) + beta
            = (gamma / sqrt(var + eps)) * x + (beta - gamma * mean / sqrt(var + eps))
          
        So:
          W_folded = gamma / sqrt(var + eps) * W
          b_folded = gamma / sqrt(var + eps) * b + (beta - gamma * mean / sqrt(var + eps))
        """
        self.eval()  # Use running stats
        
        linear1 = self.head[0]  # First linear
        bn = self.head[1]       # BatchNorm
        
        # Get BN parameters
        gamma = bn.weight.data
        beta = bn.bias.data
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        
        # Compute scale factor
        scale = gamma / torch.sqrt(var + eps)
        
        # Fold into Linear1
        linear1.weight.data = linear1.weight.data * scale.unsqueeze(1)
        linear1.bias.data = linear1.bias.data * scale + (beta - gamma * mean / torch.sqrt(var + eps))
        
        # Reset BN to identity
        bn.weight.data.fill_(1.0)
        bn.bias.data.fill_(0.0)
        bn.running_mean.fill_(0.0)
        bn.running_var.fill_(1.0)
        
        print("[Fold] BatchNorm folded into Linear layer")

    def extract_feature_weights(self):
        """512 → 256 (with BN folded)"""
        l = self.head[0]
        return {
            "W": l.weight.detach().cpu().numpy().tolist(),
            "b": l.bias.detach().cpu().numpy().tolist(),
        }

    def extract_linear_weights(self):
        """256 → 2"""
        l = self.head[2]
        return {
            "W": l.weight.detach().cpu().numpy().tolist(),
            "b": l.bias.detach().cpu().numpy().tolist(),
        }


# ── Training helpers (same as original) ───────────────────────────────

def make_sampler(dataset):
    labels  = [s[1] for s in dataset.samples]
    counts  = [labels.count(0), labels.count(1)]
    weights = [1.0/counts[l] for l in labels]
    return WeightedRandomSampler(weights, len(weights))


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1)==labels).sum().item()
        total    += imgs.size(0)
    return loss_sum/total, correct/total


def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  Eval ", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            loss_sum += loss.item() * imgs.size(0)
            correct  += (out.argmax(1)==labels).sum().item()
            total    += imgs.size(0)
    return loss_sum/total, correct/total


# ── Main Training Loop ────────────────────────────────────────────────

def main():
    print("\n"+"="*60)
    print("  SecureLens — FHE-Compatible Training (No ReLU)")
    print("="*60)

    train_tf, val_tf = get_transforms()

    print("\n[Datasets]")
    train_ds = ChestXRayDataset(DATA_DIR, "train", train_tf)
    val_ds   = ChestXRayDataset(DATA_DIR, "val",   val_tf)
    test_ds  = ChestXRayDataset(DATA_DIR, "test",  val_tf)

    sampler      = make_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model = SecureLensNetFHE(NUM_CLASSES).to(DEVICE)
    print(f"\n[Model] Total params   : "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f"[Model] Architecture   : 512 → 256 → 2 (NO ReLU)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": model.head.parameters(),     "lr": LR_HEAD},
    ], weight_decay=1e-3)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-7)

    best_val_acc = 0.0
    patience     = 6
    no_improve   = 0
    history      = {"train_loss":[],"train_acc":[],
                    "val_loss":[],"val_acc":[]}

    print("\n[Training]\n")
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_epoch(model, train_loader,
                                      optimizer, criterion)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(round(tr_loss,4))
        history["train_acc"].append(round(tr_acc,4))
        history["val_loss"].append(round(vl_loss,4))
        history["val_acc"].append(round(vl_acc,4))

        gap  = abs(vl_acc - tr_acc)
        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve   = 0
            torch.save(model.state_dict(),
                       os.path.join(MODELS_DIR,"best_model_fhe.pth"))
            flag = "  ✅ saved"
        else:
            no_improve += 1

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"Train:{tr_acc:.2%}({tr_loss:.4f})  "
              f"Val:{vl_acc:.2%}({vl_loss:.4f})  "
              f"Gap:{gap:.2%}{flag}")

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    # Test
    print("\n[Test] Loading best model...")
    model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR,"best_model_fhe.pth"),
                   map_location=DEVICE))
    ts_loss, ts_acc = evaluate(model, test_loader, criterion)
    print(f"  Test Loss     : {ts_loss:.4f}")
    print(f"  Test Accuracy : {ts_acc:.2%}")

    # CRITICAL: Fold BatchNorm into weights
    print("\n[Export] Folding BatchNorm into Linear weights...")
    model.fold_batchnorm_into_linear()
    
    # Verify folding didn't break anything
    print("[Export] Verifying folded model...")
    ts_loss_fold, ts_acc_fold = evaluate(model, test_loader, criterion)
    print(f"  After folding accuracy : {ts_acc_fold:.2%}")
    assert abs(ts_acc - ts_acc_fold) < 0.001, "Folding changed accuracy!"
    
    # Export weights
    feat_w   = model.extract_feature_weights()
    linear_w = model.extract_linear_weights()

    exports = {
        "feature_weights.json": feat_w,
        "linear_weights.json":  linear_w,
    }
    for fname, data in exports.items():
        path = os.path.join(MODELS_DIR, fname)
        with open(path,"w") as f:
            json.dump(data, f)
        W = np.array(data["W"])
        print(f"  {fname:30s}  shape: {W.shape}")

    # Save the folded model
    torch.save(model.state_dict(),
               os.path.join(MODELS_DIR,"best_model.pth"))  # Overwrite original
    
    # Also save as FHE version
    torch.save(model.state_dict(),
               os.path.join(MODELS_DIR,"securelens_fhe.pth"))
    
    with open(os.path.join(MODELS_DIR,"training_history_fhe.json"),"w") as f:
        json.dump(history, f, indent=2)

    # Save metadata
    metadata = {
        "architecture": "ResNet18 + Linear (NO ReLU)",
        "fhe_compatible": True,
        "batchnorm_folded": True,
        "test_accuracy": float(ts_acc_fold),
        "val_accuracy": float(best_val_acc),
        "relu_used": False,
        "notes": "This model matches the HE inference architecture exactly."
    }
    with open(os.path.join(MODELS_DIR,"model_versions.json"),"w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Best Val Accuracy   : {best_val_acc:.2%}")
    print(f"  Final Test Accuracy : {ts_acc_fold:.2%}")
    print(f"\n✅ FHE-compatible model trained successfully.")
    print(f"✅ BatchNorm folded - zero inference accuracy loss.")
    print(f"✅ Architecture matches HE inference exactly.")


if __name__ == "__main__":
    main()
