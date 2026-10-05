"""
train_baseline.py

Generalized version of train_vit.py that accepts a --model flag, so any
timm-supported backbone can be fine-tuned with the same architecture
(frozen backbone + 4-layer classification head) and training config used
for ViT-L/16 -- enabling direct comparison, matching the paper's
multi-model evaluation design (Section 2.2).

Local debug run (M1 Pro, MPS, tiny subset):
    python scripts/train_baseline.py --model resnet152 --subset 200 --epochs 2 --batch_size 8

Full training run (Colab, T4):
    python scripts/train_baseline.py --model resnet152 --epochs 15 --batch_size 32 --accum_steps 4
    python scripts/train_baseline.py --model vit_base_patch16_224 --epochs 15 --batch_size 32 --accum_steps 4

Supported --model values (timm names):
    resnet152
    densenet201
    efficientnet_b7
    convnext_xlarge
    vit_base_patch16_224
    vit_large_patch16_224   (same as train_vit.py, included for consistency)
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "preprocessed" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "preprocessed" / "val"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_MODELS = [
    "resnet152",
    "densenet201",
    "efficientnet_b7",
    "convnext_xlarge",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
]


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data transforms -- identical to train_vit.py, matches paper Section 2.3
# ---------------------------------------------------------------------------
def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model -- any timm backbone (frozen) + same 4-layer classification head
# Matches paper Section 2.2 / Figure 1's "same head for fair comparison"
# ---------------------------------------------------------------------------
class MelanomaClassifier(nn.Module):
    def __init__(self, model_name: str, freeze_backbone: bool = True):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0
        )  # num_classes=0 -> pooled features, no head

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features).squeeze(1)


# ---------------------------------------------------------------------------
# Training loop (identical logic to train_vit.py)
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images)
        loss = criterion(outputs, labels) / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=SUPPORTED_MODELS,
                         help="timm model name to fine-tune")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--subset", type=int, default=None,
                         help="If set, trains on only this many samples (for local debugging)")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true",
                         help="Resume from an existing checkpoint for this model, if found")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {args.model}")

    # -- Datasets --
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=get_train_transform())
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=get_val_transform())
    print(f"Classes: {train_dataset.classes}")

    if args.subset:
        train_indices = random.sample(range(len(train_dataset)), min(args.subset, len(train_dataset)))
        val_indices = random.sample(range(len(val_dataset)), min(args.subset // 4, len(val_dataset)))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        print(f"DEBUG MODE: using subset of {len(train_dataset)} train / {len(val_dataset)} val samples")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    # -- Class balance / pos_weight --
    # Handles both the full ImageFolder dataset and a Subset wrapper (debug mode).
    if hasattr(train_dataset, "targets"):
        target_labels = train_dataset.targets
    else:  # Subset: pull targets via the underlying dataset + indices
        target_labels = [train_dataset.dataset.targets[i] for i in train_dataset.indices]

    class_counts = Counter(target_labels)
    melanoma_count = class_counts[0]
    nevus_count = class_counts[1]
    pos_weight_value = melanoma_count / nevus_count
    print(f"Class counts -> melanoma: {melanoma_count}, nevus: {nevus_count}, "
          f"pos_weight: {pos_weight_value:.4f}")
    pos_weight_tensor = torch.tensor([pos_weight_value]).to(device)

    # -- Model --
    model = MelanomaClassifier(args.model, freeze_backbone=True).to(device)

    checkpoint_path = MODELS_DIR / f"{args.model}_best.pt"
    history_path = MODELS_DIR / f"{args.model}_training_history.json"

    best_val_acc = 0.0
    if args.resume and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Resumed from checkpoint: {checkpoint_path}")
        if history_path.exists():
            with open(history_path) as f:
                prior_history = json.load(f)
            if prior_history:
                best_val_acc = max(h["val_acc"] for h in prior_history)
                print(f"Carrying forward best_val_acc={best_val_acc:.4f} from prior history")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / Total params: {total_params:,}")

    # -- Loss, optimizer, scheduler (matches paper Section 2.3) --
    # pos_weight counteracts the nevus/melanoma class imbalance (see pos_weight_tensor above)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -- Training loop --
    history = []
    if args.resume and history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.accum_steps
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        print(f"[{args.model}] Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | {elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc, "elapsed_sec": elapsed
        })

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved ({checkpoint_path.name}, val_acc={val_acc:.4f})")

    print(f"\nTraining complete for {args.model}. Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
