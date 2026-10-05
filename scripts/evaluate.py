"""
evaluate.py

Computes ROC-AUC (the paper's primary metric), plus accuracy, sensitivity,
specificity, and a confusion matrix, for a trained MelanomaViT checkpoint
against a given dataset split (e.g. val/ or external_test/).

Reports results at three threshold "profiles" rather than a single 0.5
cutoff, since ISIC's class imbalance (74% nevus / 26% melanoma) means the
default threshold undersells melanoma sensitivity -- the clinically
important metric. See threshold_sweep.py for the full sweep this is based on.

Can be run standalone (adjust paths at the top for local vs Colab):
    python scripts/evaluate.py

Or imported and called directly in a notebook cell:
    from evaluate import run_evaluation
    run_evaluation(model, val_loader, device)
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import timm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Threshold profiles, based on the sweep run on the ISIC validation set.
# NOTE: these were tuned on THIS model/dataset. If you retrain or evaluate
# a different checkpoint, rerun threshold_sweep.py -- these numbers are not
# universal constants.
THRESHOLD_PROFILES = {
    "default (0.50)": 0.50,
    "balanced (0.65)": 0.65,       # strictly better than 0.50 on this val set:
                                     # higher melanoma sensitivity AND higher accuracy
    "high-sensitivity (0.80)": 0.80,  # prioritizes catching melanoma over avoiding
                                        # false positives on nevus
}


# ---------------------------------------------------------------------------
# Model definition (must match train_vit.py exactly)
# ---------------------------------------------------------------------------
class MelanomaViT(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_large_patch16_224", pretrained=True, num_classes=0
        )
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


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def get_predictions(model, loader, device):
    """
    Runs the model over a loader and returns true labels and predicted
    probabilities (post-sigmoid), needed for ROC-AUC.
    """
    model.eval()
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    return np.array(all_labels), np.array(all_probs)


def compute_metrics_at_threshold(y_true, y_probs, threshold):
    """
    Computes accuracy, sensitivity, specificity, and confusion matrix at a
    given threshold. Assumes ImageFolder's alphabetical label convention:
    melanoma=0, nevus=1.

    y_probs is the model's probability of class 1 (nevus). A HIGHER
    threshold requires more confidence before predicting nevus, so more
    borderline cases get called melanoma -- raising melanoma sensitivity
    at the cost of nevus sensitivity (specificity).
    """
    y_pred = (y_probs > threshold).astype(int)

    accuracy = (y_pred == y_true).mean()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # class 0 = melanoma: sensitivity here means "melanoma recall"
    melanoma_sensitivity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    # class 1 = nevus: this is "nevus recall" (sometimes called specificity
    # relative to melanoma-as-positive framing)
    nevus_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "melanoma_sensitivity": melanoma_sensitivity,
        "nevus_sensitivity": nevus_sensitivity,
        "confusion_matrix": cm,
    }


def plot_roc_curve(y_true, y_probs, auc, title="ROC Curve", save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curve saved to {save_path}")
    plt.show()


def run_evaluation(model, loader, device, class_names, title="Evaluation", save_path=None):
    """
    Full evaluation pipeline: get predictions, compute ROC-AUC (threshold-
    independent), then report accuracy/sensitivity/specificity at each
    threshold profile in THRESHOLD_PROFILES. Plots the ROC curve once.
    """
    print(f"Class order (index -> name): {list(enumerate(class_names))}")
    assert class_names[0] == "melanoma" and class_names[1] == "nevus", (
        "Class order doesn't match expected melanoma=0, nevus=1 convention. "
        "Sensitivity/specificity labels below would be mislabeled -- stop "
        "and investigate before trusting these numbers."
    )

    y_true, y_probs = get_predictions(model, loader, device)
    auc = roc_auc_score(y_true, y_probs)

    print(f"\n--- {title} ---")
    print(f"ROC-AUC (threshold-independent): {auc:.4f}\n")

    print(f"{'Profile':>24} | {'Thresh':>6} | {'Accuracy':>8} | {'Melanoma Sens.':>14} | {'Nevus Sens.':>11}")
    print("-" * 78)

    all_metrics = {"roc_auc": auc, "profiles": {}}

    for profile_name, threshold in THRESHOLD_PROFILES.items():
        m = compute_metrics_at_threshold(y_true, y_probs, threshold)
        all_metrics["profiles"][profile_name] = m
        print(f"{profile_name:>24} | {threshold:>6.2f} | {m['accuracy']:>8.4f} | "
              f"{m['melanoma_sensitivity']:>14.4f} | {m['nevus_sensitivity']:>11.4f}")

    # Print confusion matrix for the default profile only, to keep output readable
    default_cm = all_metrics["profiles"]["default (0.50)"]["confusion_matrix"]
    print(f"\nConfusion matrix (threshold=0.50):\n{default_cm}")

    print("\nNote: melanoma sensitivity = recall for class 0 (melanoma) = "
          "fraction of actual melanomas correctly caught.")
    print("Nevus sensitivity = recall for class 1 (nevus). Missing a melanoma "
          "(false negative) is typically far costlier clinically than a false "
          "positive on nevus, so higher-threshold profiles trade some accuracy "
          "for meaningfully better melanoma detection.")

    plot_roc_curve(y_true, y_probs, auc, title=title, save_path=save_path)

    return all_metrics


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    VAL_DIR = PROJECT_ROOT / "data" / "preprocessed" / "val"
    MODEL_PATH = PROJECT_ROOT / "models" / "vit_l16_posweight_best.pt"

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    val_dataset = datasets.ImageFolder(VAL_DIR, transform=get_eval_transform())
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = MelanomaViT(freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded checkpoint: {MODEL_PATH}")

    run_evaluation(
        model, val_loader, device,
        class_names=val_dataset.classes,
        title="ISIC 2019 Validation Set",
        save_path=PROJECT_ROOT / "models" / "roc_curve_val.png"
    )


if __name__ == "__main__":
    main()
