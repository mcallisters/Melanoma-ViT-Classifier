"""
threshold_sweep.py

Sweeps classification thresholds against the val set to show the
sensitivity/specificity tradeoff at each one, so you can pick a threshold
that better balances catching melanoma (sensitivity for class 0) against
false positives, rather than defaulting to 0.5.

Run after evaluate.py has confirmed the model loads correctly:
    python scripts/threshold_sweep.py
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from evaluate import MelanomaViT, get_eval_transform, get_predictions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_DIR = PROJECT_ROOT / "data" / "preprocessed" / "val"
MODEL_PATH = PROJECT_ROOT / "models" / "vit_l16_posweight_best.pt"


def sweep_thresholds(y_true, y_probs, thresholds):
    """
    For each threshold, compute sensitivity/specificity treating class 0
    (melanoma) as the positive class of interest.

    Recall: ImageFolder assigned melanoma=0, nevus=1. y_probs is the
    model's probability of class 1 (nevus). So a LOWER threshold means
    we require LESS confidence in "nevus" before predicting nevus --
    i.e. we predict melanoma more readily, raising melanoma sensitivity.
    """
    print(f"{'Threshold':>10} | {'Melanoma Sens.':>15} | {'Nevus Sens.':>12} | {'Accuracy':>9}")
    print("-" * 55)

    results = []
    for t in thresholds:
        y_pred = (y_probs > t).astype(int)  # 1 = predicted nevus, 0 = predicted melanoma

        # Melanoma is class 0: sensitivity = correctly caught melanomas / actual melanomas
        melanoma_mask = (y_true == 0)
        melanoma_sensitivity = (y_pred[melanoma_mask] == 0).mean()

        # Nevus is class 1
        nevus_mask = (y_true == 1)
        nevus_sensitivity = (y_pred[nevus_mask] == 1).mean()

        accuracy = (y_pred == y_true).mean()

        results.append({
            "threshold": t,
            "melanoma_sensitivity": melanoma_sensitivity,
            "nevus_sensitivity": nevus_sensitivity,
            "accuracy": accuracy,
        })

        print(f"{t:>10.2f} | {melanoma_sensitivity:>15.4f} | {nevus_sensitivity:>12.4f} | {accuracy:>9.4f}")

    return results


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}\n")

    val_dataset = datasets.ImageFolder(VAL_DIR, transform=get_eval_transform())
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    print(f"Class order: {list(enumerate(val_dataset.classes))}\n")

    model = MelanomaViT(freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true, y_probs = get_predictions(model, val_loader, device)

    thresholds = np.arange(0.1, 0.95, 0.05)
    results = sweep_thresholds(y_true, y_probs, thresholds)

    print("\nNote: at threshold=0.50 (default), this should roughly match")
    print("evaluate.py's reported sensitivity=0.9709 (nevus) / specificity=0.6055 (melanoma).")
    print("\nLower thresholds trade nevus sensitivity for melanoma sensitivity --")
    print("clinically, missing melanoma (false negative) is typically far costlier")
    print("than a false-positive nevus flag, so a lower threshold is often preferred")
    print("despite lower overall accuracy.")


if __name__ == "__main__":
    main()
