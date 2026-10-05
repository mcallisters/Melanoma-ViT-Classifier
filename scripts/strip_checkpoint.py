"""
strip_checkpoint.py

Extracts just the classifier head weights from a full MelanomaViT checkpoint,
dropping the frozen ImageNet backbone (which is re-downloaded from timm at
load time anyway, so saving it is redundant). Shrinks a ~1.1GB checkpoint
down to a couple MB, small enough to commit directly to GitHub.

Usage:
    python scripts/strip_checkpoint.py models/vit_l16_best.pt
    python scripts/strip_checkpoint.py models/vit_l16_posweight_best.pt
"""

import sys
from pathlib import Path

import torch

def strip(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    full_state = torch.load(checkpoint_path, map_location="cpu")

    classifier_state = {
        k: v for k, v in full_state.items() if k.startswith("classifier.")
    }

    n_backbone = sum(1 for k in full_state if k.startswith("backbone."))
    print(f"Full checkpoint: {len(full_state)} keys "
          f"({n_backbone} backbone, {len(classifier_state)} classifier)")

    if not classifier_state:
        raise RuntimeError("No 'classifier.' keys found — check the checkpoint format.")

    out_path = checkpoint_path.with_name(checkpoint_path.stem + "_head.pt")
    torch.save(classifier_state, out_path)

    orig_mb = checkpoint_path.stat().st_size / 1e6
    new_mb = out_path.stat().st_size / 1e6
    print(f"Saved {out_path} ({new_mb:.2f} MB, down from {orig_mb:.1f} MB)")

if __name__ == "__main__":
    strip(sys.argv[1])