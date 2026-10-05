"""
preprocessing.py

Filters ISIC 2019 to melanoma (MEL) vs nevus (NV), performs a stratified
80/20 train/val split, and resizes+pads images to 224x224 while preserving
aspect ratio (matches the paper's preprocessing approach).

Can be run standalone:
    python scripts/preprocessing.py

Or imported for interactive use in a notebook:
    from preprocessing import load_labels, filter_melanoma_nevus, resize_and_pad
"""

import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths — adjust here if your folder names ever change
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "isic_2019"
IMAGES_DIR = RAW_DIR / "isic_2019_training_input"
LABELS_CSV = RAW_DIR / "isic_2019_training_groundtruth.csv"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

TARGET_SIZE = 224
VAL_FRACTION = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Step 1: Load and filter labels
# ---------------------------------------------------------------------------
def load_labels(csv_path: Path = LABELS_CSV) -> pd.DataFrame:
    """Load the ISIC ground truth CSV."""
    df = pd.read_csv(csv_path)
    return df


def filter_melanoma_nevus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where MEL==1 or NV==1 (binary melanoma-vs-nevus task,
    matching the paper). Adds a single 'label' column: 'melanoma' or 'nevus'.
    """
    filtered = df[(df["MEL"] == 1) | (df["NV"] == 1)].copy()
    filtered["label"] = filtered["MEL"].apply(
        lambda x: "melanoma" if x == 1 else "nevus"
    )
    return filtered[["image", "label"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: Stratified train/val split
# ---------------------------------------------------------------------------
def stratified_split(
    df: pd.DataFrame, val_fraction: float = VAL_FRACTION, seed: int = RANDOM_STATE
):
    """Stratified split preserving class balance in both sets."""
    train_df, val_df = train_test_split(
        df, test_size=val_fraction, stratify=df["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3: Resize + pad (preserve aspect ratio, black padding to square)
# ---------------------------------------------------------------------------
def resize_and_pad(image: Image.Image, target_size: int = TARGET_SIZE) -> Image.Image:
    """
    Resize an image so its longest side equals target_size, then pad the
    shorter side with black to make it square. Matches the paper's approach
    of avoiding distortion from naive stretch-resize.
    """
    image = image.convert("RGB")
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))
    return padded


# ---------------------------------------------------------------------------
# Step 4: Write a split (train or val) to disk in class-per-folder layout
# ---------------------------------------------------------------------------
def write_split(df: pd.DataFrame, split_name: str, images_dir: Path = IMAGES_DIR):
    """
    Reads each image referenced in df, resizes+pads it, and writes it to
    data/preprocessed/{split_name}/{label}/{filename}.jpg
    """
    out_root = PREPROCESSED_DIR / split_name
    for label in df["label"].unique():
        (out_root / label).mkdir(parents=True, exist_ok=True)

    skipped = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        src_path = images_dir / f"{row['image']}.jpg"
        if not src_path.exists():
            skipped.append(row["image"])
            continue
        try:
            img = Image.open(src_path)
            processed = resize_and_pad(img)
            dst_path = out_root / row["label"] / f"{row['image']}.jpg"
            processed.save(dst_path, quality=95)
        except Exception as e:
            print(f"Failed on {row['image']}: {e}")
            skipped.append(row["image"])

    if skipped:
        print(f"\n{len(skipped)} images skipped in {split_name} (missing or corrupt).")
    return skipped


# ---------------------------------------------------------------------------
# Step 5: Quick sanity-check helper (for notebook use)
# ---------------------------------------------------------------------------
def summarize_split(df: pd.DataFrame, name: str):
    counts = df["label"].value_counts()
    print(f"{name}: {len(df)} total | melanoma={counts.get('melanoma', 0)} "
          f"nevus={counts.get('nevus', 0)}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("Loading labels...")
    df = load_labels()

    print("Filtering to melanoma vs nevus...")
    filtered = filter_melanoma_nevus(df)
    summarize_split(filtered, "Full filtered set")

    print("\nSplitting train/val (stratified 80/20)...")
    train_df, val_df = stratified_split(filtered)
    summarize_split(train_df, "Train")
    summarize_split(val_df, "Val")

    print("\nProcessing train split...")
    write_split(train_df, "train")

    print("\nProcessing val split...")
    write_split(val_df, "val")

    print("\nDone. Preprocessed data written to:", PREPROCESSED_DIR)


if __name__ == "__main__":
    main()