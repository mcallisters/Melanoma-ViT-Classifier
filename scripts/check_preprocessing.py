import sys
sys.path.append("scripts")
from preprocessing import load_labels, filter_melanoma_nevus, stratified_split, resize_and_pad, summarize_split, IMAGES_DIR
from PIL import Image

# Step 1 - load and filter (fast, no image I/O yet)
df = load_labels()
filtered = filter_melanoma_nevus(df)
summarize_split(filtered, "Full filtered set")

# Step 2 - split
train_df, val_df = stratified_split(filtered)
summarize_split(train_df, "Train")
summarize_split(val_df, "Val")

# Step 3 - visually inspect a few samples' resize+pad output
sample_rows = filtered.sample(5, random_state=1)
for _, row in sample_rows.iterrows():
    src_path = IMAGES_DIR / f"{row['image']}.jpg"
    img = Image.open(src_path)
    print(f"{row['image']} ({row['label']}) - original size: {img.size}")
    padded = resize_and_pad(img)
    padded.save(f"sample_check_{row['image']}.jpg")  # saves to your project root for viewing