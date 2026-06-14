"""
utils/prepare_dataset.py
Fixes the val split — moves 10% of training images into val.
Run this ONCE before training.
"""
import os, shutil, random

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "chest_xray")
CLASSES   = ["NORMAL", "PNEUMONIA"]
VAL_SPLIT = 0.10

random.seed(42)

for cls in CLASSES:
    train_dir = os.path.join(DATA_DIR, "train", cls)
    val_dir   = os.path.join(DATA_DIR, "val",   cls)
    os.makedirs(val_dir, exist_ok=True)

    all_files = [f for f in os.listdir(train_dir)
                 if f.lower().endswith((".jpeg", ".jpg", ".png"))]

    existing_val = os.listdir(val_dir)
    if len(existing_val) > 20:
        print(f"[{cls}] Val already has {len(existing_val)} images, skipping.")
        continue

    n_move = int(len(all_files) * VAL_SPLIT)
    to_move = random.sample(all_files, n_move)

    for fname in to_move:
        shutil.move(
            os.path.join(train_dir, fname),
            os.path.join(val_dir,   fname)
        )
    print(f"[{cls}] Moved {n_move} images to val/  "
          f"({len(all_files) - n_move} remain in train)")

print("\n✅ Dataset prepared.")