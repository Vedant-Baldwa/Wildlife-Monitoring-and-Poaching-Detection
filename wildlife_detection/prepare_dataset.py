#!/usr/bin/env python3
"""
prepare_wildlife_dataset.py

Usage:
    python prepare_wildlife_dataset.py

Edits to make:
 - Set SRC_DIR to the folder that contains the 4 class subfolders (e.g. "/path/to/African Wildlife")
 - Set DST_DIR to where you want dataset_2 created (default: ./dataset_2)
 - Optionally adjust split ratios and file extensions to accept.

What it does:
 - Finds class folders (uses the order specified in CLASS_NAMES if provided,
   otherwise alphabetic order).
 - For each image file it looks for a matching .txt label file (same stem).
 - Copies image + label into train/val/test images+labels with new unique names
   to avoid colliding stems across classes.
 - Writes classes.txt and data_wildlife.yaml compatible with your training script.
"""

import os
import shutil
from pathlib import Path
import random
import argparse

# ------------- Config -------------
# Change these paths if needed:
SRC_DIR = Path(r'C:\Users\vedan\Machine Learning\SWE PROJECT\African_Wildlife')     # folder that contains the 4 class subfolders
DST_DIR = Path(r"C:\Users\vedan\Machine Learning\SWE PROJECT\dataset_2")            # output dataset root used by your training script

# If your class folders are exactly these names, set them here in the desired label order:
CLASS_NAMES = ["buffalo", "elephant", "rhino", "zebra"]  # order -> label indices 0..3

# Acceptable image extensions (lowercase)
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

# Split ratios (must sum <= 1.0; remainder goes to train)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
SEED = 42

# ------------- End config -------------
random.seed(SEED)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_image_files(folder: Path):
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXTS and p.is_file()]

def copy_pair(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path):
    shutil.copy2(src_img, dst_img) 
    shutil.copy2(src_lbl, dst_lbl)

def main():
    print("SRC_DIR:", SRC_DIR.resolve())
    print("DST_DIR:", DST_DIR.resolve())

    if not SRC_DIR.exists():
        raise SystemExit(f"Source directory {SRC_DIR} does not exist. Edit SRC_DIR in the script.")

    # Validate classes exist, fallback to subfolders if CLASS_NAMES not matching
    class_folders = []
    for cname in CLASS_NAMES:
        candidate = SRC_DIR / cname
        if candidate.exists() and candidate.is_dir():
            class_folders.append((cname, candidate))
        else:
            raise SystemExit(f"Expected class folder not found: {candidate}")

    # Create destination structure
    train_img = DST_DIR / "train" / "images"
    train_lbl = DST_DIR / "train" / "labels"
    val_img = DST_DIR / "val" / "images"
    val_lbl = DST_DIR / "val" / "labels"
    test_img = DST_DIR / "test" / "images"
    test_lbl = DST_DIR / "test" / "labels"

    for p in [train_img, train_lbl, val_img, val_lbl, test_img, test_lbl]:
        ensure_dir(p)

    # Walk each class, collect matching pairs
    all_stats = {}
    global_index = 0

    for label_idx, (cname, cfolder) in enumerate(class_folders):
        print(f"\nProcessing class {label_idx} -> '{cname}' from {cfolder}")
        images = find_image_files(cfolder)
        pairs = []
        for img in images:
            txt = img.with_suffix(".txt")
            if not txt.exists():
                # also try .TXT uppercase
                txt_up = img.with_suffix(".TXT")
                if txt_up.exists():
                    txt = txt_up
                else:
                    print(f"  WARNING: label file not found for image {img.name} — skipping")
                    continue
            pairs.append((img, txt))

        n = len(pairs)
        if n == 0:
            print(f"  WARNING: no valid image-label pairs found for class {cname}")
            continue

        # shuffle then split
        random.shuffle(pairs)
        n_test = int(round(n * TEST_RATIO))
        n_val = int(round(n * VAL_RATIO))
        n_train = n - n_val - n_test
        if n_train <= 0:
            # fallback: ensure at least 1 train sample
            n_train = max(1, n - n_test - n_val)
        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:n_train + n_val]
        test_pairs = pairs[n_train + n_val:]

        print(f"  found {n} pairs -> train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

        # Copy pairs with renaming to avoid stem collision: use e.g. "0_Buffalo_0001.jpg"
        def copy_list(pairs_list, dst_images_folder: Path, dst_labels_folder: Path, subset_name: str):
            nonlocal global_index
            for i, (imgp, lblp) in enumerate(pairs_list, start=1):
                # new base name
                new_stem = f"{label_idx}_{cname}_{i:04d}"
                dst_img = dst_images_folder / (new_stem + imgp.suffix.lower())
                dst_lbl = dst_labels_folder / (new_stem + ".txt")
                copy_pair(imgp, lblp, dst_img, dst_lbl)
                global_index += 1

        copy_list(train_pairs, train_img, train_lbl, "train")
        copy_list(val_pairs, val_img, val_lbl, "val")
        copy_list(test_pairs, test_img, test_lbl, "test")

        all_stats[cname] = {"total": n, "train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)}

    # Write classes.txt
    classes_txt = DST_DIR / "classes.txt"
    with open(classes_txt, "w", encoding="utf8") as f:
        for name in CLASS_NAMES:
            f.write(f"{name}\n")
    print("\nWrote classes.txt:", classes_txt)

    # Create data yaml file (absolute paths so ultralytics can use it directly)
    data_yaml = DST_DIR / "data_wildlife.yaml"
    yaml_text = (
        f"train: {str((DST_DIR / 'train' / 'images').resolve())}\n"
        f"val:   {str((DST_DIR / 'val' / 'images').resolve())}\n"
        f"test:  {str((DST_DIR / 'test' / 'images').resolve())}\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    data_yaml.write_text(yaml_text)
    print("Wrote data_wildlife.yaml:", data_yaml)

    # Summary stats
    print("\nSummary per-class:")
    total = {"total": 0, "train": 0, "val": 0, "test": 0}
    for cname, s in all_stats.items():
        print(f"  {cname}: total={s['total']}, train={s['train']}, val={s['val']}, test={s['test']}")
        for k in total:
            total[k] += s[k]
    print("\nOverall totals:", total)
    print("\nDone. You can now point your training script's DATA_ROOT to:", DST_DIR.resolve())
    print("Example: set DATA_ROOT = Path(r'{}')".format(DST_DIR.resolve()))

if __name__ == "__main__":
    main()
