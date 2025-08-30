#!/usr/bin/env python3
"""
Script to normalize YOLO labels to proper format (0-1 range)
Converts pixel coordinates to normalized coordinates
"""

import os
import glob
from pathlib import Path
import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_labels(dataset_path="dataset"):
    """
    Normalize all label files in the dataset to YOLO format (0-1 range)
    """
    dataset_path = Path(dataset_path)
    
    # Process train, val, and test splits
    for split in ["train", "val", "test"]:
        labels_dir = dataset_path / split / "labels"
        images_dir = dataset_path / split / "images"
        
        if not labels_dir.exists():
            logging.warning(f"Labels directory {labels_dir} does not exist")
            continue
            
        logging.info(f"Processing {split} split...")
        
        # Get all label files
        label_files = list(labels_dir.glob("*.txt"))
        logging.info(f"Found {len(label_files)} label files in {split}")
        
        normalized_count = 0
        skipped_count = 0
        
        for label_file in label_files:
            # Find corresponding image file
            image_name = label_file.stem + ".jpg"
            image_path = images_dir / image_name
            
            if not image_path.exists():
                logging.warning(f"Image {image_path} not found for label {label_file}")
                continue
            
            # Read image to get dimensions
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    logging.warning(f"Could not read image {image_path}")
                    continue
                    
                img_height, img_width = img.shape[:2]
            except Exception as e:
                logging.error(f"Error reading image {image_path}: {e}")
                continue
            
            # Read and normalize labels
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                normalized_lines = []
                needs_update = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        logging.warning(f"Invalid label format in {label_file}: {line}")
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Check if already normalized (values should be between 0 and 1)
                    if x_center > 1 or y_center > 1 or width > 1 or height > 1:
                        # Normalize coordinates
                        x_center_norm = x_center / img_width
                        y_center_norm = y_center / img_height
                        width_norm = width / img_width
                        height_norm = height / img_height
                        
                        normalized_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
                        needs_update = True
                        normalized_count += 1
                    else:
                        # Already normalized, keep as is
                        normalized_lines.append(line)
                        skipped_count += 1
                
                # Write normalized labels back to file
                if needs_update:
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(normalized_lines) + '\n')
                        
            except Exception as e:
                logging.error(f"Error processing label file {label_file}: {e}")
                continue
        
        logging.info(f"{split} split: {normalized_count} labels normalized, {skipped_count} already normalized")
    
    logging.info("Label normalization complete!")

def verify_normalization(dataset_path="dataset"):
    """
    Verify that all labels are properly normalized
    """
    dataset_path = Path(dataset_path)
    
    for split in ["train", "val", "test"]:
        labels_dir = dataset_path / split / "labels"
        
        if not labels_dir.exists():
            continue
            
        label_files = list(labels_dir.glob("*.txt"))
        non_normalized = []
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) == 5:
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        if x_center > 1 or y_center > 1 or width > 1 or height > 1:
                            non_normalized.append(str(label_file))
                            break
                            
            except Exception as e:
                logging.error(f"Error checking {label_file}: {e}")
        
        if non_normalized:
            logging.warning(f"{split} split: {len(non_normalized)} files still not normalized")
            for file in non_normalized[:5]:  # Show first 5
                logging.warning(f"  {file}")
        else:
            logging.info(f"{split} split: All labels properly normalized")

if __name__ == "__main__":
    print("=== YOLO Label Normalization ===")
    
    # Normalize all labels
    normalize_labels()
    
    # Verify normalization
    print("\n=== Verification ===")
    verify_normalization()
    
    print("\nNormalization process complete!")
