#!/usr/bin/env python3
"""
Script to fix mixed label formats in the dataset
Detects which labels are already normalized vs which need normalization
Only fixes the non-normalized ones to preserve already correct labels
"""

import os
import glob
from pathlib import Path
import cv2
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_label_distribution(dataset_path="dataset"):
    """
    Analyze the distribution of normalized vs non-normalized labels
    """
    dataset_path = Path(dataset_path)
    
    for split in ["train", "val", "test"]:
        labels_dir = dataset_path / split / "labels"
        
        if not labels_dir.exists():
            continue
            
        label_files = list(labels_dir.glob("*.txt"))
        normalized_count = 0
        non_normalized_count = 0
        mixed_count = 0
        total_labels = 0
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                file_has_normalized = False
                file_has_non_normalized = False
                
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
                        
                        # Check if this label is normalized
                        if x_center <= 1 and y_center <= 1 and width <= 1 and height <= 1:
                            file_has_normalized = True
                        else:
                            file_has_non_normalized = True
                
                if file_has_normalized and file_has_non_normalized:
                    mixed_count += 1
                elif file_has_normalized:
                    normalized_count += 1
                elif file_has_non_normalized:
                    non_normalized_count += 1
                    
            except Exception as e:
                logging.error(f"Error analyzing {label_file}: {e}")
        
        logging.info(f"{split} split analysis:")
        logging.info(f"  Total files: {len(label_files)}")
        logging.info(f"  Already normalized: {normalized_count}")
        logging.info(f"  Need normalization: {non_normalized_count}")
        logging.info(f"  Mixed format files: {mixed_count}")

def fix_mixed_labels(dataset_path="dataset"):
    """
    Fix only the non-normalized labels while preserving already normalized ones
    """
    dataset_path = Path(dataset_path)
    
    for split in ["train", "val", "test"]:
        labels_dir = dataset_path / split / "labels"
        images_dir = dataset_path / split / "images"
        
        if not labels_dir.exists():
            logging.warning(f"Labels directory {labels_dir} does not exist")
            continue
            
        logging.info(f"Processing {split} split...")
        
        label_files = list(labels_dir.glob("*.txt"))
        fixed_count = 0
        preserved_count = 0
        error_count = 0
        
        for label_file in label_files:
            # Find corresponding image file
            image_name = label_file.stem + ".jpg"
            image_path = images_dir / image_name
            
            if not image_path.exists():
                logging.warning(f"Image {image_path} not found for label {label_file}")
                error_count += 1
                continue
            
            # Read image to get dimensions
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    logging.warning(f"Could not read image {image_path}")
                    error_count += 1
                    continue
                    
                img_height, img_width = img.shape[:2]
            except Exception as e:
                logging.error(f"Error reading image {image_path}: {e}")
                error_count += 1
                continue
            
            # Read and fix labels
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                fixed_lines = []
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
                    
                    # Check if this specific label needs normalization
                    if x_center > 1 or y_center > 1 or width > 1 or height > 1:
                        # This label needs normalization
                        x_center_norm = x_center / img_width
                        y_center_norm = y_center / img_height
                        width_norm = width / img_width
                        height_norm = height / img_height
                        
                        # Validate normalized values
                        if (0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1 and 
                            0 <= width_norm <= 1 and 0 <= height_norm <= 1):
                            fixed_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
                            needs_update = True
                        else:
                            logging.warning(f"Normalized values out of bounds in {label_file}: {x_center_norm:.3f}, {y_center_norm:.3f}, {width_norm:.3f}, {height_norm:.3f}")
                            # Keep original line if normalization produces invalid values
                            fixed_lines.append(line)
                    else:
                        # This label is already normalized, keep as is
                        fixed_lines.append(line)
                
                # Write fixed labels back to file
                if needs_update:
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(fixed_lines) + '\n')
                    fixed_count += 1
                else:
                    preserved_count += 1
                        
            except Exception as e:
                logging.error(f"Error processing label file {label_file}: {e}")
                error_count += 1
                continue
        
        logging.info(f"{split} split: {fixed_count} files fixed, {preserved_count} preserved, {error_count} errors")
    
    logging.info("Label fixing complete!")

def verify_fixes(dataset_path="dataset"):
    """
    Verify that all labels are now properly normalized
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
    print("=== Mixed Label Format Fix ===")
    
    # First analyze the current state
    print("\n=== Analysis ===")
    analyze_label_distribution()
    
    # Fix the mixed labels
    print("\n=== Fixing Labels ===")
    fix_mixed_labels()
    
    # Verify the fixes
    print("\n=== Verification ===")
    verify_fixes()
    
    print("\nLabel fixing process complete!")
