#!/usr/bin/env python3
"""
Script to fix labels that have extra class names in them
Removes words like "bear", "panda" from label files and normalizes coordinates
"""

import os
import glob
from pathlib import Path
import cv2
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fix_class_names_in_labels(dataset_path="dataset"):
    """
    Fix labels that have extra class names in them and normalize coordinates
    """
    dataset_path = Path(dataset_path)
    
    # Common class names that might appear in labels
    class_names_to_remove = ['bear', 'panda', 'tiger', 'lion', 'elephant', 'giraffe', 
                            'zebra', 'rhino', 'hippo', 'buffalo', 'deer', 'antelope',
                            'wolf', 'fox', 'coyote', 'jackal', 'hyena', 'leopard',
                            'cheetah', 'jaguar', 'lynx', 'bobcat', 'cougar', 'panther',
                            'monkey', 'gorilla', 'chimpanzee', 'orangutan', 'gibbon',
                            'sloth', 'koala', 'kangaroo', 'wallaby', 'wombat', 'platypus']
    
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
                    
                    # Check if we have extra class names in the label
                    if len(parts) >= 5:
                        # Try to find the class ID (should be the first number)
                        class_id = None
                        coordinates = []
                        
                        for part in parts:
                            try:
                                # If this is a number, it could be class_id or coordinate
                                num = float(part)
                                if class_id is None and num.is_integer() and 0 <= int(num) <= 35:
                                    # This looks like a class ID
                                    class_id = int(num)
                                else:
                                    # This is a coordinate
                                    coordinates.append(num)
                            except ValueError:
                                # This is not a number, skip it (it's probably a class name)
                                continue
                        
                        if class_id is not None and len(coordinates) >= 4:
                            x_center, y_center, width, height = coordinates[:4]
                            
                            # Check if coordinates need normalization
                            if x_center > 1 or y_center > 1 or width > 1 or height > 1:
                                # Normalize coordinates
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
                                # Already normalized, just fix the format
                                fixed_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                                needs_update = True
                        else:
                            logging.warning(f"Could not parse label in {label_file}: {line}")
                            fixed_lines.append(line)
                    else:
                        logging.warning(f"Invalid label format in {label_file}: {line}")
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
    Verify that all labels are now properly formatted and normalized
    """
    dataset_path = Path(dataset_path)
    
    for split in ["train", "val", "test"]:
        labels_dir = dataset_path / split / "labels"
        
        if not labels_dir.exists():
            continue
            
        label_files = list(labels_dir.glob("*.txt"))
        invalid_labels = []
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    
                    # Check if format is correct (should be exactly 5 parts: class_id + 4 coordinates)
                    if len(parts) != 5:
                        invalid_labels.append(f"{label_file}: wrong number of parts ({len(parts)})")
                        break
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Check if coordinates are normalized
                        if x_center > 1 or y_center > 1 or width > 1 or height > 1:
                            invalid_labels.append(f"{label_file}: non-normalized coordinates")
                            break
                            
                        # Check if class_id is valid
                        if class_id < 0 or class_id > 35:
                            invalid_labels.append(f"{label_file}: invalid class_id {class_id}")
                            break
                            
                    except ValueError:
                        invalid_labels.append(f"{label_file}: non-numeric values")
                        break
                            
            except Exception as e:
                logging.error(f"Error checking {label_file}: {e}")
        
        if invalid_labels:
            logging.warning(f"{split} split: {len(invalid_labels)} files still have issues")
            for issue in invalid_labels[:5]:  # Show first 5
                logging.warning(f"  {issue}")
        else:
            logging.info(f"{split} split: All labels properly formatted and normalized")

if __name__ == "__main__":
    print("=== Fix Class Names in Labels ===")
    
    # Fix the labels
    print("\n=== Fixing Labels ===")
    fix_class_names_in_labels()
    
    # Verify the fixes
    print("\n=== Verification ===")
    verify_fixes()
    
    print("\nLabel fixing process complete!")
