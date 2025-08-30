#!/usr/bin/env python3
"""
fix_african_wildlife_labels.py

This script fixes the African Wildlife dataset labels by converting numeric indices
to text labels. The current format uses numeric indices (0, 1, 2, 3) but we want
text labels like "Buffalo", "Elephant", "Rhinoceros", "Zebra".

Usage:
  python fix_african_wildlife_labels.py
"""

import os
import glob
from pathlib import Path

# Configuration
AFRICAN_WILDLIFE_ROOT = "African Wildlife"

# Mapping from numeric indices to text labels
# Based on the folder structure and common African wildlife
NUMERIC_TO_TEXT_MAPPING = {
    "0": "Buffalo",
    "1": "Elephant", 
    "2": "Rhinoceros",
    "3": "Zebra"
}

def fix_label_file(label_path, animal_class):
    """Fix a single label file by converting numeric indices to text labels"""
    if not os.path.exists(label_path):
        return False
    
    updated = False
    new_lines = []
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            # Check if first part is a numeric index
            class_index = parts[0]
            
            if class_index in NUMERIC_TO_TEXT_MAPPING:
                # Convert numeric index to text label
                text_label = NUMERIC_TO_TEXT_MAPPING[class_index]
                
                # Keep the coordinates as they are (x_center, y_center, width, height)
                if len(parts) == 5:  # class x y w h
                    x_center, y_center, width, height = parts[1:5]
                    new_line = f"{text_label} {x_center} {y_center} {width} {height}"
                    new_lines.append(new_line)
                    updated = True
                else:
                    # Keep original line if format is unexpected
                    new_lines.append(line)
            else:
                # Keep original line if index not found in mapping
                new_lines.append(line)
                print(f"Warning: Unknown index '{class_index}' in {label_path}")
    
    if updated:
        with open(label_path, 'w') as f:
            for line in new_lines:
                f.write(line + '\n')
    
    return updated

def process_animal_folder(animal_folder):
    """Process all label files in an animal folder"""
    animal_class = os.path.basename(animal_folder)
    print(f"Processing {animal_class} folder...")
    
    # Find all .txt files in the folder
    label_files = glob.glob(os.path.join(animal_folder, "*.txt"))
    
    total_updated = 0
    for label_file in label_files:
        if fix_label_file(label_file, animal_class):
            total_updated += 1
    
    print(f"  Updated {total_updated} label files in {animal_class}")
    return total_updated

def main():
    """Main function to fix African Wildlife labels"""
    print("=== African Wildlife Label Fix Script ===")
    print(f"African Wildlife root: {AFRICAN_WILDLIFE_ROOT}")
    
    if not os.path.exists(AFRICAN_WILDLIFE_ROOT):
        print(f"Error: {AFRICAN_WILDLIFE_ROOT} directory not found!")
        return
    
    # Get all animal folders
    animal_folders = []
    for item in os.listdir(AFRICAN_WILDLIFE_ROOT):
        item_path = os.path.join(AFRICAN_WILDLIFE_ROOT, item)
        if os.path.isdir(item_path):
            animal_folders.append(item_path)
    
    print(f"Found {len(animal_folders)} animal folders: {[os.path.basename(f) for f in animal_folders]}")
    
    # Process each animal folder
    total_files_updated = 0
    for animal_folder in animal_folders:
        files_updated = process_animal_folder(animal_folder)
        total_files_updated += files_updated
    
    print(f"\n=== African Wildlife Label Fix Complete ===")
    print(f"Total label files updated: {total_files_updated}")
    print(f"Numeric to text mapping used:")
    for numeric, text in NUMERIC_TO_TEXT_MAPPING.items():
        print(f"  {numeric} -> {text}")
    print("\nAll African Wildlife labels now use text format!")

if __name__ == "__main__":
    main()
