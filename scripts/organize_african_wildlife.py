#!/usr/bin/env python3
"""
organize_african_wildlife.py

This script organizes the African Wildlife dataset by creating a "labels" folder
in each animal folder and moving all .txt files into that labels folder.

Usage:
  python scripts/organize_african_wildlife.py
"""

import os
import glob
import shutil
from pathlib import Path

# Configuration
AFRICAN_WILDLIFE_ROOT = "African Wildlife"

def organize_animal_folder(animal_folder_path):
    """Organize a single animal folder by creating labels folder and moving .txt files"""
    animal_name = os.path.basename(animal_folder_path)
    print(f"Organizing {animal_name} folder...")
    
    # Create labels folder
    labels_folder = os.path.join(animal_folder_path, "labels")
    os.makedirs(labels_folder, exist_ok=True)
    
    # Find all .txt files in the animal folder
    txt_files = glob.glob(os.path.join(animal_folder_path, "*.txt"))
    
    # Move .txt files to labels folder
    moved_count = 0
    for txt_file in txt_files:
        filename = os.path.basename(txt_file)
        destination = os.path.join(labels_folder, filename)
        
        # Skip if file is already in labels folder
        if os.path.dirname(txt_file) == labels_folder:
            continue
            
        shutil.move(txt_file, destination)
        moved_count += 1
    
    print(f"  Moved {moved_count} .txt files to labels folder")
    return moved_count

def main():
    """Main function to organize all animal folders"""
    print("Organizing African Wildlife dataset...")
    
    # Check if African Wildlife folder exists
    if not os.path.exists(AFRICAN_WILDLIFE_ROOT):
        print(f"Error: {AFRICAN_WILDLIFE_ROOT} folder not found!")
        return
    
    # Get all animal folders
    animal_folders = []
    for item in os.listdir(AFRICAN_WILDLIFE_ROOT):
        item_path = os.path.join(AFRICAN_WILDLIFE_ROOT, item)
        if os.path.isdir(item_path):
            animal_folders.append(item_path)
    
    if not animal_folders:
        print("No animal folders found!")
        return
    
    print(f"Found {len(animal_folders)} animal folders:")
    for folder in animal_folders:
        print(f"  - {os.path.basename(folder)}")
    
    # Organize each animal folder
    total_moved = 0
    for animal_folder in animal_folders:
        moved_count = organize_animal_folder(animal_folder)
        total_moved += moved_count
    
    print(f"\nOrganization complete!")
    print(f"Total .txt files moved: {total_moved}")

if __name__ == "__main__":
    main()
