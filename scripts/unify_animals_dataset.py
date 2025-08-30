#!/usr/bin/env python3
"""
unify_animals_dataset.py

This script unifies the Animals Dataset by merging train and test splits into
single animal folders. Each animal folder will have:
- All images from train and test combined
- A labels folder with all labels from train and test combined

Usage:
  python scripts/unify_animals_dataset.py
"""

import os
import glob
import shutil
from pathlib import Path

# Configuration
ANIMALS_DATASET_ROOT = "Animals Dataset"
TRAIN_DIR = os.path.join(ANIMALS_DATASET_ROOT, "train")
TEST_DIR = os.path.join(ANIMALS_DATASET_ROOT, "test")

def get_animal_classes():
    """Get all animal classes from train directory"""
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: {TRAIN_DIR} not found!")
        return []
    
    animal_classes = []
    for item in os.listdir(TRAIN_DIR):
        item_path = os.path.join(TRAIN_DIR, item)
        if os.path.isdir(item_path):
            animal_classes.append(item)
    
    return sorted(animal_classes)

def unify_animal_class(animal_class):
    """Unify a single animal class by merging train and test data"""
    print(f"Unifying {animal_class}...")
    
    # Create unified animal folder
    unified_animal_dir = os.path.join(ANIMALS_DATASET_ROOT, animal_class)
    os.makedirs(unified_animal_dir, exist_ok=True)
    
    # Create labels folder
    labels_dir = os.path.join(unified_animal_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    # Process train data
    train_animal_dir = os.path.join(TRAIN_DIR, animal_class)
    if os.path.exists(train_animal_dir):
        # Copy images from train
        train_images = glob.glob(os.path.join(train_animal_dir, "*.jpg")) + \
                      glob.glob(os.path.join(train_animal_dir, "*.jpeg")) + \
                      glob.glob(os.path.join(train_animal_dir, "*.png"))
        
        for img_path in train_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(unified_animal_dir, img_name)
            shutil.copy2(img_path, dest_path)
        
        # Copy labels from train
        train_labels_dir = os.path.join(train_animal_dir, "Label")
        if os.path.exists(train_labels_dir):
            train_labels = glob.glob(os.path.join(train_labels_dir, "*.txt"))
            for label_path in train_labels:
                label_name = os.path.basename(label_path)
                dest_path = os.path.join(labels_dir, label_name)
                shutil.copy2(label_path, dest_path)
        
        print(f"  Copied {len(train_images)} images and {len(train_labels) if os.path.exists(train_labels_dir) else 0} labels from train")
    
    # Process test data
    test_animal_dir = os.path.join(TEST_DIR, animal_class)
    if os.path.exists(test_animal_dir):
        # Copy images from test
        test_images = glob.glob(os.path.join(test_animal_dir, "*.jpg")) + \
                     glob.glob(os.path.join(test_animal_dir, "*.jpeg")) + \
                     glob.glob(os.path.join(test_animal_dir, "*.png"))
        
        for img_path in test_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(unified_animal_dir, img_name)
            shutil.copy2(img_path, dest_path)
        
        # Copy labels from test
        test_labels_dir = os.path.join(test_animal_dir, "Label")
        if os.path.exists(test_labels_dir):
            test_labels = glob.glob(os.path.join(test_labels_dir, "*.txt"))
            for label_path in test_labels:
                label_name = os.path.basename(label_path)
                dest_path = os.path.join(labels_dir, label_name)
                shutil.copy2(label_path, dest_path)
        
        print(f"  Copied {len(test_images)} images and {len(test_labels) if os.path.exists(test_labels_dir) else 0} labels from test")
    
    # Count total files
    total_images = len(glob.glob(os.path.join(unified_animal_dir, "*.jpg")) + 
                      glob.glob(os.path.join(unified_animal_dir, "*.jpeg")) + 
                      glob.glob(os.path.join(unified_animal_dir, "*.png")))
    total_labels = len(glob.glob(os.path.join(labels_dir, "*.txt")))
    
    print(f"  Total: {total_images} images, {total_labels} labels")
    return total_images, total_labels

def main():
    """Main function to unify all animal classes"""
    print("Unifying Animals Dataset...")
    
    # Check if dataset exists
    if not os.path.exists(ANIMALS_DATASET_ROOT):
        print(f"Error: {ANIMALS_DATASET_ROOT} not found!")
        return
    
    # Get all animal classes
    animal_classes = get_animal_classes()
    if not animal_classes:
        print("No animal classes found!")
        return
    
    print(f"Found {len(animal_classes)} animal classes:")
    for animal in animal_classes:
        print(f"  - {animal}")
    
    # Unify each animal class
    total_images = 0
    total_labels = 0
    
    for animal_class in animal_classes:
        images, labels = unify_animal_class(animal_class)
        total_images += images
        total_labels += labels
    
    print(f"\nUnification complete!")
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print(f"Animal classes: {len(animal_classes)}")
    
    # Show final structure
    print(f"\nFinal structure:")
    for animal in animal_classes:
        animal_dir = os.path.join(ANIMALS_DATASET_ROOT, animal)
        images_count = len(glob.glob(os.path.join(animal_dir, "*.jpg")) + 
                          glob.glob(os.path.join(animal_dir, "*.jpeg")) + 
                          glob.glob(os.path.join(animal_dir, "*.png")))
        labels_count = len(glob.glob(os.path.join(animal_dir, "labels", "*.txt")))
        print(f"  {animal}: {images_count} images, {labels_count} labels")

if __name__ == "__main__":
    main()
