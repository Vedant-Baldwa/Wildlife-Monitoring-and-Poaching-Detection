#!/usr/bin/env python3
"""
Script to create YOLO-ready dataset from Unified Dataset
- Creates train/val/test splits
- Converts class names to indexes
- Creates data.yaml file
- Organizes images and labels properly
"""

import os
import shutil
import random
from pathlib import Path
import yaml

def create_dataset_structure():
    """Create the dataset folder structure"""
    dataset_dir = Path("dataset")
    
    # Create main directories
    for split in ["train", "val", "test"]:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    return dataset_dir

def get_all_classes():
    """Get all animal classes from Unified Dataset"""
    unified_dir = Path("Unified Dataset")
    classes = []
    
    for item in unified_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            classes.append(item.name)
    
    # Sort classes for consistent ordering
    classes.sort()
    return classes

def create_classes_file(classes, dataset_dir):
    """Create classes.txt file with class names and indexes"""
    classes_file = dataset_dir / "classes.txt"
    
    with open(classes_file, 'w') as f:
        for idx, class_name in enumerate(classes):
            f.write(f"{idx} {class_name}\n")
    
    print(f"Created classes.txt with {len(classes)} classes")
    return {class_name: idx for idx, class_name in enumerate(classes)}

def get_all_files(unified_dir, class_name):
    """Get all image files for a specific class"""
    class_dir = unified_dir / class_name
    labels_dir = class_dir / "labels"
    
    image_files = []
    label_files = []
    
    # Get all image files
    for img_file in class_dir.glob("*.jpg"):
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            image_files.append(img_file)
            label_files.append(label_file)
    
    return image_files, label_files

def convert_label_format(label_file, class_mapping, class_name):
    """Convert label file from 'AnimalName x y w h' to 'class_index x y w h'"""
    class_index = class_mapping[class_name]
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    converted_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # Replace animal name with class index
            parts[0] = str(class_index)
            converted_lines.append(' '.join(parts) + '\n')
    
    return converted_lines

def split_and_copy_data(unified_dir, dataset_dir, class_mapping, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Split data and copy to train/val/test directories"""
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    classes = list(class_mapping.keys())
    
    for class_name in classes:
        print(f"Processing {class_name}...")
        
        image_files, label_files = get_all_files(unified_dir, class_name)
        
        if not image_files:
            print(f"Warning: No valid image-label pairs found for {class_name}")
            continue
        
        # Shuffle files for random split
        file_pairs = list(zip(image_files, label_files))
        random.shuffle(file_pairs)
        
        total_files = len(file_pairs)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)
        
        # Split files
        train_files = file_pairs[:train_end]
        val_files = file_pairs[train_end:val_end]
        test_files = file_pairs[val_end:]
        
        # Copy files to respective directories
        splits = [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files)
        ]
        
        for split_name, files in splits:
            for img_file, label_file in files:
                # Copy image
                dest_img = dataset_dir / split_name / "images" / f"{class_name}_{img_file.name}"
                shutil.copy2(img_file, dest_img)
                
                # Convert and copy label
                converted_labels = convert_label_format(label_file, class_mapping, class_name)
                dest_label = dataset_dir / split_name / "labels" / f"{class_name}_{label_file.name}"
                
                with open(dest_label, 'w') as f:
                    f.writelines(converted_labels)
        
        print(f"  {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

def create_data_yaml(dataset_dir, classes):
    """Create data.yaml file for YOLO training"""
    data_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_file = dataset_dir / "data.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created data.yaml with {len(classes)} classes")

def main():
    """Main function to create YOLO dataset"""
    print("Creating YOLO-ready dataset from Unified Dataset...")
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Create dataset structure
    dataset_dir = create_dataset_structure()
    
    # Get all classes
    classes = get_all_classes()
    print(f"Found {len(classes)} classes: {classes}")
    
    # Create class mapping
    class_mapping = create_classes_file(classes, dataset_dir)
    
    # Split and copy data
    unified_dir = Path("Unified Dataset")
    split_and_copy_data(unified_dir, dataset_dir, class_mapping)
    
    # Create data.yaml
    create_data_yaml(dataset_dir, classes)
    
    print("\nDataset creation completed!")
    print(f"Dataset location: {dataset_dir.absolute()}")
    print("Files created:")
    print("- dataset/train/images/ (training images)")
    print("- dataset/train/labels/ (training labels)")
    print("- dataset/val/images/ (validation images)")
    print("- dataset/val/labels/ (validation labels)")
    print("- dataset/test/images/ (test images)")
    print("- dataset/test/labels/ (test labels)")
    print("- dataset/classes.txt (class names and indexes)")
    print("- dataset/data.yaml (YOLO configuration)")

if __name__ == "__main__":
    main()
