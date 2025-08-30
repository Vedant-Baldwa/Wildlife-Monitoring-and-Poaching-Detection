#!/usr/bin/env python3
"""
Script to analyze the dataset and identify potential issues
"""

import os
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def analyze_dataset():
    """Analyze the dataset distribution and quality"""
    
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "dataset"
    
    print("=== Dataset Analysis ===")
    
    # Load class names
    classes_file = dataset_path / "classes.txt"
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Number of classes: {len(class_names)}")
    else:
        print("classes.txt not found!")
        return
    
    # Analyze each split
    splits = ['train', 'val', 'test']
    split_stats = {}
    
    for split in splits:
        images_path = dataset_path / split / "images"
        labels_path = dataset_path / split / "labels"
        
        if not images_path.exists() or not labels_path.exists():
            print(f"Split {split} not found!")
            continue
            
        # Count images
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.jpeg")) + list(images_path.glob("*.png"))
        num_images = len(image_files)
        
        # Count labels and analyze class distribution
        label_files = list(labels_path.glob("*.txt"))
        num_labels = len(label_files)
        
        class_counts = Counter()
        total_objects = 0
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id < len(class_names):
                                class_counts[class_names[class_id]] += 1
                                total_objects += 1
            except Exception as e:
                print(f"Error reading {label_file}: {e}")
        
        split_stats[split] = {
            'images': num_images,
            'labels': num_labels,
            'objects': total_objects,
            'class_distribution': dict(class_counts),
            'avg_objects_per_image': total_objects / num_images if num_images > 0 else 0
        }
        
        print(f"\n{split.upper()} Split:")
        print(f"  Images: {num_images}")
        print(f"  Labels: {num_labels}")
        print(f"  Total objects: {total_objects}")
        print(f"  Avg objects per image: {total_objects / num_images:.2f}")
        
        # Show top 10 classes
        top_classes = class_counts.most_common(10)
        print(f"  Top 10 classes:")
        for class_name, count in top_classes:
            print(f"    {class_name}: {count}")
    
    # Analyze class imbalance
    print(f"\n=== Class Imbalance Analysis ===")
    
    # Combine all splits
    all_class_counts = Counter()
    for split_data in split_stats.values():
        all_class_counts.update(split_data['class_distribution'])
    
    if all_class_counts:
        min_count = min(all_class_counts.values())
        max_count = max(all_class_counts.values())
        avg_count = sum(all_class_counts.values()) / len(all_class_counts)
        
        print(f"Total objects across all splits: {sum(all_class_counts.values())}")
        print(f"Average objects per class: {avg_count:.1f}")
        print(f"Min objects per class: {min_count}")
        print(f"Max objects per class: {max_count}")
        print(f"Imbalance ratio (max/min): {max_count/min_count:.1f}")
        
        # Identify classes with very few samples
        low_count_classes = [(name, count) for name, count in all_class_counts.items() if count < 50]
        if low_count_classes:
            print(f"\nClasses with < 50 samples (potential issues):")
            for name, count in sorted(low_count_classes, key=lambda x: x[1]):
                print(f"  {name}: {count}")
    
    # Check for empty labels
    print(f"\n=== Label Quality Check ===")
    empty_labels = 0
    total_labels = 0
    
    for split in splits:
        labels_path = dataset_path / split / "labels"
        if labels_path.exists():
            for label_file in labels_path.glob("*.txt"):
                total_labels += 1
                try:
                    with open(label_file, 'r') as f:
                        content = f.read().strip()
                        if not content:
                            empty_labels += 1
                except:
                    pass
    
    print(f"Total label files: {total_labels}")
    print(f"Empty label files: {empty_labels}")
    print(f"Percentage empty: {empty_labels/total_labels*100:.1f}%" if total_labels > 0 else "N/A")
    
    # Save analysis results
    analysis_results = {
        'class_names': class_names,
        'split_statistics': split_stats,
        'total_class_distribution': dict(all_class_counts),
        'quality_metrics': {
            'total_labels': total_labels,
            'empty_labels': empty_labels,
            'empty_percentage': empty_labels/total_labels*100 if total_labels > 0 else 0
        }
    }
    
    output_file = project_root / "reports" / "dataset_analysis.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nAnalysis results saved to: {output_file}")
    
    # Recommendations
    print(f"\n=== Recommendations ===")
    
    if empty_labels > 0:
        print("⚠️  Found empty label files. Consider removing these images or adding proper labels.")
    
    if low_count_classes:
        print("⚠️  Some classes have very few samples. Consider:")
        print("   - Data augmentation for these classes")
        print("   - Collecting more data for these classes")
        print("   - Using class weights in training")
    
    if max_count/min_count > 10:
        print("⚠️  High class imbalance detected. Consider:")
        print("   - Using class weights")
        print("   - Focal loss")
        print("   - Balanced sampling")
    
    print("✅ Dataset analysis complete!")

if __name__ == "__main__":
    analyze_dataset()
