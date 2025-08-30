#!/usr/bin/env python3
"""
Simple Wildlife Detection Prediction Script
Saves results to files (no display issues on Windows)
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_class_names():
    """Load class names from dataset classes.txt file."""
    classes_file = Path("dataset/classes.txt")
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    else:
        # Fallback class names
        return [
            'Bear', 'Brown bear', 'Buffalo', 'Bull', 'Cattle', 'Cheetah', 'Chicken',
            'Deer', 'Elephant', 'Fox', 'Giraffe', 'Goat', 'Hippopotamus', 'Horse',
            'Jaguar', 'Kangaroo', 'Koala', 'Leopard', 'Lion', 'Lynx', 'Monkey',
            'Mule', 'Ostrich', 'Otter', 'Panda', 'Penguin', 'Pig', 'Polar bear',
            'Rabbit', 'Raccoon', 'Red panda', 'Rhinoceros', 'Sheep', 'Tiger',
            'Turkey', 'Zebra'
        ]

def predict_image(model_path, image_path, output_path=None, confidence=0.3):
    """Predict on a single image and save result."""
    print(f"🔍 Detecting wildlife in: {image_path}")
    
    # Load model
    model = YOLO(model_path)
    class_names = load_class_names()
    
    # Run prediction
    results = model(image_path, conf=confidence)
    
    # Process results
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            print(f"✅ Found {len(result.boxes)} animals:")
            
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                print(f"   {i+1}. {class_name} (confidence: {conf:.3f})")
        else:
            print("❌ No animals detected")
    else:
        print("❌ No animals detected")
    
    # Save result
    if output_path:
        result.save(output_path)
        print(f"💾 Result saved to: {output_path}")
    else:
        # Save to default location
        default_output = f"prediction_result_{Path(image_path).stem}.jpg"
        result.save(default_output)
        print(f"💾 Result saved to: {default_output}")
    
    return results

def predict_folder(model_path, folder_path, output_folder="predictions", confidence=0.3):
    """Predict on all images in a folder."""
    print(f"📁 Processing folder: {folder_path}")
    
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    class_names = load_class_names()
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print("❌ No image files found in the folder")
        return
    
    print(f"📸 Found {len(image_files)} images to process")
    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file.name}")
        
        # Run prediction
        results = model(str(image_file), conf=confidence)
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                print(f"   ✅ Found {len(result.boxes)} animals:")
                for box in result.boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                    print(f"      - {class_name} (confidence: {conf:.3f})")
            else:
                print("   ❌ No animals detected")
        else:
            print("   ❌ No animals detected")
        
        # Save result
        output_file = output_path / f"pred_{image_file.name}"
        result.save(str(output_file))
        print(f"   💾 Saved to: {output_file}")
    
    print(f"\n✅ Folder processing completed! Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Simple Wildlife Detection Prediction")
    parser.add_argument("--model", type=str, default="results/wildlife_detection_v3/weights/best.pt", 
                       help="Path to trained model (default: best model)")
    parser.add_argument("--input", type=str, required=True, help="Input image or folder path")
    parser.add_argument("--output", type=str, help="Output path for results")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("Available models:")
        models_dir = Path("results")
        for model_folder in models_dir.glob("wildlife_detection_*/weights"):
            if model_folder.exists():
                print(f"   - {model_folder}")
        sys.exit(1)
    
    print(f"🚀 Wildlife Detection System")
    print(f"📦 Model: {args.model}")
    print(f"🎯 Confidence: {args.confidence}")
    print("-" * 50)
    
    try:
        input_path = Path(args.input)
        
        if not input_path.exists():
            print(f"❌ Input not found: {args.input}")
            sys.exit(1)
        
        if input_path.is_file():
            # Single image
            if input_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                predict_image(args.model, str(input_path), args.output, args.confidence)
            else:
                print(f"❌ Unsupported file format: {input_path.suffix}")
                sys.exit(1)
        
        elif input_path.is_dir():
            # Folder
            predict_folder(args.model, str(input_path), args.output, args.confidence)
        
        else:
            print(f"❌ Invalid input: {args.input}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Prediction interrupted by user")
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
