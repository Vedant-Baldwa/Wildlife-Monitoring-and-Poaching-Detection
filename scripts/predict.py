#!/usr/bin/env python3
"""
Simple Wildlife Detection Prediction Script
Easy-to-use script for making predictions with your trained model
"""

import os
import sys
import cv2
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

def predict_image(model_path, image_path, output_path=None, confidence=0.5, show=True):
    """Predict on a single image."""
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
    
    # Save result if output path provided
    if output_path:
        # Save the annotated image
        result.save(output_path)
        print(f"💾 Result saved to: {output_path}")
    
    # Show result if requested
    if show:
        result.show()
    
    return results

def predict_video(model_path, video_path, output_path=None, confidence=0.5, show=True):
    """Predict on a video."""
    print(f"🎥 Processing video: {video_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Run prediction
    results = model(video_path, conf=confidence, save=output_path is not None, project="predictions", name="video_results")
    
    print(f"✅ Video processing completed!")
    if output_path:
        print(f"💾 Result saved to: predictions/video_results/")
    
    return results

def predict_realtime(model_path, confidence=0.5, camera_id=0):
    """Real-time prediction using webcam."""
    print(f"📹 Starting real-time detection (Camera {camera_id})")
    print("Press 'q' to quit, 's' to save current frame")
    
    # Load model
    model = YOLO(model_path)
    class_names = load_class_names()
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_id}")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Run prediction
            results = model(frame, conf=confidence)
            
            # Draw results on frame
            annotated_frame = results[0].plot()
            
            # Add info text
            cv2.putText(annotated_frame, "Press 'q' to quit, 's' to save", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Wildlife Detection', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                import time
                timestamp = int(time.time())
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"💾 Frame saved as {filename}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🛑 Real-time detection stopped")

def predict_folder(model_path, folder_path, output_folder=None, confidence=0.5):
    """Predict on all images in a folder."""
    print(f"📁 Processing folder: {folder_path}")
    
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
        
        # Save result if output folder provided
        if output_folder:
            output_path = Path(output_folder) / f"pred_{image_file.name}"
            result.save(str(output_path))
    
    print(f"\n✅ Folder processing completed!")

def main():
    parser = argparse.ArgumentParser(description="Wildlife Detection Prediction")
    parser.add_argument("--model", type=str, default="results/wildlife_detection_v3/weights/best.pt", 
                       help="Path to trained model (default: best model)")
    parser.add_argument("--input", type=str, help="Input image, video, or folder path")
    parser.add_argument("--output", type=str, help="Output path for results")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--camera", type=int, default=None, help="Camera ID for real-time detection")
    parser.add_argument("--no-show", action="store_true", help="Don't show results")
    
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
        if args.camera is not None:
            # Real-time detection
            predict_realtime(args.model, args.confidence, args.camera)
        
        elif args.input:
            input_path = Path(args.input)
            
            if not input_path.exists():
                print(f"❌ Input not found: {args.input}")
                sys.exit(1)
            
            if input_path.is_file():
                # Single file
                if input_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                    # Image
                    predict_image(args.model, str(input_path), args.output, args.confidence, not args.no_show)
                elif input_path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}:
                    # Video
                    predict_video(args.model, str(input_path), args.output, args.confidence, not args.no_show)
                else:
                    print(f"❌ Unsupported file format: {input_path.suffix}")
                    sys.exit(1)
            
            elif input_path.is_dir():
                # Folder
                predict_folder(args.model, str(input_path), args.output, args.confidence)
            
            else:
                print(f"❌ Invalid input: {args.input}")
                sys.exit(1)
        
        else:
            # No input provided, show help
            print("❌ Please provide input (--input or --camera)")
            print("\n📖 Usage Examples:")
            print("  # Detect in an image")
            print("  python scripts/predict.py --input path/to/image.jpg")
            print("  # Detect in a video")
            print("  python scripts/predict.py --input path/to/video.mp4")
            print("  # Detect in a folder of images")
            print("  python scripts/predict.py --input path/to/folder/")
            print("  # Real-time detection with webcam")
            print("  python scripts/predict.py --camera 0")
            print("  # Save results to file")
            print("  python scripts/predict.py --input image.jpg --output result.jpg")
            print("  # Adjust confidence threshold")
            print("  python scripts/predict.py --input image.jpg --confidence 0.7")
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
