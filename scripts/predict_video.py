#!/usr/bin/env python3
"""
Video Wildlife Detection Prediction Script
Processes videos and saves annotated results
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import logging
import cv2
import time

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

def predict_video_manual(model_path, video_path, output_path=None, confidence=0.3, max_frames=None):
    """
    Manual video processing with frame-by-frame detection and custom output.
    """
    print(f"🎥 Processing video: {video_path}")
    
    # Load model
    model = YOLO(model_path)
    class_names = load_class_names()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"💾 Output will be saved to: {output_path}")
    
    # Process frames
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    
    print("🔄 Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Limit frames if specified
        if max_frames and frame_count > max_frames:
            break
        
        # Run detection
        results = model(frame, conf=confidence)
        
        # Draw detections on frame
        annotated_frame = results[0].plot()
        
        # Add frame info
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Count detections in this frame
        if results[0].boxes is not None:
            frame_detections = len(results[0].boxes)
            detection_count += frame_detections
        else:
            frame_detections = 0
        
        # Add info text
        info_text = f"Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f} | Detections: {frame_detections}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        if writer:
            writer.write(annotated_frame)
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {frame_detections} detections")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\n✅ Video processing completed!")
    print(f"📊 Statistics:")
    print(f"   - Frames processed: {frame_count}")
    print(f"   - Total detections: {detection_count}")
    print(f"   - Average FPS: {avg_fps:.2f}")
    print(f"   - Processing time: {total_time:.2f}s")
    
    if output_path:
        print(f"💾 Result saved to: {output_path}")
    
    return frame_count

def predict_video_yolo(model_path, video_path, output_path=None, confidence=0.3):
    """
    Use YOLO's built-in video processing (faster but less control).
    """
    print(f"🎥 Processing video with YOLO: {video_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Run prediction
    results = model(video_path, conf=confidence, save=output_path is not None, 
                   project="video_predictions", name="results")
    
    print(f"✅ Video processing completed!")
    if output_path:
        print(f"💾 Result saved to: video_predictions/results/")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Video Wildlife Detection Prediction")
    parser.add_argument("--model", type=str, default="results/wildlife_detection_v3/weights/best.pt", 
                       help="Path to trained model (default: best model)")
    parser.add_argument("--input", type=str, required=True, help="Input video file path")
    parser.add_argument("--output", type=str, help="Output video file path")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--method", type=str, choices=['manual', 'yolo'], default='manual',
                       help="Processing method: manual (more control) or yolo (faster)")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process (for testing)")
    
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
    
    # Validate input video
    if not Path(args.input).exists():
        print(f"❌ Video not found: {args.input}")
        sys.exit(1)
    
    # Check video format
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v'}
    if Path(args.input).suffix.lower() not in video_extensions:
        print(f"❌ Unsupported video format: {Path(args.input).suffix}")
        print(f"Supported formats: {', '.join(video_extensions)}")
        sys.exit(1)
    
    print(f"🚀 Video Wildlife Detection System")
    print(f"📦 Model: {args.model}")
    print(f"🎯 Confidence: {args.confidence}")
    print(f"🔧 Method: {args.method}")
    print("-" * 50)
    
    try:
        if args.method == 'manual':
            predict_video_manual(args.model, args.input, args.output, args.confidence, args.max_frames)
        else:
            predict_video_yolo(args.model, args.input, args.output, args.confidence)
    
    except KeyboardInterrupt:
        print("\n🛑 Video processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Video processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
