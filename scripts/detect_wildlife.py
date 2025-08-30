#!/usr/bin/env python3
"""
Wildlife Detection Script
Detects animals in images and videos using trained YOLO model
Features:
- Image and video processing
- Real-time detection with OpenCV
- Bounding boxes with animal names
- Confidence scores
- Multiple output formats
"""

import os
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging
from time import time
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WildlifeDetector:
    def __init__(self, model_path, confidence=0.5, iou_threshold=0.45):
        """
        Initialize wildlife detector.
        
        Args:
            model_path: Path to trained YOLO model
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        
        # Load class names
        self.class_names = self.load_class_names()
        
        # Load model
        logging.info(f"Loading model from {model_path}")
        self.model = YOLO(model_path)
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {self.device}")
        
        logging.info(f"Model loaded successfully. {len(self.class_names)} classes available.")
    
    def load_class_names(self):
        """Load class names from dataset classes.txt file."""
        classes_file = Path("dataset/classes.txt")
        if classes_file.exists():
            class_names = []
            with open(classes_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        class_names.append(' '.join(parts[1:]))  # Join multi-word class names
            return class_names
        else:
            logging.warning("classes.txt not found. Using default class names.")
            return [
                'Bear', 'Brown bear', 'Buffalo', 'Bull', 'Cattle', 'Cheetah', 'Chicken',
                'Deer', 'Elephant', 'Fox', 'Giraffe', 'Goat', 'Hippopotamus', 'Horse',
                'Jaguar', 'Kangaroo', 'Koala', 'Leopard', 'Lion', 'Lynx', 'Monkey',
                'Mule', 'Ostrich', 'Otter', 'Panda', 'Penguin', 'Pig', 'Polar bear',
                'Rabbit', 'Raccoon', 'Red panda', 'Rhinoceros', 'Sheep', 'Tiger',
                'Turkey', 'Zebra'
            ]
    
    def draw_detections(self, image, results):
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image (numpy array)
            results: YOLO detection results
            
        Returns:
            Annotated image
        """
        annotated_image = image.copy()
        
        if results and len(results) > 0:
            result = results[0]  # Get first result
            
            if result.boxes is not None:
                boxes = result.boxes
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green color
                    thickness = 2
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
                    
                    # Create label text
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # Calculate text position
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_x = x1
                    text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height
                    
                    # Draw text background
                    cv2.rectangle(annotated_image, 
                                (text_x, text_y - text_height - baseline),
                                (text_x + text_width, text_y + baseline),
                                color, -1)
                    
                    # Draw text
                    cv2.putText(annotated_image, label, (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_image
    
    def detect_image(self, image_path, output_path=None, show_result=True):
        """
        Detect animals in a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            show_result: Whether to display the result
            
        Returns:
            Detection results
        """
        logging.info(f"Processing image: {image_path}")
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logging.error(f"Could not read image: {image_path}")
            return None
        
        # Run detection
        start_time = time()
        results = self.model(image, conf=self.confidence, iou=self.iou_threshold)
        inference_time = time() - start_time
        
        # Draw detections
        annotated_image = self.draw_detections(image, results)
        
        # Print detection info
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                num_detections = len(result.boxes)
                logging.info(f"Found {num_detections} animals in {inference_time:.3f}s")
                
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
                    logging.info(f"  {i+1}. {class_name}: {confidence:.3f}")
            else:
                logging.info("No animals detected")
        else:
            logging.info("No animals detected")
        
        # Save result
        if output_path:
            cv2.imwrite(str(output_path), annotated_image)
            logging.info(f"Result saved to: {output_path}")
        
        # Show result
        if show_result:
            cv2.imshow('Wildlife Detection', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results
    
    def detect_video(self, video_path, output_path=None, show_result=True):
        """
        Detect animals in a video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show_result: Whether to display the result
            
        Returns:
            Detection results
        """
        logging.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Could not open video: {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            logging.info(f"Output video will be saved to: {output_path}")
        
        # Process frames
        frame_count = 0
        start_time = time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = self.model(frame, conf=self.confidence, iou=self.iou_threshold)
            
            # Draw detections
            annotated_frame = self.draw_detections(frame, results)
            
            # Add frame info
            elapsed_time = time() - start_time
            fps_current = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            info_text = f"Frame: {frame_count}/{total_frames} | FPS: {fps_current:.1f}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            if writer:
                writer.write(annotated_frame)
            
            # Show frame
            if show_result:
                cv2.imshow('Wildlife Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                logging.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_result:
            cv2.destroyAllWindows()
        
        total_time = time() - start_time
        logging.info(f"Video processing completed in {total_time:.2f}s")
        logging.info(f"Average FPS: {frame_count/total_time:.2f}")
        
        return frame_count
    
    def detect_realtime(self, camera_id=0):
        """
        Real-time detection using webcam.
        
        Args:
            camera_id: Camera device ID
        """
        logging.info(f"Starting real-time detection with camera {camera_id}")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logging.error(f"Could not open camera {camera_id}")
            return
        
        logging.info("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        start_time = time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Run detection
            results = self.model(frame, conf=self.confidence, iou=self.iou_threshold)
            
            # Draw detections
            annotated_frame = self.draw_detections(frame, results)
            
            # Add performance info
            elapsed_time = time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            info_text = f"FPS: {fps:.1f} | Press 'q' to quit"
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Real-time Wildlife Detection', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time())
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                logging.info(f"Frame saved as {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Real-time detection stopped")

def main():
    parser = argparse.ArgumentParser(description="Wildlife Detection using YOLO")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model")
    parser.add_argument("--input", type=str, help="Input image or video file")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--camera", type=int, default=None, help="Camera ID for real-time detection")
    parser.add_argument("--no-show", action="store_true", help="Don't show results")
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model).exists():
        logging.error(f"Model not found: {args.model}")
        sys.exit(1)
    
    # Initialize detector
    detector = WildlifeDetector(args.model, args.confidence, args.iou)
    
    try:
        if args.camera is not None:
            # Real-time detection
            detector.detect_realtime(args.camera)
        elif args.input:
            # File detection
            input_path = Path(args.input)
            if not input_path.exists():
                logging.error(f"Input file not found: {args.input}")
                sys.exit(1)
            
            # Determine if input is image or video
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
            
            if input_path.suffix.lower() in image_extensions:
                # Image detection
                detector.detect_image(input_path, args.output, not args.no_show)
            elif input_path.suffix.lower() in video_extensions:
                # Video detection
                detector.detect_video(input_path, args.output, not args.no_show)
            else:
                logging.error(f"Unsupported file format: {input_path.suffix}")
                sys.exit(1)
        else:
            logging.error("Please provide either --input or --camera")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.info("Detection interrupted by user")
    except Exception as e:
        logging.exception(f"Detection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
