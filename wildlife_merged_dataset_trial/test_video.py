#!/usr/bin/env python3
"""
Test script for video wildlife detection
"""

import os
import sys
from pathlib import Path

def test_video_detection():
    """Test video detection with different methods."""
    
    print("🎥 Video Wildlife Detection Test")
    print("=" * 50)
    
    # Check if we have a test video
    test_videos = []
    
    # Look for common video files in the project
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v'}
    
    # Check in common locations
    search_paths = [
        ".",
        "dataset",
        "test_videos",
        "videos",
        "sample_videos"
    ]
    
    for search_path in search_paths:
        if Path(search_path).exists():
            for ext in video_extensions:
                test_videos.extend(Path(search_path).glob(f"*{ext}"))
                test_videos.extend(Path(search_path).glob(f"*{ext.upper()}"))
    
    if test_videos:
        print("✅ Found test videos:")
        for i, video in enumerate(test_videos, 1):
            print(f"   {i}. {video}")
        
        print("\n📖 How to test videos:")
        print("\n1. Using the dedicated video script:")
        print(f"   python scripts/predict_video.py --input {test_videos[0]}")
        print(f"   python scripts/predict_video.py --input {test_videos[0]} --output result_video.mp4")
        print(f"   python scripts/predict_video.py --input {test_videos[0]} --confidence 0.5")
        print(f"   python scripts/predict_video.py --input {test_videos[0]} --method yolo")
        print(f"   python scripts/predict_video.py --input {test_videos[0]} --max-frames 100")
        
        print("\n2. Using the main detection script:")
        print(f"   python scripts/detect_wildlife.py --model results/wildlife_detection_v3/weights/best.pt --input {test_videos[0]} --no-show")
        print(f"   python scripts/detect_wildlife.py --model results/wildlife_detection_v3/weights/best.pt --input {test_videos[0]} --output result.mp4")
        
        print("\n3. Quick test with limited frames:")
        print(f"   python scripts/predict_video.py --input {test_videos[0]} --max-frames 50 --output test_result.mp4")
        
    else:
        print("❌ No test videos found!")
        print("\n📁 To test video detection, you need a video file.")
        print("   Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv, .m4v")
        print("\n📖 Usage examples (when you have a video):")
        print("   python scripts/predict_video.py --input your_video.mp4")
        print("   python scripts/predict_video.py --input your_video.mp4 --output result.mp4")
        print("   python scripts/predict_video.py --input your_video.mp4 --confidence 0.5")
        print("   python scripts/predict_video.py --input your_video.mp4 --method yolo")
        print("   python scripts/predict_video.py --input your_video.mp4 --max-frames 100")
    
    print("\n🔧 Processing Methods:")
    print("   - manual: Frame-by-frame processing with custom output")
    print("   - yolo: YOLO's built-in video processing (faster)")
    
    print("\n⚙️  Useful Parameters:")
    print("   --confidence: Detection threshold (0.1-1.0)")
    print("   --max-frames: Limit frames for quick testing")
    print("   --output: Save result video")
    print("   --method: Choose processing method")
    
    print("\n💡 Tips:")
    print("   - Start with --max-frames 50 for quick testing")
    print("   - Use confidence 0.3-0.5 for good results")
    print("   - Video processing can take time depending on length")
    print("   - Results are saved as MP4 files")

if __name__ == "__main__":
    test_video_detection()
