#!/usr/bin/env python3
"""
Real-time Webcam Object Detection with Distance Estimation

This script uses your laptop camera to detect objects and estimate their distance
in real-time using RT-DETR and Depth Anything V2 models.

Usage:
    uv run examples/detect_webcam_distance.py
    
Controls:
    - Press 'q' to quit
    - Press 's' to save current frame
"""

from drone_object_detection import DroneObjectDistanceDetector
import sys

def main():
    print("\n" + "="*60)
    print("🚁 DRONE OBJECT DETECTION WITH DISTANCE ESTIMATION")
    print("="*60)
    print("\n📦 Initializing models...")
    print("⏳ This may take a moment on first run (downloading models)...\n")
    
    try:
        # Initialize detector
        detector = DroneObjectDistanceDetector(
            detection_model="PekingU/rtdetr_r50vd",
            depth_model="depth-anything/Depth-Anything-V2-Small-hf",
            confidence_threshold=0.5
        )
        
        # Start webcam detection
        detector.detect_webcam_with_distance(camera_index=0)
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure your webcam is not being used by another application")
        print("  2. Check if dependencies are installed: uv sync")
        print("  3. Try a different camera index if you have multiple cameras")
        sys.exit(1)

if __name__ == "__main__":
    main()
