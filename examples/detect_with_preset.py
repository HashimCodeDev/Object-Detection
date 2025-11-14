#!/usr/bin/env python3
"""
Object Detection with Model Presets

Easy switching between different model configurations using presets.
"""

import sys
from drone_object_detection import DroneObjectDistanceDetector
from drone_object_detection.models_config import PRESETS, list_presets

def main():
    # List available presets
    if len(sys.argv) > 1 and sys.argv[1] in ['--list', '-l', 'list']:
        list_presets()
        print("\nUsage: uv run examples/detect_with_preset.py [preset_name]")
        print("Example: uv run examples/detect_with_preset.py laptop_cpu_fastest")
        return
    
    # Get preset name from command line or use default
    preset_name = sys.argv[1] if len(sys.argv) > 1 else "laptop_cpu_fastest"
    
    if preset_name not in PRESETS:
        print(f"❌ Unknown preset: {preset_name}")
        print("\nAvailable presets:")
        for name in PRESETS.keys():
            print(f"  - {name}")
        sys.exit(1)
    
    preset = PRESETS[preset_name]
    
    print("\n" + "="*60)
    print(f"🎯 USING PRESET: {preset_name}")
    print("="*60)
    print(f"Detection Model: {preset['detection']}")
    print(f"Depth Model: {preset['depth']}")
    print(f"Description: {preset['description']}")
    print("="*60 + "\n")
    
    print("📦 Loading models...")
    
    try:
        detector = DroneObjectDistanceDetector(
            detection_model=preset['detection'],
            depth_model=preset['depth'],
            confidence_threshold=0.5
        )
        
        print("\n🎥 Starting webcam detection...")
        print("Press 'q' to quit, 's' to save frame\n")
        
        detector.detect_webcam_with_distance(camera_index=0)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
