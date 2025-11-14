#!/usr/bin/env python3
"""
Browse and display all available models
"""

from drone_object_detection.models_config import (
    list_all_detection_models, 
    list_all_depth_models, 
    list_presets
)

def main():
    print("\n🤖 DRONE OBJECT DETECTION - MODEL BROWSER\n")
    
    # Show everything
    list_all_detection_models()
    list_all_depth_models()
    list_presets()
    
    print("\n" + "="*80)
    print("💡 USAGE EXAMPLES")
    print("="*80)
    print("\n1. Use a preset:")
    print("   uv run examples/detect_with_preset.py laptop_cpu_fastest")
    
    print("\n2. Custom configuration in your code:")
    print("   from drone_object_detection.models_config import DETECTION_MODELS, DEPTH_MODELS")
    print("   detector = DroneObjectDistanceDetector(")
    print("       detection_model=DETECTION_MODELS['rt_detr']['fastest']['name'],")
    print("       depth_model=DEPTH_MODELS['depth_anything_v2']['small']['name']")
    print("   )")
    
    print("\n3. Direct model names:")
    print("   detector = DroneObjectDistanceDetector(")
    print("       detection_model='PekingU/rtdetr_r18vd',")
    print("       depth_model='depth-anything/Depth-Anything-V2-Small-hf'")
    print("   )")
    print()

if __name__ == "__main__":
    main()
