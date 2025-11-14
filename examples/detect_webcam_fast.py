#!/usr/bin/env python3
"""
OPTIMIZED Real-time Webcam Object Detection with Distance Estimation
Uses frame skipping and lower resolution for better performance
"""

from drone_object_detection import DroneObjectDistanceDetector
import cv2
import sys
import time

def find_working_camera():
    """Find first working camera."""
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return i
    return None

def main():
    print("\n" + "="*60)
    print("🚀 OPTIMIZED OBJECT DETECTION WITH DISTANCE ESTIMATION")
    print("="*60)
    
    camera_index = find_working_camera()
    
    if camera_index is None:
        print("❌ No camera found!")
        sys.exit(1)
    
    print(f"\n✓ Using camera at index {camera_index}")
    print("\n📦 Loading lightweight models for better performance...")
    
    try:
        # Initialize with fastest models
        detector = DroneObjectDistanceDetector(
            detection_model="PekingU/rtdetr_r18vd",  # Fastest RT-DETR model
            depth_model="depth-anything/Depth-Anything-V2-Small-hf",  # Smallest depth model
            confidence_threshold=0.5
        )
        
        # Custom optimized webcam detection
        cap = cv2.VideoCapture(camera_index)
        
        # Lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced from 640
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Reduced from 480
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            sys.exit(1)
        
        print("\n" + "="*60)
        print("🎥 CAMERA STARTED - OPTIMIZED MODE")
        print("="*60)
        print("Performance Tips:")
        print("  • Using 320x240 resolution for speed")
        print("  • Processing every 3rd frame")
        print("  • Using lightest AI models")
        print("\nControls:")
        print("  • Press 'q' to quit")
        print("  • Press 's' to save frame")
        print("="*60 + "\n")
        
        from PIL import Image
        import torch
        import numpy as np
        
        fps_values = []
        frame_count = 0
        skip_frames = 3  # Process every 3rd frame
        
        # Cache for last detection results
        last_results = None
        last_depth_map = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Only process every Nth frame
            if frame_count % skip_frames == 0:
                start_time = time.time()
                
                # Convert to PIL
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Get depth map (only every Nth frame)
                last_depth_map = detector.estimate_depth_map(pil_image)
                
                # Detect objects
                inputs = detector.detection_processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(detector.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = detector.detection_model(**inputs)
                
                # Post-process
                target_sizes = torch.tensor([pil_image.size[::-1]]).to(detector.device)
                last_results = detector.detection_processor.post_process_object_detection(
                    outputs,
                    threshold=detector.confidence_threshold,
                    target_sizes=target_sizes
                )[0]
                
                inference_time = time.time() - start_time
                fps_values.append(1.0 / inference_time)
            
            # Draw using cached results (every frame)
            if last_results is not None and last_depth_map is not None:
                for score, label, box in zip(last_results["scores"], last_results["labels"], last_results["boxes"]):
                    box_np = box.cpu().numpy().astype(int)
                    class_name = detector.class_names[label.item()]
                    confidence = score.item()
                    
                    # Get distance
                    distance_info = detector.get_object_distance(last_depth_map, box_np)
                    color = distance_info['color']
                    
                    # Draw box
                    cv2.rectangle(display_frame, (box_np[0], box_np[1]), 
                                (box_np[2], box_np[3]), color, 2)
                    
                    # Draw label
                    label_text = f"{class_name}: {confidence:.2f}"
                    distance_text = f"{distance_info['distance_category'].upper()}"
                    
                    cv2.putText(display_frame, label_text, (box_np[0], box_np[1] - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(display_frame, distance_text, (box_np[0], box_np[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display FPS
            if fps_values:
                avg_fps = np.mean(fps_values[-30:])
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show smaller depth map
            if last_depth_map is not None:
                depth_viz = (last_depth_map * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_INFERNO)
                small_depth = cv2.resize(depth_colored, (80, 60))
                display_frame[10:70, display_frame.shape[1]-90:display_frame.shape[1]-10] = small_depth
            
            cv2.imshow('Optimized Detection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"data/output/capture_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"📸 Saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if fps_values:
            print(f"\n📊 Average FPS: {np.mean(fps_values):.2f}")
        print("✓ Done!")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
