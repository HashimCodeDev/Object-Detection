import torch
import cv2
import numpy as np
from transformers import (
    RTDetrForObjectDetection, 
    RTDetrImageProcessor,
    AutoImageProcessor,
    AutoModelForDepthEstimation
)
from PIL import Image
import time

class DroneObjectDistanceDetector:
    """
    Real-time object detection with distance estimation for drone applications.
    Combines RT-DETR for object detection and Depth Anything for distance estimation.
    """
    
    def __init__(
        self, 
        detection_model="PekingU/rtdetr_r50vd",
        depth_model="depth-anything/Depth-Anything-V2-Small-hf",
        confidence_threshold=0.5,
        device=None
    ):
        """
        Initialize detector with depth estimation.
        
        Args:
            detection_model: RT-DETR model for object detection
            depth_model: Depth estimation model
            confidence_threshold: Minimum confidence for detections
            device: 'cuda', 'cpu', or None
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load object detection model
        print(f"Loading detection model: {detection_model}...")
        self.detection_processor = RTDetrImageProcessor.from_pretrained(detection_model)
        self.detection_model = RTDetrForObjectDetection.from_pretrained(detection_model)
        self.detection_model.to(self.device)
        self.detection_model.eval()
        
        # Load depth estimation model
        print(f"Loading depth model: {depth_model}...")
        self.depth_processor = AutoImageProcessor.from_pretrained(depth_model)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model)
        self.depth_model.to(self.device)
        self.depth_model.eval()
        
        self.confidence_threshold = confidence_threshold
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        print("✓ Models loaded successfully!")
    
    def estimate_depth_map(self, image):
        """Generate depth map from image."""
        inputs = self.depth_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        # Normalize depth map
        output = prediction.squeeze().cpu().numpy()
        depth_min = output.min()
        depth_max = output.max()
        
        if depth_max - depth_min > np.finfo("float").eps:
            depth_map = (output - depth_min) / (depth_max - depth_min)
        else:
            depth_map = np.zeros(output.shape)
        
        return depth_map
    
    def get_object_distance(self, depth_map, bbox):
        """Calculate average distance of object from depth map."""
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Extract object region from depth map
        object_depth = depth_map[y1:y2, x1:x2]
        
        if object_depth.size == 0:
            return {'relative_distance': 0, 'distance_category': 'unknown', 'color': (128, 128, 128)}
        
        # Calculate statistics
        avg_depth = np.mean(object_depth)
        min_depth = np.min(object_depth)
        
        # Categorize distance (inverted because closer = darker in depth map)
        if avg_depth < 0.3:
            category = 'far'
            color = (255, 0, 0)  # Blue
        elif avg_depth < 0.6:
            category = 'medium'
            color = (0, 255, 255)  # Yellow
        else:
            category = 'close'
            color = (0, 0, 255)  # Red
        
        return {
            'relative_distance': avg_depth,
            'min_distance': min_depth,
            'distance_category': category,
            'color': color
        }
    
    def detect_webcam_with_distance(self, camera_index=0, display=True):
        """
        Real-time object detection with distance estimation from webcam.
        
        Args:
            camera_index: Camera index (usually 0 for laptop camera)
            display: Whether to display video window
        """
        cap = cv2.VideoCapture(camera_index)
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("\n" + "="*60)
        print("🎥 WEBCAM DETECTION STARTED")
        print("="*60)
        print("Controls:")
        print("  • Press 'q' to quit")
        print("  • Press 's' to save current frame")
        print("\nColor Coding:")
        print("  • RED box = Object is CLOSE")
        print("  • YELLOW box = Object is MEDIUM distance")
        print("  • BLUE box = Object is FAR")
        print("="*60 + "\n")
        
        fps_values = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                frame_count += 1
                start_time = time.time()
                
                # Convert to PIL Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Get depth map
                depth_map = self.estimate_depth_map(pil_image)
                
                # Detect objects
                inputs = self.detection_processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.detection_model(**inputs)
                
                # Post-process detections
                target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
                results = self.detection_processor.post_process_object_detection(
                    outputs,
                    threshold=self.confidence_threshold,
                    target_sizes=target_sizes
                )[0]
                
                # Draw detections with distance
                detection_count = 0
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box_np = box.cpu().numpy().astype(int)
                    class_name = self.class_names[label.item()]
                    confidence = score.item()
                    
                    # Get distance info
                    distance_info = self.get_object_distance(depth_map, box_np)
                    
                    # Draw bounding box with distance-based color
                    color = distance_info['color']
                    cv2.rectangle(frame, (box_np[0], box_np[1]), 
                                (box_np[2], box_np[3]), color, 2)
                    
                    # Draw label with distance
                    label_text = f"{class_name}: {confidence:.2f}"
                    distance_text = f"{distance_info['distance_category'].upper()}"
                    
                    # Background for text
                    cv2.rectangle(frame, (box_np[0], box_np[1] - 50), 
                                (box_np[0] + 200, box_np[1]), color, -1)
                    
                    cv2.putText(frame, label_text, (box_np[0] + 5, box_np[1] - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, distance_text, (box_np[0] + 5, box_np[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    detection_count += 1
                
                # Calculate and display FPS
                inference_time = time.time() - start_time
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                fps_values.append(current_fps)
                avg_fps = np.mean(fps_values[-30:])
                
                # Display info overlay
                overlay_height = 100
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (300, overlay_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Objects: {detection_count}", (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show depth map in corner
                depth_viz = (depth_map * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_INFERNO)
                small_depth = cv2.resize(depth_colored, (160, 120))
                
                # Place depth map in top-right corner
                frame[10:130, frame.shape[1]-170:frame.shape[1]-10] = small_depth
                cv2.putText(frame, "Depth Map", (frame.shape[1]-165, 145),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                if display:
                    cv2.imshow('Drone Object Detection with Distance', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n🛑 Quitting...")
                        break
                    elif key == ord('s'):
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename = f"data/output/capture_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"📸 Saved frame: {filename}")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n📊 Session Statistics:")
            print(f"  • Total frames processed: {frame_count}")
            print(f"  • Average FPS: {np.mean(fps_values):.2f}")
            print("✓ Camera released successfully")
