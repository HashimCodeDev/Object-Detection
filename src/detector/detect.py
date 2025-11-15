from ultralytics import YOLO
import cv2

def main():
    """Real-time object detection using YOLOv8"""
    # Load YOLOv8 model
    model = YOLO('models/yolov8n.pt')  # Will auto-download on first run
    
    # Open webcam (0 for default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access camera")
        return
    
    print("Starting real-time object detection. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Run inference
            results = model(frame, verbose=False)
            
            # Render results on frame
            annotated_frame = results[0].plot()
            
            # Display
            cv2.imshow("YOLOv8 Object Detection - Fedora", annotated_frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
