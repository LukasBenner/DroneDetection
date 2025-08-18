from rfdetr import RFDETRSmall
import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import supervision as sv
import time
from collections import deque


def load_model(checkpoint_path="./checkpoints/small_checkpoint_best_ema.pth"):
    """Load and optimize the RF-DETR model for inference."""
    print(f"Loading model from: {checkpoint_path}")
    model = RFDETRSmall(pretrain_weights=checkpoint_path)
    model.optimize_for_inference()
    print("Model loaded and optimized for inference.")
    return model


def process_frame(model, frame, threshold=0.5, show_confidence=True):
    """
    Process a single video frame and return annotated result.
    
    Args:
        model: Loaded RF-DETR model
        frame: OpenCV frame (BGR format)
        threshold: Detection confidence threshold
        show_confidence: Whether to show confidence scores in labels
    
    Returns:
        annotated_frame: OpenCV frame with bounding boxes and labels
        detections: Raw detection results
    """
    # Convert BGR to RGB for PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Run detection
    detections = model.predict(pil_image, threshold=threshold)
    
    # Calculate optimal visualization parameters
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=pil_image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=pil_image.size)

    # Define color palette (yellow for drone detection)
    color = sv.ColorPalette.from_hex(["#ffff00"])

    # Create annotators
    bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.BLACK,
        text_scale=text_scale
    )

    # Annotate image with bounding boxes
    annotated_pil = bbox_annotator.annotate(pil_image.copy(), detections)
    
    # Add labels with confidence scores if requested
    if show_confidence and len(detections) > 0:
        labels = [
            f"Drone {confidence:.2f}" 
            for confidence in detections.confidence
        ]
        annotated_pil = label_annotator.annotate(annotated_pil, detections, labels)
    
    # Convert back to BGR for OpenCV
    annotated_frame = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
    
    return annotated_frame, detections


def draw_fps_info(frame, fps, detection_count, avg_fps_queue):
    """Draw FPS and detection information on frame."""
    height, width = frame.shape[:2]
    
    # Add FPS counter
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add average FPS over last 30 frames
    if len(avg_fps_queue) > 0:
        avg_fps = sum(avg_fps_queue) / len(avg_fps_queue)
        avg_fps_text = f"Avg FPS: {avg_fps:.1f}"
        cv2.putText(frame, avg_fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add detection count
    det_text = f"Detections: {detection_count}"
    cv2.putText(frame, det_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add instructions
    instructions = "Press 'q' to quit, 's' to save frame, 'r' to record"
    cv2.putText(frame, instructions, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def webcam_inference(model, camera_id=0, threshold=0.5, show_confidence=True, 
                    save_detections=False, output_dir="./detections"):
    """
    Run real-time inference on webcam feed.
    
    Args:
        model: Loaded RF-DETR model
        camera_id: Camera device ID (usually 0 for default webcam)
        threshold: Detection confidence threshold
        show_confidence: Whether to show confidence scores
        save_detections: Whether to automatically save frames with detections
        output_dir: Directory to save detection frames
    """
    # Initialize webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera {camera_id}")
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Adjust resolution as needed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Camera initialized:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Target FPS: {fps}")
    print(f"  Threshold: {threshold}")
    print(f"Controls:")
    print(f"  'q' - Quit")
    print(f"  's' - Save current frame")
    print(f"  'r' - Start/stop recording")
    print(f"  '+/-' - Increase/decrease threshold")
    
    # Create output directory if saving detections
    if save_detections and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize statistics
    frame_count = 0
    total_detections = 0
    frames_with_detections = 0
    last_time = time.time()
    fps_queue = deque(maxlen=30)  # Keep last 30 FPS measurements
    
    # Recording variables
    recording = False
    video_writer = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Calculate FPS
            current_time = time.time()
            fps_actual = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            fps_queue.append(fps_actual)
            last_time = current_time
            
            # Process frame
            try:
                annotated_frame, detections = process_frame(
                    model, frame, threshold, show_confidence
                )
                
                # Update statistics
                frame_count += 1
                num_detections = len(detections)
                total_detections += num_detections
                
                if num_detections > 0:
                    frames_with_detections += 1
                    
                    # Auto-save detections if enabled
                    if save_detections:
                        timestamp = int(time.time() * 1000)
                        save_path = os.path.join(output_dir, f"detection_{timestamp}.jpg")
                        cv2.imwrite(save_path, annotated_frame)
                        print(f"Saved detection: {save_path}")
                
                # Add FPS and detection info to frame
                display_frame = draw_fps_info(annotated_frame, fps_actual, num_detections, fps_queue)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                display_frame = draw_fps_info(frame, fps_actual, 0, fps_queue)
            
            # Record frame if recording
            if recording and video_writer is not None:
                video_writer.write(display_frame)
            
            # Display frame
            cv2.imshow('Drone Detection - Live Feed', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time() * 1000)
                save_path = f"frame_{timestamp}.jpg"
                cv2.imwrite(save_path, display_frame)
                print(f"Frame saved: {save_path}")
            elif key == ord('r'):
                # Toggle recording
                if not recording:
                    # Start recording
                    timestamp = int(time.time())
                    video_path = f"recording_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
                    recording = True
                    print(f"Started recording: {video_path}")
                else:
                    # Stop recording
                    recording = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    print("Stopped recording")
            elif key == ord('+') or key == ord('='):
                # Increase threshold
                threshold = min(1.0, threshold + 0.05)
                print(f"Threshold increased to: {threshold:.2f}")
            elif key == ord('-'):
                # Decrease threshold
                threshold = max(0.0, threshold - 0.05)
                print(f"Threshold decreased to: {threshold:.2f}")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        if frame_count > 0:
            avg_fps = sum(fps_queue) / len(fps_queue) if fps_queue else 0
            detection_rate = frames_with_detections / frame_count * 100
            avg_detections = total_detections / frame_count
            
            print(f"\nSession Statistics:")
            print(f"  Total frames processed: {frame_count}")
            print(f"  Average FPS: {avg_fps:.2f}")
            print(f"  Total detections: {total_detections}")
            print(f"  Frames with detections: {frames_with_detections}")
            print(f"  Detection rate: {detection_rate:.1f}%")
            print(f"  Average detections per frame: {avg_detections:.2f}")


def main():
    """Main function to run webcam inference from command line."""
    parser = argparse.ArgumentParser(description="Run RF-DETR inference on webcam feed")
    parser.add_argument("--camera", "-cam", type=int, default=0,
                       help="Camera device ID (default: 0)")
    parser.add_argument("--checkpoint", "-c", 
                       default="./checkpoints/small_checkpoint_best_ema.pth",
                       help="Path to model checkpoint (default: ./checkpoints/small_checkpoint_best_ema.pth)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--no-confidence", action="store_true",
                       help="Don't show confidence scores in labels")
    parser.add_argument("--save-detections", action="store_true",
                       help="Automatically save frames with detections")
    parser.add_argument("--output-dir", default="./detections",
                       help="Directory to save detection frames (default: ./detections)")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    try:
        # Load model
        model = load_model(args.checkpoint)
        
        # Start webcam inference
        webcam_inference(
            model=model,
            camera_id=args.camera,
            threshold=args.threshold,
            show_confidence=not args.no_confidence,
            save_detections=args.save_detections,
            output_dir=args.output_dir
        )
        
    except KeyboardInterrupt:
        print("\nWebcam inference interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()