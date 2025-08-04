from rfdetr import RFDETRMedium
import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import supervision as sv
from tqdm import tqdm
import time


def load_model(checkpoint_path="./checkpoints/checkpoint_best_ema.pth"):
    """Load and optimize the RF-DETR model for inference."""
    print(f"Loading model from: {checkpoint_path}")
    model = RFDETRMedium(pretrain_weights=checkpoint_path)
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
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=pil_image.size)

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
        annotated_pil = label_annotator.annotate(
            annotated_pil, detections, labels)

    # Convert back to BGR for OpenCV
    annotated_frame = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

    return annotated_frame, detections


def process_video(model, input_path, output_path, threshold=0.5, show_confidence=True,
                  skip_frames=0, max_frames=None, display_live=False):
    """
    Process entire video and save annotated result.

    Args:
        model: Loaded RF-DETR model
        input_path: Path to input video
        output_path: Path to save output video
        threshold: Detection confidence threshold
        show_confidence: Whether to show confidence scores
        skip_frames: Number of frames to skip between processing (for speed)
        max_frames: Maximum number of frames to process (None for all)
        display_live: Show live preview while processing
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")

    # Determine frames to process
    if max_frames:
        frames_to_process = min(max_frames, total_frames)
    else:
        frames_to_process = total_frames

    if skip_frames > 0:
        frames_to_process = frames_to_process // (skip_frames + 1)
        print(
            f"  Processing every {skip_frames + 1} frames: {frames_to_process} frames total")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (frame_width, frame_height))

    # Statistics
    total_detections = 0
    frames_with_detections = 0

    # Process frames
    frame_count = 0
    processed_count = 0
    start_time = time.time()

    with tqdm(total=frames_to_process, desc="Processing video") as pbar:
        while cap.isOpened() and (max_frames is None or processed_count < max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames if specified
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue

            # Process frame
            try:
                annotated_frame, detections = process_frame(
                    model, frame, threshold, show_confidence
                )

                # Update statistics
                num_detections = len(detections)
                total_detections += num_detections
                if num_detections > 0:
                    frames_with_detections += 1

                # Write frame
                out.write(annotated_frame)

                # Display live preview if requested
                if display_live:
                    # Resize for display if too large
                    display_frame = annotated_frame
                    if frame_width > 1280:
                        scale = 1280 / frame_width
                        new_width = int(frame_width * scale)
                        new_height = int(frame_height * scale)
                        display_frame = cv2.resize(
                            annotated_frame, (new_width, new_height))

                    cv2.imshow('Drone Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Processing interrupted by user")
                        break

                processed_count += 1
                pbar.update(1)

                # Update progress bar description with current stats
                if processed_count % 30 == 0:  # Update every 30 frames
                    avg_detections = total_detections / processed_count
                    detection_rate = frames_with_detections / processed_count * 100
                    pbar.set_description(
                        f"Processing (avg: {avg_detections:.1f} det/frame, {detection_rate:.1f}% frames)")

            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                # Write original frame on error
                out.write(frame)
                processed_count += 1
                pbar.update(1)

            frame_count += 1

    # Cleanup
    cap.release()
    out.release()
    if display_live:
        cv2.destroyAllWindows()

    # Calculate processing time and FPS
    end_time = time.time()
    processing_time = end_time - start_time
    processing_fps = processed_count / processing_time if processing_time > 0 else 0

    # Print final statistics
    print(f"\nProcessing complete!")
    print(f"  Processed frames: {processed_count}")
    print(f"  Processing time: {processing_time:.2f} seconds")
    print(f"  Processing speed: {processing_fps:.2f} FPS")
    print(
        f"  Speed ratio: {processing_fps / fps:.2f}x real-time" if fps > 0 else "")
    print(f"  Total detections: {total_detections}")
    print(f"  Frames with detections: {frames_with_detections}")

    if processed_count > 0:
        print(
            f"  Average detections per frame: {total_detections / processed_count:.2f}")
        print(
            f"  Detection rate: {frames_with_detections / processed_count * 100:.1f}%")
    else:
        print("  No frames were processed")

    print(f"  Output saved to: {output_path}")


def main():
    """Main function to run video inference from command line."""
    parser = argparse.ArgumentParser(
        description="Run RF-DETR inference on a video")
    parser.add_argument("input_video", help="Path to the input video")
    parser.add_argument("output_video", help="Path to save the output video")
    parser.add_argument("--checkpoint", "-c",
                        default="./checkpoints/checkpoint_best_ema.pth",
                        help="Path to model checkpoint (default: ./checkpoints/checkpoint_best_ema.pth)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Number of frames to skip between processing (default: 0, process all frames)")
    parser.add_argument("--max-frames", type=int,
                        help="Maximum number of frames to process (default: all frames)")
    parser.add_argument("--no-confidence", action="store_true",
                        help="Don't show confidence scores in labels")
    parser.add_argument("--live-preview", action="store_true",
                        help="Show live preview while processing (press 'q' to stop)")

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input_video):
        print(f"Error: Input video not found: {args.input_video}")
        sys.exit(1)

    # Check if output directory exists
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    try:
        # Load model
        model = load_model(args.checkpoint)

        # Process video
        process_video(
            model=model,
            input_path=args.input_video,
            output_path=args.output_video,
            threshold=args.threshold,
            show_confidence=not args.no_confidence,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames,
            display_live=args.live_preview
        )

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
