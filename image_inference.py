from rfdetr import RFDETRMedium
import os
import sys
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import supervision as sv


def load_model(checkpoint_path="./checkpoints/checkpoint_best_ema.pth"):
    """Load and optimize the RF-DETR model for inference."""
    print(f"Loading model from: {checkpoint_path}")
    model = RFDETRMedium(pretrain_weights=checkpoint_path)
    model.optimize_for_inference()
    print("Model loaded and optimized for inference.")
    return model


def inference(model, image_path, threshold=0.5, show_confidence=True):
    """
    Run inference on a single image and return annotated result.

    Args:
        model: Loaded RF-DETR model
        image_path: Path to the input image
        threshold: Detection confidence threshold
        show_confidence: Whether to show confidence scores in labels

    Returns:
        annotated_image: PIL Image with bounding boxes and labels
        detections: Raw detection results
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Running inference on: {image_path}")

    # Load image
    image = Image.open(image_path)
    print(f"Image size: {image.size}")

    # Run detection
    detections = model.predict(image, threshold=threshold)
    print(f"Found {len(detections)} detections")

    # Calculate optimal visualization parameters
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

    # Define color palette
    color = sv.ColorPalette.from_hex([
        "#ffff00"
    ])

    # Create annotators
    bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.BLACK,
        text_scale=text_scale
    )

    # Annotate image with bounding boxes
    annotated_image = bbox_annotator.annotate(image.copy(), detections)

    # Add labels with confidence scores if requested
    if show_confidence and len(detections) > 0:
        labels = [
            f"Detection {confidence:.2f}"
            for confidence in detections.confidence
        ]
        annotated_image = label_annotator.annotate(
            annotated_image, detections, labels)

    return annotated_image, detections


def display_image(image, title="Detection Results"):
    """Display image using matplotlib."""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def save_result(image, output_path):
    """Save the annotated image to specified path."""
    image.save(output_path)
    print(f"Result saved to: {output_path}")


def main():
    """Main function to run inference from command line."""
    parser = argparse.ArgumentParser(
        description="Run RF-DETR inference on an image")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--checkpoint", "-c",
                        default="./checkpoints/checkpoint_best_ema.pth",
                        help="Path to model checkpoint (default: ./checkpoints/checkpoint_best_ema.pth)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--output", "-o",
                        help="Output path to save annotated image (optional)")
    parser.add_argument("--no-display", action="store_true",
                        help="Don't display the image (useful for batch processing)")
    parser.add_argument("--no-confidence", action="store_true",
                        help="Don't show confidence scores in labels")

    args = parser.parse_args()

    try:
        # Load model
        model = load_model(args.checkpoint)

        # Run inference
        annotated_image, detections = inference(
            model,
            args.image_path,
            threshold=args.threshold,
            show_confidence=not args.no_confidence
        )

        # Print detection summary
        if len(detections) > 0:
            print(f"\nDetection Summary:")
            for i, conf in enumerate(detections.confidence):
                bbox = detections.xyxy[i]
                print(
                    f"  Detection {i+1}: confidence={conf:.3f}, bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        else:
            print("No detections found.")

        # Save result if output path specified
        if args.output:
            save_result(annotated_image, args.output)

        # Display image unless --no-display is specified
        if not args.no_display:
            display_image(annotated_image,
                          f"Detections: {os.path.basename(args.image_path)}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
