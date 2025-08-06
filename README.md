# 🚁 Drone Detection with RF-DETR

A comprehensive drone detection system using RF-DETR (Real-time DEtection TRansformer) for training and inference on images, videos, and live webcam feeds.

## 📂 Project Structure

```
DroneDetection/
├── finetune_rf_detr_runpod.ipynb      # RunPod training notebook
├── rf_detr_local_inference.ipynb      # Local inference notebook
├── image_inference.py                 # Image inference script (alternative)
├── video_inference.py                 # Video processing script
├── video_stream_interference.py       # Real-time webcam inference
├── requirements.txt                   # Python dependencies
├── dataset/                           # Training dataset (COCO format)
├── output/                            # Training outputs and checkpoints
└── checkpoints/                       # Model checkpoints
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training and fast inference)
- Webcam (for real-time detection)

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/LukasBenner/DroneDetection.git
   cd DroneDetection
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv drone-detection
   source drone-detection/bin/activate  # On Windows: drone-detection\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify GPU setup (optional):**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## 📊 Dataset Setup

Organize your dataset in COCO format:
```
dataset/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── test/
│   ├── _annotations.coco.json
│   └── *.jpg
└── valid/
    ├── _annotations.coco.json
    └── *.jpg
```

## 🎯 Training

### RunPod Training
1. Upload `finetune_rf_detr_runpod.ipynb` to your RunPod instance
2. Ensure your dataset is in `/workspace/data`
3. Run the notebook cells

## 🔍 Inference

### Image Inference
Process a single image:

```bash
# Basic usage
python image_inference.py path/to/image.jpg

# With custom threshold
python image_inference.py image.jpg --threshold 0.3

# Save result
python image_inference.py image.jpg --output detected_image.jpg

# Use custom checkpoint
python image_inference.py image.jpg --checkpoint ./output/checkpoint_best_ema.pth
```

**Options:**
- `--checkpoint`, `-c`: Path to model checkpoint
- `--threshold`, `-t`: Detection confidence threshold (0.0-1.0)
- `--output`, `-o`: Save annotated image
- `--no-display`: Don't show the image
- `--no-confidence`: Hide confidence scores

### Video Processing
Process entire videos:

```bash
# Basic usage
python video_inference.py input.mp4 output.mp4

# Fast processing (skip frames)
python video_inference.py input.mp4 output.mp4 --skip-frames 2

# Process limited frames
python video_inference.py input.mp4 output.mp4 --max-frames 300

# Live preview
python video_inference.py input.mp4 output.mp4 --live-preview
```

**Options:**
- `--checkpoint`, `-c`: Path to model checkpoint
- `--threshold`, `-t`: Detection confidence threshold
- `--skip-frames`: Skip frames for faster processing
- `--max-frames`: Limit number of frames to process
- `--live-preview`: Show processing in real-time
- `--no-confidence`: Hide confidence scores

### Real-time Webcam Detection
Live drone detection from webcam:

```bash
# Default webcam
python video_stream_interference.py

# External camera
python video_stream_interference.py --camera 1

# Auto-save detections
python video_stream_interference.py --save-detections

# Custom threshold
python video_stream_interference.py --threshold 0.3
```

**Interactive Controls:**
- `q`: Quit application
- `s`: Save current frame
- `r`: Start/stop recording
- `+/-`: Increase/decrease threshold in real-time

**Options:**
- `--camera`: Camera device ID (0, 1, 2, ...)
- `--checkpoint`, `-c`: Path to model checkpoint
- `--threshold`, `-t`: Detection confidence threshold
- `--save-detections`: Auto-save frames with detections
- `--output-dir`: Directory for saved detections

## 📈 Performance Tips

### Training
- Adjust `batch_size` based on GPU memory
- Use `grad_accum_steps` to simulate larger batch sizes
- Monitor training with TensorBoard logs

### Inference
- Use `--skip-frames` for faster video processing
- Lower `--threshold` for more sensitive detection
- Use `--no-display` for batch processing

## 🔧 Model Checkpoints

Training generates several checkpoint files:
- `checkpoint_best_ema.pth`: Best model with exponential moving average (recommended)
- `checkpoint_best_total.pth`: Best overall model
- `checkpoint.pth`: Latest checkpoint

## 📝 Usage Examples

### Quick Start
```bash
# Train model (in Jupyter notebook)
jupyter notebook fintune_rf_detr.ipynb

# Test on image
python image_inference.py test_image.jpg --output result.jpg

# Process video
python video_inference.py input_video.mp4 output_video.mp4

# Real-time detection
python video_stream_interference.py
```

### Batch Processing
```bash
# Process multiple videos without display
for video in *.mp4; do
    python video_inference.py "$video" "detected_$video" --no-display
done
```

### Performance Testing
```bash
# Test on first 100 frames only
python video_inference.py test.mp4 output.mp4 --max-frames 100 --live-preview

# Fast preview (every 3rd frame)
python video_inference.py test.mp4 output.mp4 --skip-frames 2
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in training
   - Use `grad_accum_steps` for effective larger batches

2. **Low FPS in Real-time**
   - Increase `--threshold` value
   - Use lower camera resolution

3. **Camera Not Found**
   - Try different `--camera` values (0, 1, 2)
   - Check camera permissions

4. **Import Errors**
   - Reinstall dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility