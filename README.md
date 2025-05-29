# Cookie Quality Classification System

A real-time cookie quality classification system using TensorFlow and OpenCV that can classify cookies as "Good" or "Bad" through webcam feed, single image processing, or manual capture mode.

## Features

- **Real-time Classification**: Live webcam feed with continuous cookie quality detection
- **Manual Capture Mode**: Capture specific frames for classification with visual feedback
- **Single Image Processing**: Classify individual cookie images from file
- **Visual Feedback**: Colored bounding boxes and confidence scores
- **Model Inspection**: View trained model architecture

## Requirements

```
opencv-python>=4.5.0
tensorflow>=2.8.0
numpy>=1.21.0
```

## Installation

1. Clone or download this repository
2. Install the required dependencies:
```bash
pip install opencv-python tensorflow numpy
```
3. Ensure you have the trained model file `model3.1.h5` in the project directory

## Model Requirements

The system expects a trained TensorFlow model (`model3.1.h5`) with the following specifications:
- Input shape: (200, 200, 3) - RGB images resized to 200x200 pixels
- Output: Binary classification (0 for "Bad", 1 for "Good")
- Activation: Sigmoid output layer for binary classification

## Usage

### 1. Real-time Classification (`classification.py`)
Continuously classifies cookies through your webcam feed:

```bash
python classification.py
```

**Controls:**
- Press `q` to quit the application

**Features:**
- Live video feed with real-time classification
- Green text/box for "Good" cookies
- Red text/box for "Bad" cookies
- Confidence percentage displayed

### 2. Manual Capture Mode (`capture.py`)
Allows you to manually capture and classify specific frames:

```bash
python capture.py
```

**Controls:**
- Press `c` to capture and classify the current frame
- Press `q` to quit the application

**Features:**
- Live preview of camera feed
- Manual capture with instant classification
- Visual bounding box around classified image
- Saves processed images as `classified_image.jpg`

### 3. Single Image Classification (`onepic.py`)
Classify a single cookie image from file:

```bash
python onepic.py <image_path>
```

**Example:**
```bash
python onepic.py cookie_sample.jpg
```

**Output:** Prints either "Good" or "Bad" to console

### 4. Model Inspection (`cameraCheck.py`)
View the architecture of your trained model:

```bash
python cameraCheck.py
```

## Camera Configuration

The system uses different camera indices:
- `classification.py`: Camera index 2 (`cv2.VideoCapture(2)`)
- `capture.py`: Camera index 0 (`cv2.VideoCapture(0)`)

If you encounter camera access issues, modify the camera index in the respective files:
```python
cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc.
```

## Image Processing Pipeline

1. **Preprocessing**: Images are resized to 200x200 pixels and normalized to [0,1] range
2. **Prediction**: Model outputs a probability score
3. **Classification**: Threshold of 0.5 determines "Good" (>0.5) vs "Bad" (≤0.5)
4. **Visualization**: Results displayed with colored indicators and confidence scores

## File Structure

```
cookie-classifier/
├── classification.py      # Real-time classification
├── capture.py            # Manual capture mode
├── onepic.py            # Single image classification
├── cameraCheck.py       # Model architecture viewer
├── model3.1.h5          # Trained model (required)
├── classified_image.jpg # Output from capture mode
└── README.md           # This file
```

## Troubleshooting

### Common Issues

**Camera not found:**
- Check camera index in the code (try 0, 1, 2)
- Ensure camera is not being used by another application
- Check camera permissions

**Model loading error:**
- Verify `model3.1.h5` exists in the project directory
- Ensure the model file is not corrupted
- Check TensorFlow version compatibility

**UTF-8 encoding issues:**
- The code includes `sys.stdout.reconfigure(encoding='utf-8')` to handle encoding

**Prediction errors:**
- Ensure input images are valid
- Check if model expects specific input dimensions
- Verify image file format compatibility

## Model Training

This README assumes you have a pre-trained model (`model3.1.h5`). The model should be trained on cookie images with:
- Binary labels: 0 for "Bad" cookies, 1 for "Good" cookies
- Input size: 200x200 RGB images
- Appropriate data augmentation and preprocessing

## Performance Notes

- Real-time classification processes every frame, which may be CPU intensive
- Manual capture mode only processes when 'c' is pressed, saving computational resources
- Single image mode is most efficient for batch processing

## Contributing

When contributing to this project:
1. Maintain consistent image preprocessing across all scripts
2. Keep the same model input/output format
3. Follow the existing error handling patterns
4. Update this README for any new features

## License

[Add your license information here]

## Version History

- v1.0: Initial release with real-time classification, manual capture, and single image processing
