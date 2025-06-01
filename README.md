# Cookie Detection Project

A Python-based computer vision application that uses a trained TensorFlow model to classify cookies as **Good** or **Bad** in real time or from static images. Leveraging OpenCV for image capture and preprocessing, the project demonstrates end-to-end deployment of a deep learning model for simple object classification tasks.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model](#model)
- [Usage](#usage)
  - [1. Model Architecture Check](#1-model-architecture-check)
  - [2. Real-Time Webcam Classification](#2-real-time-webcam-classification)
  - [3. Interactive Capture Mode](#3-interactive-capture-mode)
  - [4. Single Image Classification](#4-single-image-classification)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Load and inspect the trained model architecture.
- Real-time classification from a webcam feed.
- Interactive capture mode with visual annotations and confidence scores.
- Batch classification of single images via command line.

## Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x
- OpenCV
- NumPy

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cookie-detection.git
   cd cookie-detection
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   .\venv\Scripts\activate  # On Windows
   ```

3. Install required Python packages:
   ```bash
   pip install tensorflow opencv-python numpy
   ```

## Model

Place the trained model file `model3.1.h5` in the project root directory. This model was trained to distinguish between "Good" and "Bad" cookies based on image features.

## Usage

### 1. Model Architecture Check

Inspect the model layers and parameters:

```bash
python cameraCheck.py
```

### 2. Real-Time Webcam Classification

Start real-time classification on your default camera (index may vary):

```bash
python classification.py
```

Press `q` to quit the live window.

### 3. Interactive Capture Mode

Capture frames on-demand, overlay bounding box and label, then save the annotated image:

```bash
python capture.py
```

- Press `c` to capture and classify the current frame.
- Press `q` to exit.

### 4. Single Image Classification

Classify a static image file and print the predicted label:

```bash
python onepic.py path/to/image.jpg
```


## Project Structure

```
cookie-detection/
├── model3.1.h5          # Trained TensorFlow model
├── cameraCheck.py       # Print model summary
├── classification.py    # Real-time webcam classification
├── capture.py           # Interactive capture & classification
├── onepic.py            # Single image classification script
└── README.md            # Project documentation
```