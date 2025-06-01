import cv2
import tensorflow as tf
import numpy as np
import sys

# Check if image path is provided
if len(sys.argv) != 2:
    print("Usage: python classify_cookie.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Load the trained model
try:
    model = tf.keras.models.load_model('model3.1.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Define class names
class_names = ['Bad', 'Good']

def preprocess_image(img, target_size=(200, 200)):
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Load the image
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image.")
    sys.exit(1)

# Preprocess the image
preprocessed_image = preprocess_image(image)

# Make prediction
try:
    prediction = model.predict(preprocessed_image)
    class_idx = int(prediction[0][0] > 0.5)
    class_label = class_names[class_idx]
    print(class_label)
except Exception as e:
    print(f"Prediction error: {e}")
    sys.exit(1)
