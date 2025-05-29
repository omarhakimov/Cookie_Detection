import cv2
import tensorflow as tf
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load the trained model
try:
    model = tf.keras.models.load_model('model3.1.h5')  # Replace with the path to your trained model
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# Define class names
class_names = ['Bad', 'Good']

def preprocess_image(img, target_size=(200, 200)):
    """
    Preprocesses the image for model prediction.
    """
    img = cv2.resize(img, target_size)  # Resize to the model's input size
    img = img.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Initialize webcam (replace `0` with your camera index if necessary)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the frame
    img_array = preprocess_image(frame)

    # Make prediction
    try:
        prediction = model.predict(img_array)
        class_idx = int(prediction[0][0] > 0.5)  # Threshold at 0.5
        class_label = class_names[class_idx]
        confidence = prediction[0][0] if class_idx == 1 else 1 - prediction[0][0]
    except Exception as e:
        print(f"Prediction error: {e}")
        break

    # Display the prediction on the frame
    label = f"{class_label} ({confidence*100:.2f}%)"
    color = (0, 255, 0) if class_label == 'Good' else (0, 0, 255)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Real-Time Cookie Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
