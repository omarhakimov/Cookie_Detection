

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the frame in real-time
    cv2.imshow('Phone Camera - Press "c" to Capture', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Press 'c' to capture and process the image
        # Save the captured frame temporarily
        captured_frame = frame.copy()

        # Preprocess the captured frame
        preprocessed_image = preprocess_image(captured_frame)

        # Make prediction
        try:
            prediction = model.predict(preprocessed_image)
            class_idx = int(prediction[0][0] > 0.5)  # Threshold at 0.5
            class_label = class_names[class_idx]
            confidence = prediction[0][0] if class_idx == 1 else 1 - prediction[0][0]
        except Exception as e:
            print(f"Prediction error: {e}")
            break

        # Add a bounding box and label to the image
        label = f"{class_label} ({confidence*100:.2f}%)"
        box_color = (0, 255, 0) if class_label == 'Good' else (0, 0, 255)

        # Draw bounding box around the whole frame (for simplicity)
        cv2.rectangle(captured_frame, (10, 10), (captured_frame.shape[1] - 10, captured_frame.shape[0] - 10), box_color, 2)

        # Add label to the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        cv2.putText(captured_frame, label, (20, 40), font, font_scale, box_color, thickness)

        # Display the processed frame
        cv2.imshow('Captured Image - Classification', captured_frame)

        # Save the processed image
        cv2.imwrite('classified_image.jpg', captured_frame)
        print("Image processed and saved as 'classified_image.jpg'.")

    elif key == ord('q'):  # Press 'q' to quit
        print("Exiting the program.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
