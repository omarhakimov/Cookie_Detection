import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model3.1.h5')  # Ensure this path is correct

# Display the model architecture
model.summary()
