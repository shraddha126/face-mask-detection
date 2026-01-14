import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go UP one directory, then into models/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mask_detector.keras")
TFLITE_PATH = os.path.join(PROJECT_ROOT, "models", "mask_detector.tflite")

print("Model path:", MODEL_PATH)
print("Model exists:", os.path.exists(MODEL_PATH))

# Load Keras model
model = tf.keras.models.load_model(MODEL_PATH)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantization
tflite_model = converter.convert()

# Save TFLite model
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("TFLite model saved at:", TFLITE_PATH)
