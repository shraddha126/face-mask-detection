import tensorflow as tf
import numpy as np
import os
import cv2
import random

# -------- CONFIG --------
IMG_SIZE = 224
BATCH_SIZE = 1
CALIBRATION_SAMPLES = 200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "mask_detector.keras")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "models", "mask_detector_int8.tflite")
# ------------------------

def representative_data_gen():
    image_paths = []

    for cls in os.listdir(DATASET_DIR):
        cls_path = os.path.join(DATASET_DIR, cls)
        if os.path.isdir(cls_path):
            for img in os.listdir(cls_path):
                image_paths.append(os.path.join(cls_path, img))

    random.shuffle(image_paths)
    image_paths = image_paths[:CALIBRATION_SAMPLES]

    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        yield [img]

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable INT8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Force INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert
tflite_model = converter.convert()

# Save
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print("✅ INT8 TFLite model saved at:")
print(OUTPUT_PATH)
