import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import time

# ---------------- CONFIG ----------------
IMG_SIZE = 224

# INT8 MODEL
MODEL_PATH = "/app/models/mask_detector_int8.tflite"

# FP32 MODEL (for comparison / benchmarking)
# MODEL_PATH = "/app/models/mask_detector_fp32.tflite"   # ← FP32 CHANGE

CLASS_NAMES = ["mask_incorrect", "with_mask", "without_mask"]
# ----------------------------------------

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Debug (run once, very useful)
print("Input dtype       :", input_details[0]["dtype"])
print("Input quantization:", input_details[0]["quantization"])
print("Output dtype      :", output_details[0]["dtype"])
print("Output quantization:", output_details[0]["quantization"])


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # ---------------- INT8 PREPROCESSING ----------------
    # INT8 / UINT8 models DO NOT use normalization
    img = img.astype(np.uint8)

    # ---------------- FP32 PREPROCESSING ----------------
    # img = img.astype(np.float32) / 255.0        # ← FP32 CHANGE
    # ----------------------------------------------------

    img = np.expand_dims(img, axis=0)
    return img


def predict(image_path):
    input_data = preprocess_image(image_path)

    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    # ---------------- INT8 OUTPUT DEQUANTIZATION ----------------
    if output_details[0]["dtype"] == np.uint8:
        scale, zero_point = output_details[0]["quantization"]
        preds = scale * (preds.astype(np.float32) - zero_point)

    # ---------------- FP32 OUTPUT ----------------
    # No dequantization needed for FP32
    # ------------------------------------------------------------

    class_id = np.argmax(preds)
    confidence = preds[class_id]

    return CLASS_NAMES[class_id], float(confidence)


# ---------------- MAIN ----------------
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python tflite_infer_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Make Docker-safe path
    if not os.path.isabs(image_path):
        image_path = os.path.join("/app", image_path)

    start = time.time()
    label, conf = predict(image_path)
    end = time.time()

    print("\nPrediction Result")
    print("------------------")
    print(f"Label          : {label}")
    print(f"Confidence     : {conf:.3f}")
    print(f"Inference time : {(end - start) * 1000:.2f} ms")
