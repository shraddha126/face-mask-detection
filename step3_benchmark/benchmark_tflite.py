import tensorflow as tf
import numpy as np
import cv2
import time
import os

IMG_SIZE = 224
IMAGE_PATH = "/app/test.jpg"

MODELS = {
    "FP32": "/app/models/mask_detector_fp32.tflite",
    "INT8": "/app/models/mask_detector_int8.tflite"
}

def load_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess(image_path, dtype):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    if dtype == np.uint8:
        img = img.astype(np.uint8)           # INT8
    else:
        img = img.astype(np.float32) / 255.0 # FP32

    img = np.expand_dims(img, axis=0)
    return img

def benchmark(model_name, model_path, runs=100):
    interpreter = load_interpreter(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]

    input_data = preprocess(IMAGE_PATH, input_dtype)

    # Warm-up
    for _ in range(10):
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

    start = time.time()
    for _ in range(runs):
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
    end = time.time()

    avg_latency = (end - start) / runs * 1000
    fps = 1000 / avg_latency

    return avg_latency, fps

# ---------------- MAIN ----------------
if __name__ == "__main__":

    print("\n📊 TFLite Benchmark Results")
    print("=" * 40)

    for name, path in MODELS.items():
        latency, fps = benchmark(name, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)

        print(f"\n{name} MODEL")
        print(f"Model size     : {size_mb:.2f} MB")
        print(f"Avg latency    : {latency:.2f} ms")
        print(f"FPS            : {fps:.2f}")
