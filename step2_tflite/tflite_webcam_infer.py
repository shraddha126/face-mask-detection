import cv2
import numpy as np
import tensorflow as tf
import os
import time

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
TFLITE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mask_detector.tflite")

# Labels (must match training order)
class_names = ["mask_incorrect", "with_mask", "without_mask"]

IMG_SIZE = 224

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded successfully")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    label_id = np.argmax(preds)
    label = class_names[label_id]
    conf = preds[0][label_id]

    fps = 1 / (time.time() - start)

    cv2.putText(frame, f"{label} ({conf:.2f})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 0, 0), 2)

    cv2.imshow("TFLite Edge Mask Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
