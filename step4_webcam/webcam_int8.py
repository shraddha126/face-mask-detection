import tensorflow as tf
import numpy as np
import cv2
import time

# ---------------- CONFIG ----------------
MODEL_PATH = "models/mask_detector_int8.tflite"
IMG_SIZE = 224
CLASS_NAMES = ["mask_incorrect", "with_mask", "without_mask"]
# ---------------------------------------

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.uint8)                # INT8 input
    img = np.expand_dims(img, axis=0)

    start = time.time()
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    end = time.time()

    fps = 1 / (end - start)

    class_id = np.argmax(preds)
    label = CLASS_NAMES[class_id]
    conf = preds[class_id]

    cv2.putText(frame, f"INT8 | {label} ({conf:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow("INT8 Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
