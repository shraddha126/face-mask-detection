import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load trained model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "mask_detector.keras")

print("Loading model from:", MODEL_PATH)
print("Model exists:", os.path.exists(MODEL_PATH))

model = tf.keras.models.load_model(MODEL_PATH)


PRED_HISTORY = deque(maxlen = 5)  # Last 5

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("🎥 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face = frame[y + h//3 : y + h, x : x + w]
        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_array = np.array(face_resized, dtype="float32")
        face_array = preprocess_input(face_array)
        face_array = np.expand_dims(face_array, axis=0)

        preds = model.predict(face_array)[0]
        idx = np.argmax(preds)

        labels = [ "Incorrect Mask","Mask","No Mask"]
        current_label = labels[idx]
        current_conf = preds[idx]

        PRED_HISTORY.append((current_label, current_conf))

        # Majority vote
        labels_only = [l for l, _ in PRED_HISTORY]
        label = max(set(labels_only), key=labels_only.count)

# Average confidence for that label
        confidence = sum(c for l, c in PRED_HISTORY if l == label) / labels_only.count(label)

        if confidence < 0.75:
            label = "Uncertain"

        COLORS = {
                 "Mask": (0, 255, 0),
                 "No Mask": (0, 0, 255),
                 "Incorrect Mask": (0, 255, 255),
                 "Uncertain": (255, 255, 255)
        }

        color = COLORS[label]

        cv2.putText(
            frame,
            f"{label}: {confidence*100:.2f}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d  %H:%M:%S")  
    
    (h, w) = frame.shape[:2]
    cv2.putText(
        frame, timestamp,(w - 260, 30),  # adjust if needed
        cv2.FONT_HERSHEY_SIMPLEX,0.6,
        (255, 255, 255),2)
   

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
