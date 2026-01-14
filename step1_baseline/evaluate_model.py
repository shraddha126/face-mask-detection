import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- ABSOLUTE PATH FIX ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "mask_detector.keras")

print("Dataset path:", DATASET_PATH)
print("Exists:", os.path.exists(DATASET_PATH))

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Dataset path
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset")


# Image settings
IMG_SIZE = 224
BS = 32

# Data generator (NO augmentation here)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

testGen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BS,
    class_mode="categorical",
    shuffle=False
)

# Predictions
preds = model.predict(testGen)
y_pred = np.argmax(preds, axis=1)

# Class names
class_names = list(testGen.class_indices.keys())

# Report
print("\nClassification Report:\n")
print(classification_report(testGen.classes, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(testGen.classes, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
