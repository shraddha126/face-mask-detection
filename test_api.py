import requests
from PIL import Image
import numpy as np

# -----------------------------
# 1️⃣ Configuration
# -----------------------------
API_URL = "http://localhost:8000/predict/image"  # your FastAPI endpoint
IMAGE_PATH = "test_1.jpg"  # change this to your test image path

# -----------------------------
# 2️⃣ Send image to API
# -----------------------------
with open(IMAGE_PATH, "rb") as f:
    files = {"file": f}
    response = requests.post(API_URL, files=files)

# -----------------------------
# 3️⃣ Check response
# -----------------------------
if response.status_code != 200:
    print(f"Error: Status code {response.status_code}")
    print("Response:", response.text)
    exit(1)

data = response.json()
print("Raw model output:", data)

# -----------------------------
# 4️⃣ Interpret prediction
# -----------------------------
# Example: convert raw array to readable label
# Modify this according to your model's output format
prediction = np.array(data["prediction"])

# If your model returns class probabilities or labels, map accordingly
# Example: assume [NoMask, Mask] one-hot encoded or similar
if prediction.shape[-1] == 3:  # if output is RGB mask
    # simple heuristic: check green channel for "Mask"
    if prediction[0][1] > 128:
        label = "Mask"
    else:
        label = "No Mask"
else:
    # fallback: just show raw output
    label = str(prediction)

print(f"Prediction: {label}")
