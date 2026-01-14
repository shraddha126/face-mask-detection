from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from app.inference import MaskDetector
import io

# -----------------------------
# 1️⃣ Create FastAPI app
# -----------------------------
app = FastAPI(
    title="Face Mask Detection API",
    description="Detects face masks from uploaded images using TFLite",
    version="1.0"
)

# -----------------------------
# 2️⃣ Initialize the TFLite detector on startup
# -----------------------------
detector = None

@app.on_event("startup")
def load_model():
    global detector
    try:
        # Make sure this matches your actual TFLite filename
        detector = MaskDetector("models/mask_detector_int8.tflite")
    except FileNotFoundError:
        raise RuntimeError("Model file not found in /models folder!")

# -----------------------------
# 3️⃣ Health check endpoint
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# 4️⃣ Predict endpoint
# -----------------------------
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read uploaded file and convert to PIL Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run prediction using MaskDetector
        result = detector.predict(image)

        # Convert result to list for JSON
        return {"prediction": result.tolist()}

    except Exception as e:
        # Catch any unexpected error and return 500
        raise HTTPException(status_code=500, detail=str(e))
