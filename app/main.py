from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import keras
import io
from PIL import Image

app = FastAPI(title="Breast Cancer Histopathology Classifier API")

# Allow frontend to connect
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
MODEL_PATH = 'models/breast_cancer_classifier_cpu_final.keras'
model = None
try:
    # The model's preprocessing is built-in, so no custom objects here.
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # The app will still run, but endpoints will report the model is unavailable.

# --- Image Preprocessing (Corrected) ---
def preprocessing_image(image_bytes: bytes) -> np.ndarray:
    """
    Loads image bytes and prepares it for the model.
    steps:
    1. Loads image with Pillow.
    2. Resizes to (224, 224).
    3. Converts to NumPy array with pixel values in [0, 255].
    It does NOT apply ResNet preprocessing since the model expects raw images.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        pil_image_resized = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
        image_array = tf.keras.preprocessing.image.img_to_array(pil_image_resized)
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid image file or processing error: {e}')

# --- API Endpoints ---
@app.get("/")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    return {"status": "ok", "message": "API is running and model is loaded."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")

    image_bytes = await file.read()
    preprocessed_image = preprocessing_image(image_bytes)

    # The model expects a batch of images, which preprocessed_image provides.
    prediction = model.predict(preprocessed_image)
    
    # The output is a 2D array, e.g., [[0.98]], so we access the value.
    confidence = float(prediction[0][0])

    if confidence > 0.5:
        label = "Malignant"
        confidence_score = confidence
    else:
        label = "Benign"
        confidence_score = 1 - confidence

    return {
        "prediction": label,
        "confidence": f"{confidence_score:.4f}" # Format to 4 decimal places
    }

