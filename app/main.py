from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image
import numpy as np
import tensorflow as tf
import keras
# import io

app = FastAPI(title="Breast Cancer Histopathology Classifier API")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # When you deploy your frontend, you'll add its URL here too
    # e.g., "https://your-frontend-app.a.run.app" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

try:
    model = keras.models.load_model('models/breast_cancer_classifier.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocessing_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32)

        return tf.expand_dims(image, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid image file or processing error: {e}')

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

    preprocessed_image=preprocessing_image(image_bytes)

    prediction=model.predict(preprocessed_image)
    confidence=float(prediction)

    label="Malignant" if confidence > 0.5 else "Benign"
    confidence_score=confidence if label=='Malignant' else 1 - confidence

    return {
        "filename": file.filename, 
        "prediction": label, 
        "confidence": round(confidence_score, 4)
    }




