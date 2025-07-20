from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
import io
import numpy as np

app = FastAPI(title="Breast Cancer Histopathology Classifier API")

try:
    model = keras.models.load_model('models/breast_cancer_classifier.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocessing_image(image_bytes: bytes) -> np.ndarray:
    try:
        image=Image.open(io.BytesIO(image_bytes))
        if image.mode!='RGB':
            image=image.convert('RGB')
        
        image=image.resize((224, 224))
        image_array=np.array(image)
        # preprocessed_array = keras.applications.resnet_v2.preprocess_input(image_array)

        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid image file: {e}')

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




