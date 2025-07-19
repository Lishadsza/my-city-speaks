from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from utils.feature_extractor import extract_features
import joblib
import numpy as np
import uvicorn
import os
app = FastAPI()
app.add_middleware(
    CORSMiddleware,#cors as a part of frontennd for communication with servers.
    allow_origins=["*"],
    allow_credentials=True,# foorAllowing frontend requests
    allow_methods=["*"],
    allow_headers=["*"],
)
#loadmodel
model = joblib.load("random_forest_model.pkl")
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    temp_file_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(contents)

    try:
        features = extract_features(temp_file_path)
        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])[0]
        confidence = round(np.max(proba) * 100, 2)
        prediction_parts = prediction.split(" - ")#spliting predictions into parts


        result = {
            "language": prediction_parts[0],
            "city": prediction_parts[1],
            "state": prediction_parts[2],
            "confidence": round(confidence, 2)
        }
        return result
    finally:
        os.remove(temp_file_path)
