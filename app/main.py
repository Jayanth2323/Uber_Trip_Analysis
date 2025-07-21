# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pickle
import os

# Initialize FastAPI app
app = FastAPI(
    title="Uber Trip Forecasting API",
    description="Predict Uber trip demand using trained ML models.",
    version="1.0.0",
)

# Define request structure using Pydantic
class TripFeatures(BaseModel):
    hour: int
    day: int
    day_of_week: int
    month: int
    lat: float
    lon: float
    base_code: Optional[str] = None

# Load model on startup
MODEL_PATH = os.path.join("models", "xgb_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("⚠️ Warning: Model not found. Please train and save model at:", MODEL_PATH)

# Health check
@app.get("/")
def root():
    return {"message": "🚀 Uber Trip Forecasting API is up and running!"}

# Prediction endpoint
@app.post("/predict")
def predict_trip_demand(features: TripFeatures):
    if not model:
        return {"error": "Model not loaded. Please upload trained model to 'models/xgb_model.pkl'."}

    input_array = np.array([[features.hour, features.day, features.day_of_week, features.month,
                             features.lat, features.lon]])

    prediction = model.predict(input_array)[0]

    return {
        "predicted_trip_count": round(prediction, 2),
        "inputs": features.dict()
    }
