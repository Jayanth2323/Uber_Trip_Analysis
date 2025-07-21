# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from app.model import load_model

app = FastAPI(
    title="Uber Trip Forecasting API",
    description="Forecast daily Uber trip counts using FOIL dataset features.",
    version="1.0.0",
)

class TripFeatures(BaseModel):
    hour: int         # Usually 0 for FOIL
    day: int
    day_of_week: int
    month: int
    active_vehicles: int

# Load model
try:
    model = load_model()
except Exception as e:
    model = None
    print("⚠️ Model load error:", e)

@app.get("/")
def root():
    return {"message": "🚀 Uber FOIL Trip Forecasting API is running!"}

@app.post("/predict")
def predict_trips(features: TripFeatures):
    if not model:
        return {"error": "Model not loaded."}

    try:
        input_data = np.array([[features.hour, features.day, features.day_of_week,
                                features.month, features.active_vehicles]])

        prediction = model.predict(input_data)[0]
        prediction = float(prediction)  # ✅ Convert from np.float32 to native float

        return {
            "predicted_trips": round(prediction, 2),
            "inputs": features.dict()
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
