# app/model.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from app.model import load_model

# Paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")


def load_data(filepath: str) -> pd.DataFrame:
    """Load Uber FOIL CSV format with aggregated trip data"""
    df = pd.read_csv(filepath)

    # Convert 'date' column to datetime
    df['datetime'] = pd.to_datetime(df['date'])

    # Extract temporal features
    df['Hour'] = 0  # Since it's daily aggregate
    df['Day'] = df['datetime'].dt.day
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['Month'] = df['datetime'].dt.month

    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """Prepare features and target for FOIL-style dataset"""
    X = df[['Hour', 'Day', 'DayOfWeek', 'Month', 'active_vehicles']]
    y = df['trips']
    return train_test_split(X, y, test_size=0.3, random_state=42)


def train_and_save_model(filepath: str = "data/Uber-Jan-Feb-FOIL.csv") -> None:
    """Train and save model"""
    df = load_data(filepath)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = XGBRegressor(objective="reg:squarederror", n_estimators=300, max_depth=6, learning_rate=0.1)
    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model trained and saved to {MODEL_PATH}")


def load_model() -> XGBRegressor:
    """Load trained model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


app = FastAPI(
    title="Uber Trip Forecasting API",
    description="Forecast daily Uber trip counts based on vehicle activity and time features.",
    version="1.0.0",
)

# Aligned with FOIL dataset features
class TripFeatures(BaseModel):
    hour: int  # Will usually be 0 (aggregated)
    day: int
    day_of_week: int
    month: int
    active_vehicles: int

# Load model
try:
    model = load_model()
except Exception as e:
    model = None
    print("⚠️ Model loading error:", str(e))

@app.get("/")
def root():
    return {"message": "🚀 Uber Trip Forecasting API (FOIL version) is running!"}

@app.post("/predict")
def predict_trips(features: TripFeatures):
    if not model:
        return {"error": "Model not loaded. Please train the model first."}

    input_array = np.array([[features.hour, features.day, features.day_of_week,
                             features.month, features.active_vehicles]])
    
    prediction = model.predict(input_array)[0]

    return {
        "predicted_trips": round(prediction, 2),
        "inputs": features.dict()
    }
