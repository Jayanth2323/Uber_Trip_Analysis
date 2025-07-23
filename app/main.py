# app/main.py

import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from app.model import load_model
from fastapi.responses import FileResponse

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

@app.get("/health")
def health_check():
    return {
        "model_loaded": model is not None,
        "status": "✅ Model is ready!" if model else "❌ Model failed to load."
    }

# === Metrics ===
model_metrics = {
    "XGBoost": 8.37,
    "Random Forest": 9.61,
    "GBRT": 10.02,
    "Ensemble": 8.60
}

@app.get("/metrics")
def get_metrics():
    return {"status": "Model metrics loaded successfully",
            "MAPE (%)": model_metrics}

# === Explainability Plot ===
@app.get("/explain")
def get_shap_plot():
    path = os.path.join("plots", "shap_summary.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return {"error": "SHAP plot not found. Please generate it."}

# === Analysis Plot Endpoints ===
PLOT_FILES = {
    "trips_per_hour": "trips_per_hour.png",
    "trips_per_day": "trips_per_day.png",
    "decomposition": "decomposition.png",
    "train_test_split": "train_test_split.png",
    "xgb_vs_actual": "xgb_vs_actual.png",
    "rf_vs_actual": "rf_vs_actual.png",
    "ensemble_vs_actual": "ensemble_vs_actual.png",
    # add other plot filenames here as generated
}

def _make_endpoint(fname):
    def _endpoint():
        p = os.path.join("plots", fname)
        if os.path.exists(p):
            return FileResponse(p, media_type="image/png")
        return {"error": f"Plot {fname} not found."}
    return _endpoint

for route, filename in PLOT_FILES.items():
    app.add_api_route(f"/plots/{route}", _make_endpoint(filename), methods=["GET"])
