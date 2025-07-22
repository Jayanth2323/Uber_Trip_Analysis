import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from app.model import load_model

app = FastAPI(
    title="Uber Trip Forecasting API",
    description="Forecast hourly Uber trip counts and serve analysis plots.",
    version="1.0.0",
)

# === Prediction Schema ===
class TripFeatures(BaseModel):
    hour: int         # 0-23
    day: int          # 1-31
    day_of_week: int  # 0-6
    month: int        # 1-12
    active_vehicles: int

# === Load Model ===
try:
    model = load_model()
except Exception as e:
    model = None
    print("⚠️ Model load error:", e)

@app.get("/")
def root():
    return {"message": "🚀 Uber Trip Forecasting API is running!"}

@app.post("/predict")
def predict_trips(features: TripFeatures):
    if not model:
        return {"error": "Model not loaded."}
    data = np.array([[features.hour, features.day, features.day_of_week,
                      features.month, features.active_vehicles]])
    try:
        pred = float(model.predict(data)[0])
        return {"predicted_trips": round(pred, 2), "inputs": features.dict()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"model_loaded": model is not None,
            "status": "✅ Model is ready!" if model else "❌ Model failed to load."}

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
    # add other plot filenames here as generated
}

for route, filename in PLOT_FILES.items():
    def _make_endpoint(fname):
        def _endpoint():
            p = os.path.join("plots", fname)
            if os.path.exists(p):
                return FileResponse(p, media_type="image/png")
            return {"error": f"Plot {fname} not found."}
        return _endpoint
    app.get(f"/plots/{route}")(_make_endpoint(filename))
