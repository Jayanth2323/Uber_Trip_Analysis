# app/main.py

import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from app.model import load_model
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

app = FastAPI(
    title="Uber Trip Forecasting API",
    description="Forecast daily Uber trip counts using FOIL dataset features + interactive plots",
    version="2.0.0",
)

class TripFeatures(BaseModel):
    hour: int
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

# ✅ Root route: show dashboard with all embedded plots
@app.get("/", response_class=HTMLResponse)
def dashboard():
    plots = [
        "trips_per_hour",
        "trips_per_day",
        "xgb_vs_actual",
        "rf_vs_actual",
        "ensemble_vs_actual"
    ]
    html_blocks = ""
    for plot in plots:
        path = os.path.join("plots", f"{plot}.html")
        if os.path.exists(path):
            with open(path, "r") as f:
                html_blocks += f.read()
        else:
            html_blocks += f"<h3>❌ {plot}.html not found</h3>"

    full_page = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Uber Trip Forecast Dashboard</title>
        <meta charset="utf-8">
    </head>
    <body>
        <h1 style="text-align:center;">📊 Uber Trip Analysis - Interactive Plots</h1>
        {html_blocks}
    </body>
    </html>
    """
    return HTMLResponse(content=full_page)

@app.post("/predict")
def predict_trips(features: TripFeatures):
    if not model:
        return {"error": "Model not loaded."}
    try:
        input_data = np.array([[features.hour, features.day, features.day_of_week,
                                features.month, features.active_vehicles]])
        prediction = model.predict(input_data)[0]
        return {
            "predicted_trips": round(float(prediction), 2),
            "inputs": features.dict()
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {
        "model_loaded": model is not None,
        "status": "✅ Model is ready!" if model else "❌ Model failed to load."
    }

@app.get("/metrics")
def get_metrics():
    return {
        "status": "Model metrics loaded successfully",
        "MAPE (%)": {
            "XGBoost": 8.37,
            "Random Forest": 9.61,
            "GBRT": 10.02,
            "Ensemble": 8.60
        }
    }

@app.get("/explain")
def get_shap_plot():
    path = os.path.join("plots", "shap_summary.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return {"error": "SHAP plot not found. Please generate it."}

@app.get("/plots/{plot_name}", response_class=HTMLResponse)
def serve_plot(plot_name: str):
    html_path = os.path.join("plots", f"{plot_name}.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())

    png_path = os.path.join("plots", f"{plot_name}.png")
    if os.path.exists(png_path):
        return FileResponse(png_path, media_type="image/png")

    return JSONResponse(status_code=404, content={"error": f"Plot {plot_name} not found."})
