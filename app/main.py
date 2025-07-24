# app/main.py (final dashboard with all plots including SHAP, decomposition, train/test)

import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from app.model import load_model
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

app = FastAPI(
    title="Uber Trip Forecasting API",
    description="Forecast daily Uber trip counts using FOIL dataset features + interactive dashboard",
    version="2.0.1",
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

@app.get("/", response_class=HTMLResponse)
def dashboard():
    plots = [
        "trips_per_hour", "trips_per_day",
        "xgb_vs_actual", "rf_vs_actual", "ensemble_vs_actual",
        "train_test_split", "decomposition", "shap_summary"
    ]

    html_blocks = ""
    for plot in plots:
        path = os.path.join("plots", f"{plot}.html")
        if os.path.exists(path):
            with open(path, "r") as f:
                body = f.read()
                inner = body.split("<body>")[1].split("</body>")[0] if "<body>" in body else body
                html_blocks += f"""
                <section class="plot-section">
                    <h2>{plot.replace('_', ' ').title()}</h2>
                    {inner}
                </section>
                """
        else:
            html_blocks += f"<h3>❌ {plot}.html not found</h3>"

    # === UI Template ===
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Uber Trip Forecasting Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
                color: #2c3e50;
            }}
            header {{
                background: #2d3436;
                color: #fff;
                padding: 20px;
                text-align: center;
                font-size: 2em;
                letter-spacing: 1px;
            }}
            .plot-section {{
                margin: 40px auto;
                padding: 20px;
                max-width: 1100px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            h2 {{
                margin-top: 0;
                text-align: center;
                color: #0984e3;
            }}
        </style>
    </head>
    <body>
        <header>📊 Uber Trip Forecasting Dashboard</header>
        {html_blocks}
    </body>
    </html>
    """
    return HTMLResponse(content=html)


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
