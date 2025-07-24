# app/main.py (final dashboard with premium UI and interactive tabs)

import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from app.model import load_model
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

app = FastAPI(
    title="Uber Trip Forecasting API",
    description="Forecast daily Uber trip counts using FOIL dataset features + premium dashboard",
    version="3.0.0",
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
        ("Forecast Models", ["xgb_vs_actual", "rf_vs_actual", "ensemble_vs_actual"]),
        ("Exploration", ["trips_per_hour", "trips_per_day"]),
        ("Time Series", ["train_test_split", "decomposition"]),
        ("Explainability", ["shap_summary"])
    ]

    nav_tabs = ""
    tab_contents = ""
    for idx, (tab_name, plot_keys) in enumerate(plots):
        tab_id = f"tab{idx}"
        nav_tabs += f"<input type='radio' id='{tab_id}' name='tabs' {'checked' if idx == 0 else ''}><label for='{tab_id}'>{tab_name}</label>"

        html_inner = ""
        for plot in plot_keys:
            path = os.path.join("plots", f"{plot}.html")
            if os.path.exists(path):
                with open(path, "r") as f:
                    content = f.read()
                    body = content.split("<body>")[1].split("</body>")[0] if "<body>" in content else content
                    html_inner += f"<div class='plot-card'><h2>{plot.replace('_', ' ').title()}</h2>{body}</div>"
            else:
                html_inner += f"<div class='plot-card'><h2>{plot.replace('_', ' ').title()}</h2><p>❌ Plot not found</p></div>"

        tab_contents += f"<div class='tab-content' id='content{idx}'>{html_inner}</div>"

    html = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Uber Trip Analysis Dashboard</title>
        <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background: #ecf0f1; }}
            header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; font-size: 1.8em; }}
            .tabs {{ max-width: 1200px; margin: 0 auto; }}
            .tabs input[type="radio"] {{ display: none; }}
            .tabs label {{
                padding: 15px;
                background: #dfe6e9;
                cursor: pointer;
                font-weight: bold;
                border-bottom: 1px solid #b2bec3;
                display: inline-block;
                margin-right: 5px;
            }}
            .tabs label:hover {{ background: #b2bec3; }}
            .tabs .tab-content {{
                display: none;
                padding: 20px;
                background: white;
                border-radius: 0 0 8px 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            #tab0:checked ~ #content0,
            #tab1:checked ~ #content1,
            #tab2:checked ~ #content2,
            #tab3:checked ~ #content3 {{
                display: block;
            }}
            #tab0:checked + label,
            #tab1:checked + label,
            #tab2:checked + label,
            #tab3:checked + label {{
                background: #0984e3;
                color: white;
            }}
            .plot-card {{ margin-bottom: 40px; }}
            h2 {{ color: #0984e3; margin-bottom: 10px; }}
            footer {{ text-align: center; padding: 20px; background: #2c3e50; color: white; margin-top: 40px; }}
        </style>
    </head>
    <body>
        <header>📊 Uber Trip Forecasting Dashboard</header>
        <div class='tabs'>
            {nav_tabs}
            {tab_contents}
        </div>
        <footer>Built by Jayanth Chennoju | Tools: FastAPI, XGBoost, Plotly, SHAP, Render</footer>
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
