# app/main.py (final dashboard with premium UI, interactive tabs, PDF export)

import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from app.model import load_model
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fpdf import FPDF

app = FastAPI(
   title="Uber Trip Forecasting API",
   description="Forecast daily Uber trip counts using FOIL dataset features + premium dashboard",
   version="3.1.0",
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
   print("âš ï¸ Model load error:", e)

@app.get("/", response_class=HTMLResponse)
def dashboard():
   plots = [
       ("Forecast Models", ["xgb_vs_actual", "rf_vs_actual", "ensemble_vs_actual"]),
       ("Exploration", ["trips_per_hour", "trips_per_day"]),
       ("Time Series", ["train_test_split", "decomposition"]),
       ("Explainability", ["shap_summary"])
   ]

   # Load plot blocks
   tab_headers = ""
   tab_contents = ""
   for idx, (tab_name, plot_keys) in enumerate(plots):
       active_class = "active" if idx == 0 else ""
       tab_id = f"tab{idx}"
       tab_headers += f"<li class='{active_class}' data-tab='{tab_id}'>{tab_name}</li>"
       tab_html = ""
       for plot in plot_keys:
           path = os.path.join("plots", f"{plot}.html")
           if os.path.exists(path):
               with open(path, "r") as f:
                   body = f.read()
                   inner = body.split("<body>")[1].split("</body>")[0] if "<body>" in body else body
                   tab_html += f"<div class='plot-card'><h2>{plot.replace('_', ' ').title()}</h2>{inner}</div>"
           else:
               tab_html += f"<div class='plot-card'><h2>{plot.replace('_', ' ').title()}</h2><p>Plot not found</p></div>"
       tab_contents += f"<div class='tab-content {active_class}' id='{tab_id}'>{tab_html}</div>"

   html = f"""
   <!DOCTYPE html>
   <html lang='en'>
   <head>
       <meta charset='UTF-8'>
       <title>Uber Trip Forecasting Dashboard</title>
       <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
       <style>
           body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background: #f1f2f6; }}
           header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; font-size: 2em; }}
           nav {{ display: flex; justify-content: center; background: #dcdde1; padding: 10px 0; }}
           nav ul {{ list-style: none; display: flex; padding: 0; margin: 0; }}
           nav li {{ padding: 10px 20px; cursor: pointer; border-radius: 6px; margin: 0 5px; background: #dfe6e9; }}
           nav li.active, nav li:hover {{ background: #0984e3; color: white; }}
           .tab-content {{ display: none; padding: 30px; max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-top: 20px; }}
           .tab-content.active {{ display: block; }}
           .plot-card {{ margin-bottom: 40px; }}
           h2 {{ color: #0984e3; margin-bottom: 10px; text-align: center; }}
           footer {{ text-align: center; padding: 20px; background: #2c3e50; color: white; margin-top: 40px; }}
           .actions {{ text-align: center; margin-top: 20px; }}
           .btn {{ background: #00cec9; color: white; padding: 10px 20px; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; }}
           .btn:hover {{ background: #0984e3; }}
       </style>
   </head>
   <body>
       <header>📊 Uber Trip Forecasting Dashboard</header>
       <nav>
           <ul>
               {tab_headers}
           </ul>
       </nav>
       {tab_contents}
       <div class="actions">
           <form action="/export/pdf">
               <button class="btn" type="submit">📥 Export All Plots to PDF</button>
           </form>
       </div>
       <footer>Built by Jayanth Chennoju | Tools: FastAPI, XGBoost, Plotly, SHAP, Render</footer>
       <script>
           document.querySelectorAll('nav li').forEach((tab, index) => {{
               tab.addEventListener('click', () => {{
                   document.querySelectorAll('nav li').forEach(t => t.classList.remove('active'));
                   document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
                   tab.classList.add('active');
                   document.getElementById(`tab${{index}}`).classList.add('active');
               }});
           }});
       </script>
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
       "status": "Model is ready!" if model else "Model failed to load."
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

@app.get("/export/pdf")
def export_pdf():
   pdf = FPDF()
   pdf.add_page()
   pdf.set_font("Arial", size=16)
   pdf.cell(200, 10, txt="Uber Trip Forecasting - Plots Summary", ln=True, align="C")

   plot_images = [
       "xgb_vs_actual.png", "rf_vs_actual.png", "ensemble_vs_actual.png",
       "trips_per_hour.png", "trips_per_day.png",
       "train_test_split.png", "decomposition.png", "shap_summary.png"
   ]
   for plot in plot_images:
       path = os.path.join("plots", plot)
       if os.path.exists(path):
           pdf.add_page()
           pdf.image(path, x=10, y=30, w=180)
       else:
           pdf.add_page()
           pdf.set_font("Arial", size=12)
           pdf.cell(200, 20, txt=f"{plot} not found.", ln=True, align="C")

   out_path = "plots/uber_dashboard_report.pdf"
   pdf.output(out_path)
   return FileResponse(out_path, media_type="application/pdf", filename="uber_dashboard_report.pdf")

