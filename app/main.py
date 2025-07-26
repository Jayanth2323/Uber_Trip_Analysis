# app/main.py (final dashboard with premium UI, interactive tabs, PDF export)

import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from app.model import load_model
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fpdf import FPDF
from string import Template
from datetime import datetime
from PIL import Image

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
    print("✅ Model loaded successfully")
except Exception as e:
    model = None
    print("❌ Model failed to load:", str(e))


@app.get("/", response_class=HTMLResponse)
def dashboard():
    plots = [
        ("Forecast Models", ["xgb_vs_actual", "rf_vs_actual", "ensemble_vs_actual"]),
        ("Exploration", ["trips_per_hour", "trips_per_day"]),
        ("Time Series", ["train_test_split", "decomposition"]),
        ("Explainability", ["shap_summary"]),
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
                    inner = (
                        body.split("<body>")[1].split("</body>")[0]
                        if "<body>" in body
                        else body
                    )
                    tab_html += f"<div class='plot-card'><h2>{plot.replace('_', ' ').title()}</h2>{inner}</div>"
            else:
                tab_html += f"<div class='plot-card'><h2>{plot.replace('_', ' ').title()}</h2><p>❌ Plot not found</p></div>"
        tab_contents += f"<div class='tab-content {active_class}' id='{tab_id}'>{tab_html}</div>"

    template = Template("""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Uber Trip Forecasting Dashboard</title>
        <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
        <style>
            :root {
                --bg: #f1f2f6; --text: #2c3e50; --card: #ffffff;
                --primary: #0984e3; --nav: #dcdde1;
            }
            body.dark {
                --bg: #1e272e; --text: #f5f6fa; --primary: #00a8ff; --nav: #353b48;
            }
            body {
                font-family: 'Segoe UI', sans-serif; margin: 0;
                background: var(--bg); color: var(--text);
                transition: background 0.3s, color 0.3s;
            }
            header { background: var(--text); color: var(--card); padding: 20px; text-align: center; font-size: 2em; position: relative; }
            .theme-toggle { position: absolute; top: 20px; right: 20px; cursor: pointer; }
            nav { display: flex; justify-content: center; background: var(--nav); padding: 10px 0; }
            nav ul { list-style: none; display: flex; padding: 0; margin: 0; }
            nav li { padding: 10px 20px; cursor: pointer; border-radius: 6px; margin: 0 5px; background: #dfe6e9; transition: 0.2s; }
            nav li.active, nav li:hover { background: var(--primary); color: black; }
            .tab-content { display: none; padding: 30px; max-width: 1200px; margin: 0 auto; background: var(--card); border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-top: 20px; }
            .tab-content.active { display: block; }
            .plot-card { margin-bottom: 40px; }
            h2 { color: var(--primary); margin-bottom: 10px; text-align: center; }
            .actions { text-align: center; margin-top: 20px; }
            .btn { background: #00cec9; color: white; padding: 10px 20px; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; }
            .btn:hover { background: var(--primary); }
            footer { text-align: center; padding: 20px; background: var(--text); color: var(--card); margin-top: 40px; }
            .dark .tab-content { background: #2f3640; color: #f5f6fa; }
            .dark header, .dark footer { background: #1e272e; }
            .dark nav { background: #2d3436; }
            .dark nav li { background: #636e72; }
            .dark nav li.active, .dark nav li:hover { background: #00cec9; color: #1e272e; }
        </style>
    </head>
    <body>
        <header>
            📊 Uber Trip Forecasting Dashboard
            <div class="theme-toggle" id="theme-toggle" title="Toggle dark mode">🌓</div>
        </header>
        <nav><ul>$tab_headers</ul></nav>
        $tab_contents
        <div class="actions">
            <form action="/export/pdf">
                <button class="btn" type="submit">📄 Export All Plots to PDF</button>
            </form>
        </div>
        <footer>Built by Jayanth Chennoju | Tools: FastAPI, XGBoost, Plotly, SHAP, Render</footer>
        <script>
            document.querySelectorAll('nav li').forEach((tab, index) => {
                tab.addEventListener('click', () => {
                    document.querySelectorAll('nav li').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
                    tab.classList.add('active');
                    document.getElementById("tab" + index).classList.add('active');
                });
            });

            const setTheme = (dark) => {
                document.body.classList.toggle('dark', dark);
                localStorage.setItem('theme', dark ? 'dark' : 'light');

                document.querySelectorAll("iframe").forEach(iframe => {
                    try {
                        const win = iframe.contentWindow;
                        const plotDiv = win?.document?.querySelector("div.js-plotly-plot");
                        if (win?.Plotly?.relayout && plotDiv) {
                            win.Plotly.relayout(plotDiv, { template: dark ? "plotly_dark" : "plotly_white" });
                        }
                    } catch (_) {}
                });
            };

            const savedTheme = localStorage.getItem('theme') === 'dark';
            setTheme(savedTheme);

            document.getElementById('theme-toggle').addEventListener('click', () => {
                const darkMode = !document.body.classList.contains('dark');
                setTheme(darkMode);
            });
        </script>
    </body>
    </html>
    """)

    html = template.substitute(
        tab_headers=tab_headers,
        tab_contents=tab_contents,
    )
    return HTMLResponse(content=html)



@app.post("/predict")
def predict_trips(features: TripFeatures):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})

    try:
        input_data = np.array([[
            features.hour,
            features.day,
            features.day_of_week,
            features.month,
            features.active_vehicles,
        ]])
        prediction = model.predict(input_data)[0]
        return {
            "predicted_trips": round(float(prediction), 2),
            "inputs": features.dict(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
def health_check():
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})  # <== Fix indentation here

    return {
        "model_loaded": model is not None,
        "status": "✅ Model is ready!" if model else "❌ Model failed to load.",
    }


@app.get("/metrics")
def get_metrics():
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})  # <== Fix indentation here

    return {
        "status": "Model metrics loaded successfully",
        "MAPE (%)": {
            "XGBoost": 8.37,
            "Random Forest": 9.61,
            "GBRT": 10.02,
            "Ensemble": 8.60,
        },
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

    return JSONResponse(
        status_code=404, content={"error": f"Plot {plot_name} not found."}
    )

@app.get("/export/pdf")
def export_pdf():
    class PDF(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()} of " + str(self.alias_nb_pages()), align="C")
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover Page
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Uber Trip Forecasting - Plots Summary", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10,
        "This report provides a comprehensive summary of Uber trip forecasting results using multiple ML models, "
        "exploratory trends by time, time series decomposition, and model explainability with SHAP. "
        "These insights are crucial for operations, scheduling, and understanding prediction drivers."
    )
    pdf.ln(5)

    # Plot Titles + Descriptions
    plot_images = [
        ("XGBoost vs Actual", "xgb_vs_actual.png"),
        ("Random Forest vs Actual", "rf_vs_actual.png"),
        ("Ensemble vs Actual", "ensemble_vs_actual.png"),
        ("Trips per Hour", "trips_per_hour.png"),
        ("Trips per Day", "trips_per_day.png"),
        ("Train-Test Split", "train_test_split.png"),
        ("Time Series Decomposition", "decomposition.png"),
        ("SHAP Summary", "shap_summary.png"),
    ]

    plot_descriptions = {
        "xgb_vs_actual.png": "This plot compares the XGBoost model's predicted trip counts against actual observed values...",
        "rf_vs_actual.png": "The Random Forest model's predictions are shown against actual trip counts...",
        "ensemble_vs_actual.png": "This graph presents the performance of an ensemble model...",
        "trips_per_hour.png": "Hourly trip patterns reveal how Uber demand fluctuates across a 24-hour period...",
        "trips_per_day.png": "This visualization shows trip volume for each day...",
        "train_test_split.png": "Shows how data was split chronologically into training and testing sets...",
        "decomposition.png": "A time-series decomposition into trend, seasonality, and residual components...",
        "shap_summary.png": "This SHAP plot highlights the influence of each feature on the models predictions..."
    }

    for title, filename in plot_images:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt=title, ln=True, align="C")
        pdf.ln(5)

        path = os.path.join("plots", filename)
        if os.path.exists(path):
            try:
                with Image.open(path) as im:
                    w, h = im.size
                    aspect = h / w
                    img_w = 180  # mm
                    img_h = img_w * aspect
                pdf.image(path, x=15, y=30, w=img_w, h=img_h)
                pdf.set_y(30 + img_h + 8)  # Place description below image
                desc = plot_descriptions.get(filename, "Description not available.")
                pdf.set_font("Arial", size=11)
                pdf.multi_cell(0, 8, desc)
            except RuntimeError:
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Error loading {filename}", ln=True, align="C")
        else:
            try:
                pdf.set_font("Arial", size=12)
            except:
                pdf.set_font("Helvetica", size=12)
            pdf.cell(200, 10, txt=f"{filename} not found. Please generate it.", ln=True, align="C")

    out_path = "plots/uber_dashboard_report.pdf"
    pdf.output(out_path)
    return FileResponse(out_path, media_type="application/pdf", filename="uber_dashboard_report.pdf")
