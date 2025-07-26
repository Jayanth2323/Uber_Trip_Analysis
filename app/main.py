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

# @app.get("/export/pdf")
# def export_pdf():
#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=15)

#     # 📄 Cover Page
#     pdf.add_page()
#     pdf.set_font("Helvetica", "B", 16)
#     pdf.cell(200, 10, txt="Uber Trip Forecasting - Plots Summary", ln=True, align="C")
#     pdf.set_font("Helvetica", size=12)
#     pdf.ln(10)
#     pdf.cell(200, 10, txt=f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}", ln=True, align="C")
#     pdf.ln(20)

#     # 📊 Plots to include (based on your repo structure)
#     plots_with_desc = [
#         ("XGBoost vs Actual", "xgb_vs_actual.png", "Predicted vs actual trip counts using XGBoost model."),
#         ("Random Forest vs Actual", "rf_vs_actual.png", "Forecast comparison using Random Forest model."),
#         ("Ensemble vs Actual", "ensemble_vs_actual.png", "Combined predictions from multiple models."),
#         ("Trips per Hour", "trips_per_hour.png", "Hourly distribution of Uber trips."),
#         ("Trips per Day", "trips_per_day.png", "Daily volume of Uber rides."),
#         ("Train-Test Split", "train_test_split.png", "Dataset partitioning for model training and evaluation."),
#         ("Time Series Decomposition", "decomposition.png", "Trend, seasonality, and residuals in trip data."),
#         ("SHAP Summary", "shap_summary.png", "Feature impact visualization using SHAP values."),
#     ]

#     missing = []

#     for title, filename, description in plots_with_desc:
#         path = os.path.join("plots", filename)
#         if os.path.exists(path):
#             pdf.add_page()
#             pdf.set_font("Helvetica", "B", 14)
#             pdf.cell(200, 10, txt=title, ln=True, align="C")
#             pdf.set_font("Helvetica", size=11)
#             pdf.multi_cell(0, 8, description, align="C")
#             pdf.ln(5)
#             try:
#                 pdf.image(path, x=10, y=40, w=180)
#             except RuntimeError as e:
#                 pdf.set_font("Helvetica", size=11)
#                 pdf.ln(10)
#                 pdf.cell(200, 10, txt=f"⚠️ Error loading {filename}", ln=True, align="C")
#                 missing.append(filename)
#         else:
#             pdf.add_page()
#             pdf.set_font("Helvetica", "B", 14)
#             pdf.cell(200, 10, txt=title, ln=True, align="C")
#             pdf.set_font("Helvetica", size=11)
#             pdf.cell(200, 10, txt=f"⚠️ {filename} not found. Please generate it.", ln=True, align="C")
#             missing.append(filename)

#     output_path = "plots/uber_dashboard_report.pdf"
#     try:
#         pdf.output(output_path)
#     except Exception as e:
#         from fastapi import HTTPException
#         raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

#     if not os.path.exists(output_path):
#         raise HTTPException(status_code=500, detail="PDF file was not created.")

#     return FileResponse(
#         path=out_path,
#         media_type="application/pdf",
#         filename="uber_dashboard_report.pdf"
#     )

# @app.get("/export/pdf")
# def export_pdf():
#     from datetime import datetime

#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=15)

#     # 📄 Cover Page
#     pdf.add_page()
#     pdf.set_font("Helvetica", "B", 16)
#     pdf.cell(200, 10, txt="Uber Trip Forecasting - Plots Summary", ln=True, align="C")
#     pdf.set_font("Helvetica", size=12)
#     pdf.ln(10)
#     pdf.cell(200, 10, txt=f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}", ln=True, align="C")
#     pdf.ln(20)

#     # Plot titles and their descriptions
#     plots_with_desc = [
#         ("XGBoost vs Actual", "xgb_vs_actual.png", "Predicted vs actual trip counts using XGBoost model."),
#         ("Random Forest vs Actual", "rf_vs_actual.png", "Forecast comparison using Random Forest model."),
#         ("Ensemble vs Actual", "ensemble_vs_actual.png", "Combined predictions from multiple models."),
#         ("Trips per Hour", "trips_per_hour.png", "Hourly distribution of Uber trips."),
#         ("Trips per Day", "trips_per_day.png", "Daily volume of Uber rides."),
#         ("Train-Test Split", "train_test_split.png", "Dataset partitioning for model training and evaluation."),
#         ("Time Series Decomposition", "decomposition.png", "Trend, seasonality, and residuals in trip data."),
#         ("SHAP Summary", "shap_summary.png", "Feature impact visualization using SHAP values."),
#     ]

#     missing = []

#     for title, filename, description in plots_with_desc:
#         path = os.path.join("plots", filename)
#         if os.path.exists(path):
#             pdf.add_page()
#             pdf.set_font("Helvetica", "B", 14)
#             pdf.cell(200, 10, txt=title, ln=True, align="C")
#             pdf.set_font("Helvetica", size=11)
#             pdf.multi_cell(0, 8, description, align="C")
#             pdf.ln(5)
#             try:
#                 pdf.image(path, x=10, y=40, w=180)
#             except RuntimeError as e:
#                 pdf.set_font("Helvetica", size=11)
#                 pdf.ln(10)
#                 pdf.cell(200, 10, txt=f"⚠️ Error loading {filename}", ln=True, align="C")
#                 missing.append(filename)
#         else:
#             pdf.add_page()
#             pdf.set_font("Helvetica", "B", 14)
#             pdf.cell(200, 10, txt=title, ln=True, align="C")
#             pdf.set_font("Helvetica", size=11)
#             pdf.cell(200, 10, txt=f"⚠️ {filename} not found. Please generate it.", ln=True, align="C")
#             missing.append(filename)

#     out_path = "plots/uber_dashboard_report.pdf"
#     try:
#         pdf.output(out_path)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

#     if not os.path.exists(out_path):
#         raise HTTPException(status_code=500, detail="PDF file was not created.")

#     return FileResponse(
#         path=out_path,
#         media_type="application/pdf",
#         filename="uber_dashboard_report.pdf"
#     )

# @app.get("/export/pdf")
# def export_pdf():
#     from datetime import datetime

#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=15)

#     # Cover Page
#     pdf.add_page()
#     pdf.set_font("Arial", "B", 18)
#     pdf.cell(200, 10, txt="Uber Trip Forecasting - Plots Summary", ln=True, align="C")
#     pdf.set_font("Arial", size=12)
#     pdf.ln(10)
#     pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
#     pdf.ln(20)

#     # Plots + Descriptions
#     plots = [
#         ("XGBoost vs Actual", "xgb_vs_actual.png", "Predicted vs actual trip counts using XGBoost model."),
#         ("Random Forest vs Actual", "rf_vs_actual.png", "Forecast comparison using Random Forest model."),
#         ("Ensemble vs Actual", "ensemble_vs_actual.png", "Combined predictions from multiple models."),
#         ("Trips per Hour", "trips_per_hour.png", "Hourly distribution of Uber trips across different times of the day."),
#         ("Trips per Day", "trips_per_day.png", "Daily volume of Uber rides showing weekday trends."),
#         ("Train-Test Split", "train_test_split.png", "Dataset split for training and validating model accuracy."),
#         ("Time Series Decomposition", "decomposition.png", "Breakdown of trend, seasonality, and residuals."),
#         ("SHAP Summary", "shap_summary.png", "Feature impact visualization using SHAP values."),
#     ]

#     for title, file, desc in plots:
#         path = os.path.join("plots", file)
#         pdf.add_page()
#         pdf.set_font("Arial", "B", 14)
#         pdf.cell(200, 10, txt=title, ln=True, align="C")
#         if os.path.exists(path):
#             try:
#                 pdf.image(path, x=15, y=30, w=180)
#                 pdf.ln(100)  # adjust spacing after image
#             except Exception as e:
#                 pdf.ln(20)
#                 pdf.set_font("Arial", size=12)
#                 pdf.cell(200, 10, txt=f"⚠️ Failed to load {file}", ln=True, align="C")
#         else:
#             pdf.ln(20)
#             pdf.set_font("Arial", size=12)
#             pdf.cell(200, 10, txt=f"⚠️ {file} not found", ln=True, align="C")

#         pdf.ln(85)  # space before description
#         pdf.set_font("Arial", size=11)
#         pdf.multi_cell(0, 10, desc, align="C")

#     out_path = "plots/uber_dashboard_report.pdf"
#     pdf.output(out_path)

#     return FileResponse(
#         out_path,
#         media_type="application/pdf",
#         filename="uber_dashboard_report.pdf"
#     )

@app.get("/export/pdf")
def export_pdf():
    class PDF(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover page
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt=" Uber Trip Forecasting - Plots Summary", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10, "This report provides a visual summary of Uber trip forecasting results, exploration insights, time series components, and explainability of model predictions.")
    pdf.ln(10)

    # Plot metadata
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
        "xgb_vs_actual.png": "Comparison of predicted vs actual trip counts using the XGBoost model. This highlights the model's predictive accuracy.",
        "rf_vs_actual.png": "Random Forest model predictions plotted against actual trip values to assess its performance visually.",
        "ensemble_vs_actual.png": "Ensemble model combining predictions from multiple models to improve accuracy.",
        "trips_per_hour.png": "Distribution of trip counts per hour indicating peak and off-peak ride times.",
        "trips_per_day.png": "Trip frequency across days helps identify weekly patterns or anomalies.",
        "train_test_split.png": "Time-based train-test data split for model training and evaluation.",
        "decomposition.png": "Seasonal decomposition of time series into trend, seasonal, and residual components.",
        "shap_summary.png": "SHAP summary plot shows the importance of each feature in model decision making.",
    }

    missing_plots = []

    for title, filename in plot_images:
        path = os.path.join("plots", filename)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt=title, ln=True, align="C")
        pdf.ln(5)

        if os.path.exists(path):
            try:
                pdf.image(path, x=15, y=30, w=180)
                pdf.ln(110)
                desc = plot_descriptions.get(filename, "Description not available.")
                pdf.set_font("Arial", size=11)
                pdf.multi_cell(0, 10, f"📌 {desc}")
            except RuntimeError:
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"⚠️ Error loading image: {filename}", ln=True, align="C")
        else:
            missing_plots.append(filename)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"⚠️ {filename} not found. Please generate it.", ln=True, align="C")

    out_path = "plots/uber_dashboard_report.pdf"
    pdf.output(out_path)

    if missing_plots:
        print("⚠️ Missing plots:")
        for p in missing_plots:
            print(f"   - {p}")

    return FileResponse(out_path, media_type="application/pdf", filename="uber_dashboard_report.pdf")
