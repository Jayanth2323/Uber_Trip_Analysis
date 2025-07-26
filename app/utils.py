import os
import numpy as np
from fpdf import FPDF
from datetime import datetime

# === MODEL UTILS ===
def prepare_input(features):
    """Convert TripFeatures to NumPy array for prediction."""
    return np.array([[
        features.hour,
        features.day,
        features.day_of_week,
        features.month,
        features.active_vehicles,
    ]])

# === PLOT UTILS ===
def get_plot_path(plot_name, folder="plots"):
    """
    Returns the correct plot file path (.html or .png).
    """
    html_path = os.path.join(folder, f"{plot_name}.html")
    if os.path.exists(html_path):
        return html_path
    png_path = os.path.join(folder, f"{plot_name}.png")
    return png_path if os.path.exists(png_path) else None


# === PDF UTILS ===
class PDFReport(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")


def generate_pdf_report(plots, output_path="plots/uber_dashboard_report.pdf"):
    """
    Generate a PDF report with plots and descriptions.
    """
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover page
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="📊 Uber Trip Forecasting - Plots Summary", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10, "This report includes model performance, time-series trends, and SHAP explainability visuals.")

    # Descriptions for each plot
    descriptions = {
        "xgb_vs_actual.png": "Comparison of predicted vs actual trip counts using XGBoost model.",
        "rf_vs_actual.png": "Random Forest model predictions vs actual trip values.",
        "ensemble_vs_actual.png": "Ensemble model combining multiple models' predictions.",
        "trips_per_hour.png": "Hourly distribution of Uber trips showing peak periods.",
        "trips_per_day.png": "Trip frequency per day to identify weekly patterns.",
        "train_test_split.png": "Time-based data split for model training and evaluation.",
        "decomposition.png": "Time-series decomposition into trend, seasonality, and residuals.",
        "shap_summary.png": "SHAP plot indicating feature importance for predictions.",
    }

    missing = []
    for filename in plots:
        path = os.path.join("plots", filename)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt=filename.replace("_", " ").replace(".png", "").title(), ln=True, align="C")
        pdf.ln(5)
        if os.path.exists(path):
            try:
                pdf.image(path, x=15, y=30, w=180)
                pdf.ln(110)
                desc = descriptions.get(filename, "Description not available.")
                pdf.set_font("Arial", size=11)
                pdf.multi_cell(0, 10, f"📌 {desc}")
            except RuntimeError:
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"⚠️ Error loading {filename}", ln=True, align="C")
        else:
            missing.append(filename)
            pdf.cell(200, 10, txt=f"⚠️ {filename} not found.", ln=True, align="C")

    pdf.output(output_path)
    return output_path, missing
