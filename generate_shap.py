# generate_shap.py

import shap
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.io as pio

# === Config ===
MODEL_PATH = "models/xgb_model.pkl"
DATA_PATH = "data/uber_processed.csv"
OUTPUT_HTML = "plots/shap_summary.html"
FEATURES = ['hour', 'day', 'day_of_week', 'month', 'active_vehicles']

# === Ensure output directory exists
os.makedirs("plots", exist_ok=True)

# === Load model and data
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
X = df[FEATURES].tail(100).astype(float)

# === Generate SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X)

# === Convert SHAP values to DataFrame
shap_df = pd.DataFrame(shap_values.values, columns=FEATURES)
mean_abs = shap_df.abs().mean().sort_values(ascending=True)

# === Create interactive Plotly bar chart
fig = px.bar(
    x=mean_abs.values,
    y=mean_abs.index,
    orientation='h',
    title="SHAP Feature Importance",
    labels={'x': "Mean |SHAP value|", 'y': "Feature"},
    color=mean_abs.values,
    color_continuous_scale="Viridis"
)
fig.update_layout(template="plotly_white", height=400)

# === Save to HTML
pio.write_html(fig, file=OUTPUT_HTML, auto_open=False)
print(f"✅ Saved: {OUTPUT_HTML}")
