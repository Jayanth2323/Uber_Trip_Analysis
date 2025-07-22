import shap
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# === CONFIG ===
MODEL_PATH = "./models/xbg_model.pkl"
DATA_PATH = "data/Uber_Jan_Feb_FOIL.csv"   # Replace with your actual final CSV
OUTPUT_PATH = "plots/shap_summary.png"
FEATURES = ['hour', 'day', 'day_of_week', 'month', 'active_vehicles']

# === Ensure plots dir exists ===
os.makedirs("plots", exist_ok=True)

# === Load model ===
model = joblib.load(MODEL_PATH)

# === Load sample data ===
df = pd.read_csv(DATA_PATH)
X = df[FEATURES].tail(100)  # Use last 100 rows as SHAP sample

# === Create SHAP explainer ===
explainer = shap.Explainer(model)
shap_values = explainer(X)

# === Generate & save summary plot ===
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.close()

print(f"✅ SHAP summary saved to {OUTPUT_PATH}")
