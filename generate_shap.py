# import shap
# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib
# import os

# # === CONFIG ===
# MODEL_PATH = "./models/xgb_model.pkl"
# DATA_PATH = "./data/uber_processed.csv"   
# OUTPUT_PATH = "plots/shap_summary.png"
# FEATURES = ['dispatching_base_number', 'date', 'active_vehicles', 'trips', 'hour', 'day', 'day_of_week', 'month']

# # === Ensure plots dir exists ===
# os.makedirs("plots", exist_ok=True)

# # === Load model ===
# model = joblib.load(MODEL_PATH)

# # === Load sample data ===
# df = pd.read_csv(DATA_PATH)
# X = df[FEATURES].tail(100)  # Use last 100 rows as SHAP sample

# # === Create SHAP explainer ===
# explainer = shap.Explainer(model)
# shap_values = explainer(X)

# # === Generate & save summary plot ===
# plt.figure()
# shap.summary_plot(shap_values, X, show=False)
# plt.savefig(OUTPUT_PATH, bbox_inches="tight")
# plt.close()

# print(f"✅ SHAP summary saved to {OUTPUT_PATH}")

import shap
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# === CONFIG ===
MODEL_PATH = "./models/xgb_model.pkl"
DATA_PATH = "./data/uber_processed.csv"
OUTPUT_PATH = "plots/shap_summary.png"

# ✅ Only include numeric model input features
FEATURES = ['hour', 'day', 'day_of_week', 'month', 'active_vehicles']

# === Ensure plots dir exists ===
os.makedirs("plots", exist_ok=True)

# === Load model ===
model = joblib.load(MODEL_PATH)

# === Load sample data
df = pd.read_csv(DATA_PATH)
X = df[FEATURES].tail(100).copy()
X = X.astype(float)  # Convert to float to avoid dtype issues

# === Create SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X)

# === Generate & save summary plot
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.close()

print(f"✅ SHAP summary saved to {OUTPUT_PATH}")
