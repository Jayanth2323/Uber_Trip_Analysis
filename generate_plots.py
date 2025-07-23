# generate_plots.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# === Load Data ===
df = pd.read_csv("data/uber_processed.csv")  # 🔁 Replace with actual file path if needed

# === Seasonal Decomposition of Hourly Trips ===

# Ensure date is datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Resample to hourly trip count
ts = df['trips'].resample('h').sum()

# Perform decomposition (additive)
result = seasonal_decompose(ts, model='additive', period=24)

# Plot & save
fig = result.plot()
fig.set_size_inches(12, 8)
plt.suptitle("Seasonal Decomposition of Uber Trips", fontsize=16)
plt.tight_layout()
plt.savefig("plots/decomposition.png")
plt.close()

print("✅ Saved: plots/decomposition.png")

# === Train/Test Split Visualization ===

# Reload data to avoid 'date' already set as index
df = pd.read_csv("data/uber_processed.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Aggregate hourly trip counts
ts = df['trips'].resample('h').sum()

# Define split date
split_date = '2015-06-01'  # 📌 adjust as per your PDF if needed

# Plot full time series with split line
plt.figure(figsize=(12, 6))
plt.plot(ts, label="Trips", color="skyblue")
plt.axvline(pd.Timestamp(split_date), color='red', linestyle='--', linewidth=2, label='Train/Test Split')
plt.title("Train/Test Split on Uber Trip Data")
plt.xlabel("Date")
plt.ylabel("Trips per Hour")
plt.legend()
plt.tight_layout()
plt.savefig("plots/train_test_split.png")
plt.close()

print("✅ Saved: plots/train_test_split.png")

# === XGBoost Predictions vs Actual ===
xgb_df = pd.read_csv("data/xgb_predictions.csv")
xgb_df['date'] = pd.to_datetime(xgb_df['date'])

plt.figure(figsize=(12, 6))
plt.plot(xgb_df['date'], xgb_df['actual'], label='Actual Trips', color='black')
plt.plot(xgb_df['date'], xgb_df['predicted_xgb'], label='XGBoost Prediction', color='orange')
plt.title("XGBoost Prediction vs Actual")
plt.xlabel("Date")
plt.ylabel("Trips")
plt.legend()
plt.tight_layout()
plt.savefig("plots/xgb_vs_actual.png")
plt.close()

print("✅ Saved: plots/xgb_vs_actual.png")

# === Random Forest Predictions vs Actual ===
rf_df = pd.read_csv("data/xgb_predictions.csv")  # same file, now includes RF too
rf_df['date'] = pd.to_datetime(rf_df['date'])

plt.figure(figsize=(12, 6))
plt.plot(rf_df['date'], rf_df['actual'], label='Actual Trips', color='black')
plt.plot(rf_df['date'], rf_df['predicted_rf'], label='Random Forest Prediction', color='blue')
plt.title("Random Forest Prediction vs Actual")
plt.xlabel("Date")
plt.ylabel("Trips")
plt.legend()
plt.tight_layout()
plt.savefig("plots/rf_vs_actual.png")
plt.close()

print("✅ Saved: plots/rf_vs_actual.png")

# === Ensemble Predictions vs Actual ===
ens_df = pd.read_csv("data/xgb_predictions.csv")
ens_df['date'] = pd.to_datetime(ens_df['date'])

plt.figure(figsize=(12, 6))
plt.plot(ens_df['date'], ens_df['actual'], label='Actual Trips', color='black')
plt.plot(ens_df['date'], ens_df['predicted_ensemble'], label='Ensemble Prediction', color='green')
plt.title("Ensemble Prediction vs Actual")
plt.xlabel("Date")
plt.ylabel("Trips")
plt.legend()
plt.tight_layout()
plt.savefig("plots/ensemble_vs_actual.png")
plt.close()

print("✅ Saved: plots/ensemble_vs_actual.png")
