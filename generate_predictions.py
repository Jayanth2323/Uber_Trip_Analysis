import pandas as pd
import joblib
import os

# === Load processed data ===
df = pd.read_csv("data/uber_processed.csv")
df['date'] = pd.to_datetime(df['date'])

# === Load models ===
xgb_model = joblib.load("models/xgb_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")

# === Common lowercase input features ===
features = ['hour', 'day', 'day_of_week', 'month', 'active_vehicles']
X_raw = df[features].copy()

# === Rename to match model training formats ===
X_model = X_raw.copy()
X_model.columns = ['Hour', 'Day', 'DayOfWeek', 'Month', 'active_vehicles']  # Match training features

# === Make predictions ===
y_actual = df['trips']
y_pred_xgb = xgb_model.predict(X_model)
y_pred_rf = rf_model.predict(X_model)

# === Combine results into one DataFrame ===
out_df = pd.DataFrame({
    "date": df['date'],
    "actual": y_actual,
    "predicted_xgb": y_pred_xgb,
    "predicted_rf": y_pred_rf
})

# === Save to CSV ===
os.makedirs("data", exist_ok=True)
out_df.to_csv("data/xgb_predictions.csv", index=False)
print("✅ Saved: data/xgb_predictions.csv with XGB + RF predictions")
