# generate_predictions.py

import pandas as pd
import joblib
import os

# === Load model
model = joblib.load("models/xgb_model.pkl")

# === Load processed data
df = pd.read_csv("data/uber_processed.csv")
df['date'] = pd.to_datetime(df['date'])

# === Features used in the model
features = ['hour', 'day', 'day_of_week', 'month', 'active_vehicles']

# === Predict
X = df[features]
y_actual = df['trips']
y_pred = model.predict(X)

# === Save to CSV
out_df = pd.DataFrame({
    "date": df['date'],
    "actual": y_actual,
    "predicted_xgb": y_pred
})

os.makedirs("data", exist_ok=True)
out_df.to_csv("data/xgb_predictions.csv", index=False)
print("✅ Saved: data/xgb_predictions.csv")
