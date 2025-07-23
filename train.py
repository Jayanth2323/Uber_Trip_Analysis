import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# === Load preprocessed data ===
df = pd.read_csv("data/uber_processed.csv")

# ✅ Rename columns to match training-time format
df = df.rename(columns={
    'hour': 'Hour',
    'day': 'Day',
    'day_of_week': 'DayOfWeek',
    'month': 'Month'
})

# === Features and Target ===
features = ['Hour', 'Day', 'DayOfWeek', 'Month', 'active_vehicles']
X = df[features]
y = df['trips']

# === Train/Test Split (optional if needed) ===
# You can split by date or just use all data for now
# For this example, using all available data

# === Model Configs ===
models = {
    "xgb": XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.6,
        colsample_bytree=1.0
    ),
    "rf": RandomForestRegressor(
        n_estimators=100,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=None,
        random_state=42
    ),
    "gbr": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        max_features='sqrt',
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42
    )
}

# === Train & Save Models ===
os.makedirs("models", exist_ok=True)

print("🚀 Training models...")

for name, model in models.items():
    model.fit(X, y)
    joblib.dump(model, f"models/{name}_model.pkl")
    print(f"✅ Saved {name.upper()} model to models/{name}_model.pkl")

print("✅ All models trained and saved successfully.")
