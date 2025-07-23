# train.py

import os
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from scripts.prepare_data import load_and_resample
from scripts.feature_engineering import create_lag_features

def train_and_save_models(X_train, y_train, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)

    models = {
        "xgb": XGBRegressor(objective='reg:squarederror', n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.6, colsample_bytree=1.0),
        "rf": RandomForestRegressor(n_estimators=100, max_depth=30, min_samples_split=5, min_samples_leaf=2, max_features=None),
        "gbr": GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, max_features='sqrt', min_samples_split=5, min_samples_leaf=1)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(output_dir, f"{name}_model.pkl"))
        print(f"✅ Saved {name.upper()} model to {output_dir}/{name}_model.pkl")

if __name__ == "__main__":
    print("🚀 Starting training pipeline...")

    hourly_df = load_and_resample()
    X, y = create_lag_features(hourly_df['Count'].values, window_size=24)

    cutoff = hourly_df.index.get_loc("2014-09-15 00:00:00") - 24
    X_train, y_train = X.iloc[:cutoff], y.iloc[:cutoff]

    train_and_save_models(X_train, y_train)
    print("✅ All models trained and saved successfully.")
