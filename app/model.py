# app/model.py

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['date'])
    df['Hour'] = 0  # static for daily aggregate
    df['Day'] = df['datetime'].dt.day
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['Month'] = df['datetime'].dt.month
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    X = df[['Hour', 'Day', 'DayOfWeek', 'Month', 'active_vehicles']]
    y = df['trips']
    return train_test_split(X, y, test_size=0.3, random_state=42)


def train_and_save_model(filepath: str = "data/Uber-Jan-Feb-FOIL.csv") -> None:
    df = load_data(filepath)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = XGBRegressor(objective="reg:squarederror", n_estimators=300, max_depth=6, learning_rate=0.1)
    model.fit(X_train, y_train)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model trained and saved to {MODEL_PATH}")


def load_model() -> XGBRegressor:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model
