# scripts/ensemble_predict.py

import joblib
import numpy as np
import pandas as pd

def ensemble_predict(X_test, model_dir="models"):
    # Load all models
    xgb = joblib.load(f"{model_dir}/xgb_model.pkl")
    rf = joblib.load(f"{model_dir}/rf_model.pkl")
    gbr = joblib.load(f"{model_dir}/gbr_model.pkl")

    # Predict individually
    xgb_preds = xgb.predict(X_test)
    rf_preds = rf.predict(X_test)
    gbr_preds = gbr.predict(X_test)

    # Model weights (from reference MAPE scores)
    weights = np.array([0.368, 0.322, 0.310])
    ensemble_preds = (weights[0] * xgb_preds + weights[1] * rf_preds + weights[2] * gbr_preds)

    return {
        "xgb": xgb_preds,
        "rf": rf_preds,
        "gbr": gbr_preds,
        "ensemble": ensemble_preds
    }
