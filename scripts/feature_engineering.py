import numpy as np
import pandas as pd


def create_lag_features(series, window_size=24):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    return pd.DataFrame(X), pd.Series(y)
