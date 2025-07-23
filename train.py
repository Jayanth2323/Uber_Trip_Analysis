# train.py

from app.model import train_and_save_model
from scripts.prepare_data import load_and_resample
from scripts.feature_engineering import create_lag_features

if __name__ == "__main__":
    train_and_save_model("data/Uber-Jan-Feb-FOIL.csv")

# Load and transform
hourly_df = load_and_resample()
X, y = create_lag_features(hourly_df['Count'].values, window_size=24)

# Temporal Split (train until Sept 15, 2014)
cutoff = hourly_df.index.get_loc("2014-09-15 00:00:00") - 24
X_train, y_train = X.iloc[:cutoff], y.iloc[:cutoff]
X_test, y_test = X.iloc[cutoff:], y.iloc[cutoff:]

