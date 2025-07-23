# scripts/prepare_foil_data.py

import pandas as pd

def prepare_foil_csv(input_path="data/Uber-Jan-Feb-FOIL.csv", output_path="data/uber_processed.csv"):
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])

    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    df.to_csv(output_path, index=False)
    print(f"✅ Processed FOIL CSV saved to {output_path}")
