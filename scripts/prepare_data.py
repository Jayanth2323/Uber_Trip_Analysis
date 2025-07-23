import pandas as pd
import os

# Load original FOIL CSV
df = pd.read_csv("data/Uber-Jan-Feb-FOIL.csv")

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract useful datetime features
df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Save final processed file
df.to_csv("data/uber_processed.csv", index=False)

print("✅ Data prepared and saved to data/uber_processed.csv")

def load_and_resample(data_path="data/raw"):
    files = [f for f in os.listdir(data_path) if f.endswith('.csv') and 'uber-raw-data' in f]
    dfs = [pd.read_csv(os.path.join(data_path, f)) for f in files]
    
    df = pd.concat(dfs, ignore_index=True)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df = df.rename(columns={'Date/Time': 'Date'}).sort_values('Date')
    df.set_index('Date', inplace=True)
    
    # Hourly aggregation
    hourly_df = df['Base'].resample('H').count().reset_index()
    hourly_df.columns = ['Date', 'Count']
    hourly_df.set_index('Date', inplace=True)
    
    return hourly_df
