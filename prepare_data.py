import pandas as pd

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
