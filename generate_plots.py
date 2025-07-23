import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# === Load Data ===
df = pd.read_csv("data/uber_processed.csv")  # 🔁 Replace with actual file path if needed

# === Seasonal Decomposition of Hourly Trips ===

# Ensure date is datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Resample to hourly trip count
ts = df['trips'].resample('h').sum()

# Perform decomposition (additive)
result = seasonal_decompose(ts, model='additive', period=24)

# Plot & save
fig = result.plot()
fig.set_size_inches(12, 8)
plt.suptitle("Seasonal Decomposition of Uber Trips", fontsize=16)
plt.tight_layout()
plt.savefig("plots/decomposition.png")
plt.close()

print("✅ Saved: plots/decomposition.png")

# === Train/Test Split Visualization ===

# Reload data to avoid 'date' already set as index
df = pd.read_csv("data/uber_processed.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Aggregate hourly trip counts
ts = df['trips'].resample('h').sum()

# Define split date
split_date = '2015-06-01'  # 📌 adjust as per your PDF if needed

# Plot full time series with split line
plt.figure(figsize=(12, 6))
plt.plot(ts, label="Trips", color="skyblue")
plt.axvline(pd.Timestamp(split_date), color='red', linestyle='--', linewidth=2, label='Train/Test Split')
plt.title("Train/Test Split on Uber Trip Data")
plt.xlabel("Date")
plt.ylabel("Trips per Hour")
plt.legend()
plt.tight_layout()
plt.savefig("plots/train_test_split.png")
plt.close()

print("✅ Saved: plots/train_test_split.png")
