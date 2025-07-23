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
