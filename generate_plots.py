import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from statsmodels.tsa.seasonal import seasonal_decompose

# Ensure output directory
os.makedirs("plots", exist_ok=True)

# === Load datasets ===
df = pd.read_csv("data/xgb_predictions.csv")
df['date'] = pd.to_datetime(df['date'])

# === Helper to save both HTML and PNG ===
def save_plot(fig, name):
    html_path = f"plots/{name}.html"
    png_path = f"plots/{name}.png"
    pio.write_html(fig, file=html_path, auto_open=False)
    fig.write_image(png_path, width=1000, height=600)
    print(f"✅ Saved: {html_path}, {png_path}")

# === XGBoost vs Actual ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['actual'], name="Actual Trips", line=dict(color='black')))
fig.add_trace(go.Scatter(x=df['date'], y=df['predicted_xgb'], name="XGBoost Prediction", line=dict(color='orange')))
fig.update_layout(title="XGBoost Prediction vs Actual", xaxis_title="Date", yaxis_title="Trips", hovermode="x unified")
save_plot(fig, "xgb_vs_actual")

# === Random Forest vs Actual ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['actual'], name="Actual Trips", line=dict(color='black')))
fig.add_trace(go.Scatter(x=df['date'], y=df['predicted_rf'], name="Random Forest Prediction", line=dict(color='blue')))
fig.update_layout(title="Random Forest Prediction vs Actual", xaxis_title="Date", yaxis_title="Trips", hovermode="x unified")
save_plot(fig, "rf_vs_actual")

# === Ensemble vs Actual ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['actual'], name="Actual Trips", line=dict(color='black')))
fig.add_trace(go.Scatter(x=df['date'], y=df['predicted_ensemble'], name="Ensemble Prediction", line=dict(color='green')))
fig.update_layout(title="Ensemble Prediction vs Actual", xaxis_title="Date", yaxis_title="Trips", hovermode="x unified")
save_plot(fig, "ensemble_vs_actual")

# === Trips per Hour ===
df['hour'] = df['date'].dt.hour
hourly = df.groupby('hour')['actual'].sum().reset_index()
fig = px.bar(hourly, x='hour', y='actual', title="Trips per Hour", labels={'hour': 'Hour', 'actual': 'Total Trips'})
save_plot(fig, "trips_per_hour")

# === Trips per Day of Week ===
df['day_of_week'] = df['date'].dt.dayofweek
dow = df.groupby('day_of_week')['actual'].sum().reset_index()
dow['day'] = dow['day_of_week'].map({0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'})
fig = px.bar(dow, x='day', y='actual', title="Trips per Day of Week", labels={'day': 'Day', 'actual': 'Total Trips'})
save_plot(fig, "trips_per_day")

# === Train/Test Split ===
df_full = pd.read_csv("data/uber_processed.csv")
df_full['date'] = pd.to_datetime(df_full['date'])
df_full.set_index('date', inplace=True)
ts = df_full['trips'].resample('h').sum()
split_date = pd.to_datetime("2015-06-01")

fig = go.Figure()
fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Trips", line=dict(color="skyblue")))
fig.add_vline(x=split_date, line=dict(dash="dash", color="red"), annotation_text="Train/Test Split", annotation_position="top left")
fig.update_layout(title="Train/Test Split on Uber Trip Data", xaxis_title="Date", yaxis_title="Trips per Hour")
save_plot(fig, "train_test_split")

# === Decomposition Plot ===
result = seasonal_decompose(ts, model='additive', period=24)
decomp_df = pd.DataFrame({
    "observed": result.observed,
    "trend": result.trend,
    "seasonal": result.seasonal,
    "resid": result.resid
}).dropna().reset_index()

fig = go.Figure()
for col in ['observed', 'trend', 'seasonal', 'resid']:
    fig.add_trace(go.Scatter(x=decomp_df['date'], y=decomp_df[col], name=col.title()))
fig.update_layout(title="Seasonal Decomposition of Uber Trips", xaxis_title="Date", yaxis_title="Value")
save_plot(fig, "decomposition")
