# === generate_plots.py ===
import pandas as pd
import plotly.graph_objects as go
import os

os.makedirs("plots", exist_ok=True)

# === Load data ===
df = pd.read_csv("data/xgb_predictions.csv")
df['date'] = pd.to_datetime(df['date'])

# === XGBoost Plot ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['actual'], name='Actual Trips', line=dict(color='black')))
fig.add_trace(go.Scatter(x=df['date'], y=df['predicted_xgb'], name='XGBoost Prediction', line=dict(color='orange')))
fig.update_layout(title="XGBoost Prediction vs Actual", xaxis_title="Date", yaxis_title="Trips", hovermode="x unified")
fig.write_html("plots/xgb_vs_actual.html")
print("✅ Saved: plots/xgb_vs_actual.html")

# === Random Forest Plot ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['actual'], name='Actual Trips', line=dict(color='black')))
fig.add_trace(go.Scatter(x=df['date'], y=df['predicted_rf'], name='Random Forest Prediction', line=dict(color='blue')))
fig.update_layout(title="Random Forest Prediction vs Actual", xaxis_title="Date", yaxis_title="Trips", hovermode="x unified")
fig.write_html("plots/rf_vs_actual.html")
print("✅ Saved: plots/rf_vs_actual.html")

# === Ensemble Plot ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['actual'], name='Actual Trips', line=dict(color='black')))
fig.add_trace(go.Scatter(x=df['date'], y=df['predicted_ensemble'], name='Ensemble Prediction', line=dict(color='green')))
fig.update_layout(title="Ensemble Prediction vs Actual", xaxis_title="Date", yaxis_title="Trips", hovermode="x unified")
fig.write_html("plots/ensemble_vs_actual.html")
print("✅ Saved: plots/ensemble_vs_actual.html")
