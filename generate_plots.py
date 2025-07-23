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

# === XGBoost vs Actual ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['actual'], name='Actual Trips', line=dict(color='black')))
fig.add_trace(go.Scatter(x=df['date'], y=df['predicted_xgb'], name='XGBoost Prediction', line=dict(color='orange')))
fig.update_layout(title="XGBoost Prediction vs Actual", xaxis_title="Date", yaxis_title="Trips", hovermode="x unified")
fig.write_html("plots/xgb_vs_actual.html")
print("✅ Saved: plots/xgb_vs_actual.html")

# === Random Forest vs Actual ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['actual'], name='Actual Trips', line=dict(color='black')))
fig.add_trace(go.Scatter(x=df['date'], y=df['predicted_rf'], name='Random Forest Prediction', line=dict(color='blue')))
fig.update_layout(title="Random Forest Prediction vs Actual", xaxis_title="Date", yaxis_title="Trips", hovermode="x unified")
fig.write_html("plots/rf_vs_actual.html")
print("✅ Saved: plots/rf_vs_actual.html")

# === Ensemble vs Actual ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['actual'], name='Actual Trips', line=dict(color='black')))
fig.add_trace(go.Scatter(x=df['date'], y=df['predicted_ensemble'], name='Ensemble Prediction', line=dict(color='green')))
fig.update_layout(title="Ensemble Prediction vs Actual", xaxis_title="Date", yaxis_title="Trips", hovermode="x unified")
fig.write_html("plots/ensemble_vs_actual.html")
print("✅ Saved: plots/ensemble_vs_actual.html")

# === Trips per Hour (EDA) ===
df['hour'] = pd.to_datetime(df['date']).dt.hour
hourly = df.groupby('hour')['actual'].sum().reset_index()
fig = go.Figure(data=go.Bar(x=hourly['hour'], y=hourly['actual'], name="Trips"))
fig.update_layout(title="Trips per Hour", xaxis_title="Hour", yaxis_title="Total Trips")
fig.write_html("plots/trips_per_hour.html")
print("✅ Saved: plots/trips_per_hour.html")

# === Trips per Day of Week (EDA) ===
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
dow = df.groupby('day_of_week')['actual'].sum().reset_index()
dow['day_name'] = dow['day_of_week'].map({0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'})
fig = go.Figure(data=go.Bar(x=dow['day_name'], y=dow['actual'], name="Trips"))
fig.update_layout(title="Trips per Day of Week", xaxis_title="Day of Week", yaxis_title="Total Trips")
fig.write_html("plots/trips_per_day.html")
print("✅ Saved: plots/trips_per_day.html")
