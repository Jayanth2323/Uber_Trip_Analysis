# === generate_plots.py ===
import pandas as pd
import plotly.graph_objects as go
import os
import plotly.io as pio

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

# === Convert train_test_split.png → train_test_split.html (interactive version)
df = pd.read_csv("data/uber_processed.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
ts = df['trips'].resample('h').sum()
split_date = pd.Timestamp('2015-06-01')

fig_split = go.Figure()
fig_split.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Trips", line=dict(color="skyblue")))
fig_split.add_vline(x=split_date, line_dash="dash", line_color="red")

# Add annotation separately
fig_split.add_annotation(
    x=split_date,
    y=ts.max(),
    text="Train/Test Split",
    showarrow=True,
    arrowhead=1,
    ax=0,
    ay=-40
)
fig_split.update_layout(title="Train/Test Split on Uber Trip Data", xaxis_title="Date", yaxis_title="Trips per Hour")
pio.write_html(fig_split, file="plots/train_test_split.html", auto_open=False)
print("✅ Saved: plots/train_test_split.html")


# === Convert decomposition to interactive version
from statsmodels.tsa.seasonal import seasonal_decompose

ts = df['trips'].resample('h').sum()
result = seasonal_decompose(ts, model='additive', period=24)
decomp_df = pd.DataFrame({
    "observed": result.observed,
    "trend": result.trend,
    "seasonal": result.seasonal,
    "resid": result.resid
})
decomp_df = decomp_df.reset_index().dropna()

fig_decomp = go.Figure()
for col in ["observed", "trend", "seasonal", "resid"]:
    fig_decomp.add_trace(go.Scatter(x=decomp_df['date'], y=decomp_df[col], name=col.title()))
fig_decomp.update_layout(title="Seasonal Decomposition of Uber Trips", xaxis_title="Date")
pio.write_html(fig_decomp, file="plots/decomposition.html", auto_open=False)
print("✅ Saved: plots/decomposition.html")
