# 🚕 Uber Trip Demand Forecasting – NYC (FOIL Dataset)

**Forecasting hourly Uber trip counts using ML models like XGBoost, Random Forest, GBRT, and an Ensemble.**

---

## 📌 Project Overview

This project analyzes and predicts Uber trip demand using historical data collected via the NYC Taxi and Limousine Commission (TLC) FOIL release.

We focus on:
- Exploratory data analysis (EDA)
- Feature engineering (including time-lag features)
- Model training (XGBoost, Random Forest, GBRT)
- Ensemble prediction
- Deployment as a FastAPI-based prediction service

---

## 🛠️ Tools & Technologies

- **Languages:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, SHAP, Statsmodels
- **API:** FastAPI
- **IDE:** VS Code, Jupyter Notebook
- **Deployment Ready:** Docker, Uvicorn

---

## 📂 Dataset Info

> **Uber FOIL Trip Data**  
NYC Uber pickups from April–September 2014 & January–June 2015  
Data Source: [NYC TLC + FiveThirtyEight](https://github.com/fivethirtyeight/uber-tlc-foil-response)

| Column | Description |
|--------|-------------|
| Date/Time | Pickup timestamp |
| Lat, Lon | Pickup coordinates |
| Base | Uber dispatching base ID |

---

## 📈 Machine Learning Models

| Model           | Description                          | MAPE (%) |
|----------------|--------------------------------------|----------|
| **XGBoost**     | Tree-based boosting regressor         | 8.37     |
| **Random Forest** | Ensemble of decision trees            | 9.61     |
| **GBRT**        | Gradient Boosted Regression Trees     | 10.02    |
| **Ensemble**    | Weighted average based on inverse MAPE | 8.60     |

✅ TimeSeriesSplit + GridSearchCV used for hyperparameter tuning  
✅ Feature engineering includes hourly resampling and 24-hour lag features

---
## API (FastAPI)

[![API](https://img.shields.io/badge/Live-Render-success?style=for-the-badge&logo=fastapi)](https://uber-trip-analysis.onrender.com)

## 🔌 API Endpoints (FastAPI)

| Method | Endpoint      | Description                        |
|--------|---------------|------------------------------------|
| GET    | `/`           | Root welcome message               |
| GET    | `/health`     | Returns model load status          |
| GET    | `/metrics`    | Returns MAPE scores for all models |
| POST   | `/predict`    | Predicts hourly Uber trips         |

### 🔧 Example POST `/predict`
```json
{
  "hour": 14,
  "day": 12,
  "day_of_week": 2,
  "month": 5,
  "active_vehicles": 4120
}
