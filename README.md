# 🚕 Uber Trip Demand Forecasting – NYC (FOIL Dataset)

Forecasting hourly Uber trip demand using ML models like **XGBoost, Random Forest, GBRT**, and a custom **Ensemble**, built with full deployment support via FastAPI.

---

## 📌 Project Overview

This project leverages historical Uber FOIL trip data to analyse and forecast hourly demand in NYC. It features:

- 📊 Exploratory Data Analysis (EDA)
- 🛠️ Feature Engineering (incl. lag features)
- 🤖 Model Training & Tuning:
  - XGBoost
  - Random Forest
  - Gradient Boosted Trees (GBRT)
  - Weighted Ensemble
- 🚀 Deployment using FastAPI

---

## 📂 Dataset Information

**Source:** [NYC TLC FOIL via FiveThirtyEight](https://github.com/fivethirtyeight/uber-tlc-foil-response)  
**Timeframe:**  
- April–September 2014  
- January–June 2015  

| Column      | Description                    |
|-------------|--------------------------------|
| `Date/Time` | Timestamp of pickup            |
| `Lat`, `Lon`| Pickup location (coordinates)  |
| `Base`      | Uber dispatching base ID       |

---

## 🛠️ Tools & Technologies

- **Language:** Python 3
- **Core Libraries:**  
  `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`, `XGBoost`, `SHAP`, `Statsmodels`
- **API & Deployment:**  
  `FastAPI`, `Uvicorn`, `Docker`
- **Environment:**  
  `Jupyter Notebook`, `VS Code`

---

## 📈 Machine Learning Models & Performance

| Model             | Description                            | MAPE (%) |
|------------------|----------------------------------------|----------|
| 🧠 **XGBoost**     | Boosted decision tree regressor         | 8.37     |
| 🌲 **Random Forest** | Bagging ensemble of decision trees      | 9.61     |
| 📉 **GBRT**        | Gradient Boosted Regression Trees       | 10.02    |
| 🤝 **Ensemble**    | Weighted average (based on inverse MAPE)| 8.60     |

✅ **Cross-validation:** `TimeSeriesSplit`  
✅ **Tuning:** `GridSearchCV`  
✅ **Features:** Hourly resampling, 24-hour lag, day-of-week & month

---

## 🚀 Live API Dashboard

[![Live Render](https://img.shields.io/badge/Live-Dashboard-00c853?style=for-the-badge&logo=fastapi)](https://uber-trip-analysis.onrender.com)

---

## 🔌 API Endpoints (FastAPI)

| Method | Endpoint      | Description                        |
|--------|---------------|------------------------------------|
| `GET`  | `/`           | Returns interactive dashboard      |
| `GET`  | `/health`     | Model load status                  |
| `GET`  | `/metrics`    | MAPE scores for all models         |
| `POST` | `/predict`    | Predicts hourly Uber trip counts   |

### 🔧 Sample POST `/predict` Request

```json
{
  "hour": 14,
  "day": 12,
  "day_of_week": 2,
  "month": 5,
  "active_vehicles": 4100
}
```

---
# 📄 PDF Export – Uber Trip Forecasting Dashboard

The dashboard includes a **PDF export feature** that generates a full visual report containing:

### ✅ What’s Inside the PDF:

| Column                   | Description                    |
|--------------------------|--------------------------------|
| `📊 Forecast vs Actual`  |- XGBoost                       |
|                          |- Random Forest                 |
|                          |- Ensemble Models               |
| `🕒 Time Series Insights`|- Train-Test Split              |
|                          |- Seasonal Decomposition        |
| `🧠 SHAP Explainability` |-Feature importance visualised  |
|                          |using SHAP summary plots        |
| `📝 Plot Descriptions`   |- A descriptive interpretation  | 
|                          |accompanies each graph          |
| `📄 Page Footer`         |-Dynamic footer showing         |
|                          |` Page X of Y` on every page    |

---


## 👨‍💻 Author

**Jayanth Chennoju**  
*Built with ❤️ using FastAPI, XGBoost, SHAP, and Plotly*

---
## 🤝 Connect with Me

- [**🪪LinkedIn Profile**](https://www.linkedin.com/in/jayanth-chennoju-5a738923k/)
  
- [**📧Email**](mailto:jayanthchennoju@gmail.com)
