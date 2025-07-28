# 🚕 Uber Trip Demand Forecasting – NYC (FOIL Dataset)

Forecasting hourly Uber trip demand using ML models like **XGBoost, Random Forest, GBRT**, and a custom **Ensemble**, built with full deployment support via FastAPI.

---

## 📂 Project Structure

```
Uber_Trip_Analysis/
│
├── data/
│   ├── raw/                     # Original FOIL dataset (e.g., Uber-Jan-Feb-FOIL.csv)
│   ├── processed/               # Cleaned & preprocessed datasets
│   └── external/                # Any additional or reference datasets
│
├── scripts/
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── eda_visuals.py           # EDA visualizations and reusable plotting scripts
│   ├── feature_engineering.py   # Feature extraction and transformations
│   └── model_training.py        # Model training, evaluation, and metrics
│
├── app/
│   ├── main.py                  # Entry point for Streamlit/FastAPI/Gradio
│   ├── utils.py                 # Helper functions and configurations
│   └── config.py                # Global configuration (paths, constants, etc.)
│
├── tests/
│   ├── test_data_pipeline.py    # Tests for data preprocessing pipeline
│   └── test_model.py            # Tests for model functionality
│
├── reports/
│   ├── figures/                 # Exported charts, plots, and visual assets
│   └── summary.pdf              # Project report (if any)
│
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment file (optional)
├── Dockerfile                   # For containerization (if required)
├── .gitignore                   # Files/folders ignored in Git
├── README.md                    # Project documentation
└── LICENSE                      # License file
```

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
## ⚙️ Installation

Clone the repository and set up your environment:

```bash
# Clone the repository
git clone https://github.com/Jayanth2323/Uber_Trip_Analysis.git
cd Uber_Trip_Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows

# Install dependencies
pip install -r requirements.txt
```

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

[![Live Render](https://img.shields.io/badge/Live-Dashboard-00c853?style=for-the-badge&logo=fastapi)](https://uber-trip-analysis-7sbt.onrender.com/)

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

**GitHub** — [Jayanth2323](https://github.com/Jayanth2323).

**LinkedIn** — [**🪪-Jayanth Chennoju**](https://linkedin.com/in/jayanth-chennoju-5a738923k/).

**Gmail** — [**📧Mailto**](mailto:jayanthchennoju@gmail.com)
