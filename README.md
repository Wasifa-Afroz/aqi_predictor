# 🌫️ Karachi AQI Predictor

Real-time air quality monitoring and 72-hour forecast system powered by machine learning.

**🚀 Live Demo**: https://aqipredictor-9vmzkfmxqvlv9svubxwilu.streamlit.app/

---

## 📊 Overview

Automated end-to-end ML system that:
- Collects hourly air quality data from OpenWeather API
- Trains **3 separate ML models** daily — one for each forecast horizon (24h, 48h, 72h)
- Provides **real, consistent** AQI predictions for current + next 3 days
- Displays a beautiful interactive dashboard with health recommendations

**Current Performance**: R² = 0.569 | RMSE = 15.18 | MAE = 11.90

---

## 🔄 Automation Workflow

### ⏰ **Every Hour (Automatic)**
```
GitHub Actions → OpenWeather API → Feature Engineering → MongoDB
```
- Fetches current weather + AQI data for Karachi
- Engineers 62 features (time-based, lag, rolling averages, pollutants)
- Stores in MongoDB Atlas collection `aqi_features`

### 📅 **Daily at 2:00 AM PKT (Automatic)**
```
MongoDB → Load Data → Train 3×3 Models → Save Best per Horizon → Upload Artifacts
```
- Loads all historical data from MongoDB
- Trains 3 algorithms × 3 horizons = 9 models total
- Selects best model per forecast horizon (currently LightGBM wins all 3)
- Uploads trained models as GitHub Actions artifacts (30-day retention)

### 🔘 **Manual Trigger (On-Demand)**
- Go to [Actions](https://github.com/Wasifa-Afroz/aqi_predictor/actions)
- Click "Run workflow" → Choose `feature` / `training` / `both`

---

## ✨ Features

**Dashboard (5 Pages via Navigation Menu)**:
- 📊 **Dashboard** — Live AQI card, weather conditions, 3-day ML forecast, trend chart
- 📈 **Analytics & Metrics** — All model performance scores, SHAP feature importance
- 📋 **Historical Data** — Last 10 days table with AQI stats + CSV download
- 🧠 **Model Details** — Architecture, hyperparameters, R² scores, training info
- 💡 **Health Guide** — Mask advice, outdoor activity guide, clothing recommendations by AQI level

**ML Prediction System**:
- 3 separately trained models (model_24h.pkl, model_48h.pkl, model_72h.pkl)
- Predictions are **deterministic** — same input always gives same output
- No random values — consistent results on every dashboard refresh

---

## 🗄️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Models** | LightGBM, XGBoost, Random Forest, Ridge |
| **Data Processing** | Python, Pandas, NumPy, Scikit-learn |
| **Storage** | MongoDB Atlas (cloud feature store) |
| **Dashboard** | Streamlit, Plotly |
| **Automation** | GitHub Actions (cron schedules) |
| **API** | OpenWeather Air Pollution + Weather APIs |
| **Deployment** | Streamlit Cloud |

---

## 📂 Data Storage

**MongoDB Database**: `aqi_predictor` → **Collection**: `aqi_features`

```json
{
  "timestamp": "2026-02-15T19:00:00",
  "city": "karachi",
  "aqi": 150,
  "pm25": 65,
  "pm10": 90,
  "temperature": 25,
  "humidity": 50,
  "pressure": 1013,
  "wind_speed": 5,
  "hour": 19,
  "day_of_week": 5,
  "aqi_lag_24": 145,
  "aqi_rolling_mean_24": 148,
  "target_aqi_24h": 155,
  "target_aqi_48h": 162,
  "target_aqi_72h": 158
}
```

**Current Dataset**:
- 180+ days of historical data
- 4,248+ samples with 62 features each
- Grows automatically by +1 record every hour

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Wasifa-Afroz/aqi_predictor.git
cd aqi_predictor

# Install dependencies
pip install -r requirements.txt

# Create .env file with:
# OPENWEATHER_API_KEY=your_key
# MONGODB_URI=mongodb+srv://...
# MONGODB_DB_NAME=aqi_predictor

# Train 3 models (loads from MongoDB automatically)
python src/training_pipeline.py

# Run dashboard
streamlit run app.py
```

---

## 📈 Performance Metrics

| Model | R² Score | RMSE | MAE | Status |
|-------|----------|------|-----|--------|
| **LightGBM** | **0.569** | **15.18** | **11.90** | ✅ Active (Best) |
| XGBoost | 0.551 | 15.49 | 12.30 | Available |
| Random Forest | 0.549 | 15.53 | 12.09 | Available |
| Ridge Regression | 0.400 | 27.25 | 21.35 | Baseline |

**Target**: R² = 0.7+ (improves automatically as more real data accumulates — ~90 more days needed)

---


## 🎯 Automation Schedule

| Time | Action | Duration |
|------|--------|----------|
| **Every hour** | 📡 Data Collection | ~20s |
| **Daily 2:00 AM PKT** | 🤖 Model Training (3 models) | ~8 min |
| **On-demand** | 🔘 Manual Trigger | Variable |

**Total**: 24 collections/day + 1 training/day = **100% automated**

---

## 🔗 Links

- **Live Dashboard**: [aqipredictor-5wky2jfgigj5ex9fwfvfr.streamlit.app](https://aqipredictor-9vmzkfmxqvlv9svubxwilu.streamlit.app/)
- **GitHub Repo**: [github.com/Wasifa-Afroz/aqi_predictor](https://github.com/Wasifa-Afroz/aqi_predictor)
- **Automation Logs**: [GitHub Actions](https://github.com/Wasifa-Afroz/aqi_predictor/actions)

---

## 👤 Author

**Wasifa Afroz**
GitHub: [@Wasifa-Afroz](https://github.com/Wasifa-Afroz)

---

**Built with ❤️ for Karachi** | *Helping people make informed decisions about air quality* | Data Science Project 2026
