# TimeForge AI — Project Time Estimation System

A production-ready ML web application that predicts software project completion time using a Random Forest Regressor trained on 6,000 synthetic but realistic data points.

## Features

- **7 input features**: team size, complexity, experience, modules, working hours, deadline pressure, requirement clarity
- **Random Forest** (300 trees) achieving **R² ≈ 0.95**, **MAE ≈ 8 days**
- **Confidence interval** from individual tree predictions (10th–90th percentile)
- **Feature importance** visualization
- **Actionable suggestions** to reduce project duration
- **NLP mode**: describe your project in plain English → auto-extract parameters → predict
- Clean dark industrial UI built with Syne + Space Mono + DM Sans

## Project Structure

```
ai_time_estimator/
├── app.py                  # Flask application
├── requirements.txt
├── model/                  # Saved model artifacts (auto-generated)
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   ├── metrics.json
│   ├── feature_importances.json
│   └── feature_cols.json
├── utils/
│   ├── generate_dataset.py # Synthetic data generation
│   ├── train_model.py      # Training pipeline
│   └── predictor.py        # Inference + NLP parsing
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    ├── js/app.js
    └── img/                # Auto-generated plots
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (runs automatically on first launch too)
python utils/train_model.py

# Start the app
python app.py
# → Open http://localhost:5000
```

## Model Performance

| Metric | Value |
|--------|-------|
| MAE    | ~8.25 days |
| RMSE   | ~10.48 days |
| R²     | 0.9509 |
| CV-MAE | ~8.07 days |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main UI |
| `/predict` | POST | JSON prediction from structured features |
| `/predict_nlp` | POST | Prediction from free-text description |
| `/metrics` | GET | Model evaluation metrics |
| `/health` | GET | Health check |

### `/predict` example

```json
POST /predict
{
  "num_employees": 10,
  "complexity": 7,
  "team_experience": 5,
  "num_modules": 12,
  "working_hours_day": 8,
  "deadline_pressure": 6,
  "requirement_clarity": 7
}
```

Response:
```json
{
  "days": 84.3,
  "low": 72.1,
  "high": 96.8,
  "weeks": 12.0,
  "months": 2.8,
  "suggestions": [...]
}
```
