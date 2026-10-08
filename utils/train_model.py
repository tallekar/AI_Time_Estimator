"""
Model training: Random Forest Regressor for project time estimation.
Produces model artifacts in the /model directory.
"""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Make sure we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.generate_dataset import generate_dataset

FEATURE_COLS = [
    "num_employees",
    "complexity",
    "team_experience",
    "num_modules",
    "working_hours_day",
    "deadline_pressure",
    "requirement_clarity",
]
TARGET_COL = "completion_days"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(STATIC_DIR, "img"), exist_ok=True)

    # 1. Data
    print("Generating dataset …")
    df = generate_dataset(n_samples=6000)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Scaling (kept for future use; RF doesn't need it but pipeline is robust)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 3. Train
    print("Training Random Forest …")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    # 4. Evaluate
    y_pred = rf.predict(X_test_s)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(rf, scaler.transform(X), y, cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -cv.mean()

    metrics = {
        "mae":    round(mae, 2),
        "rmse":   round(rmse, 2),
        "r2":     round(r2, 4),
        "cv_mae": round(cv_mae, 2),
    }
    print(f"MAE={mae:.2f}d  RMSE={rmse:.2f}d  R²={r2:.4f}  CV-MAE={cv_mae:.2f}d")

    # 5. Feature importances
    importances = dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))

    # 6. Save artifacts
    joblib.dump(rf,     os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(MODEL_DIR, "feature_importances.json"), "w") as f:
        json.dump(importances, f, indent=2)

    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(FEATURE_COLS, f)

    # 7. Feature importance plot
    _plot_feature_importance(importances)

    # 8. Prediction vs Actual scatter (sample)
    _plot_pred_vs_actual(y_test[:300].values, y_pred[:300])

    print("Model artifacts saved to /model/")
    return metrics


def _plot_feature_importance(importances: dict):
    labels = [f.replace("_", " ").title() for f in importances]
    values = list(importances.values())
    idx    = np.argsort(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        [labels[i] for i in idx],
        [values[i] for i in idx],
        color="#4f8ef7",
        edgecolor="none",
    )
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title("Feature Importances", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, [values[i] for i in idx]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "img", "feature_importance.png"), dpi=120)
    plt.close(fig)


def _plot_pred_vs_actual(actual, predicted):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.45, s=18, color="#4f8ef7", edgecolors="none")
    lims = [min(actual.min(), predicted.min()) - 5,
            max(actual.max(), predicted.max()) + 5]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual (days)", fontsize=11)
    ax.set_ylabel("Predicted (days)", fontsize=11)
    ax.set_title("Predicted vs Actual", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "img", "pred_vs_actual.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    train()
