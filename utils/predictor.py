"""
Prediction utilities: single prediction, confidence interval, suggestions, NLP parsing.
"""

import json
import os
import re

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")

_rf     = None
_scaler = None
_feature_cols = None


def _load():
    global _rf, _scaler, _feature_cols
    if _rf is None:
        _rf     = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
        _scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
            _feature_cols = json.load(f)


def predict(features: dict) -> dict:
    """
    Returns:
      {
        "days": float,
        "low": float,   # 10th-percentile of tree predictions
        "high": float,  # 90th-percentile of tree predictions
        "weeks": float,
        "months": float,
        "suggestions": [...],
      }
    """
    _load()

    x = pd.DataFrame([[features[c] for c in _feature_cols]], columns=_feature_cols)
    x_s = _scaler.transform(x)

    # Point prediction
    pred = _rf.predict(x_s)[0]

    # Confidence interval from individual tree predictions
    tree_preds = np.array([t.predict(x_s)[0] for t in _rf.estimators_])
    low  = float(np.percentile(tree_preds, 10))
    high = float(np.percentile(tree_preds, 90))

    suggestions = _generate_suggestions(features, pred)

    return {
        "days":        round(pred, 1),
        "low":         round(low, 1),
        "high":        round(high, 1),
        "weeks":       round(pred / 7, 1),
        "months":      round(pred / 30, 1),
        "suggestions": suggestions,
    }


def _generate_suggestions(f: dict, pred_days: float) -> list[dict]:
    tips = []

    if f["num_employees"] < 5:
        delta = _delta_days(f, "num_employees", f["num_employees"] + 3)
        tips.append({
            "icon": "👥",
            "text": f"Add 3 more team members — could cut ~{abs(delta):.0f} days.",
        })

    if f["team_experience"] < 6:
        delta = _delta_days(f, "team_experience", min(f["team_experience"] + 2, 10))
        tips.append({
            "icon": "🎓",
            "text": f"Upskill or hire experienced devs — could save ~{abs(delta):.0f} days.",
        })

    if f["requirement_clarity"] < 6:
        delta = _delta_days(f, "requirement_clarity", min(f["requirement_clarity"] + 3, 10))
        tips.append({
            "icon": "📋",
            "text": f"Improve requirement clarity — could reduce ~{abs(delta):.0f} days of rework.",
        })

    if f["working_hours_day"] < 8:
        delta = _delta_days(f, "working_hours_day", min(f["working_hours_day"] + 2, 12))
        tips.append({
            "icon": "⏰",
            "text": f"Increase daily working hours — could shorten by ~{abs(delta):.0f} days.",
        })

    if f["deadline_pressure"] > 7:
        tips.append({
            "icon": "🧘",
            "text": "High deadline pressure increases errors & rework. Consider a realistic timeline buffer.",
        })

    if f["num_modules"] > 15:
        tips.append({
            "icon": "🧩",
            "text": "Break project into phases/sprints to improve focus and reduce context switching.",
        })

    if not tips:
        tips.append({
            "icon": "✅",
            "text": "Project parameters look well-balanced. Maintain communication cadence.",
        })

    return tips


def _delta_days(f: dict, key: str, new_val) -> float:
    """Difference in prediction when changing one feature."""
    _load()
    modified = {**f, key: new_val}
    x1 = pd.DataFrame([[modified[c] for c in _feature_cols]], columns=_feature_cols)
    x2 = pd.DataFrame([[f[c] for c in _feature_cols]], columns=_feature_cols)
    return _rf.predict(_scaler.transform(x1))[0] - _rf.predict(_scaler.transform(x2))[0]


# ---------------------------------------------------------------------------
# NLP parser – lightweight regex-based extraction (no external NLP deps)
# ---------------------------------------------------------------------------

_NLP_PATTERNS = {
    "num_employees": [
        r"(\d+)\s*(?:developers?|devs?|engineers?|team members?|people|persons?|employees?|programmers?)",
        r"team\s*of\s*(\d+)",
    ],
    "complexity":       [r"complexity[\s:=]+(\d+)", r"(\d+)[/\s]*10\s*complex"],
    "team_experience":  [r"experience[\s:=]+(\d+)", r"(\d+)\s*years?\s*(?:of\s*)?exp"],
    "num_modules":      [r"(\d+)\s*modules?", r"(\d+)\s*features?", r"(\d+)\s*components?"],
    "working_hours_day":[r"(\d+)\s*hours?\s*(?:a\s*|per\s*)?day", r"(\d+)h/day"],
    "deadline_pressure":[r"deadline[\s_]?pressure[\s:=]+(\d+)"],
    "requirement_clarity":[r"clarity[\s:=]+(\d+)", r"requirements?\s+clarity[\s:=]+(\d+)"],
}

_COMPLEXITY_WORDS = {
    "simple": 3, "easy": 2, "basic": 2, "small": 3,
    "medium": 5, "moderate": 5, "average": 5,
    "complex": 7, "complicated": 8, "hard": 8, "difficult": 8, "large": 7,
    "very complex": 9, "highly complex": 9, "enterprise": 9, "massive": 10,
}

DEFAULTS = {
    "num_employees": 5,
    "complexity": 5,
    "team_experience": 5,
    "num_modules": 8,
    "working_hours_day": 8,
    "deadline_pressure": 5,
    "requirement_clarity": 5,
}


def parse_nlp(text: str) -> dict:
    """Extract features from free-text project description."""
    text_lower = text.lower()
    result = {}

    for feat, patterns in _NLP_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, text_lower)
            if m:
                result[feat] = int(m.group(1))
                break

    # Complexity from adjectives if not found
    if "complexity" not in result:
        for word, val in sorted(_COMPLEXITY_WORDS.items(), key=lambda x: -len(x[0])):
            if word in text_lower:
                result["complexity"] = val
                break

    # Fill defaults for missing keys
    for k, v in DEFAULTS.items():
        result.setdefault(k, v)

    # Clamp values to valid ranges
    clamps = {
        "num_employees": (1, 50),
        "complexity": (1, 10),
        "team_experience": (1, 10),
        "num_modules": (1, 30),
        "working_hours_day": (4, 12),
        "deadline_pressure": (1, 10),
        "requirement_clarity": (1, 10),
    }
    for k, (lo, hi) in clamps.items():
        result[k] = max(lo, min(hi, result[k]))

    return result
