"""
Flask web application for AI-Based Project Time Estimation System.
"""

import json
import os
import sys

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(__file__))
from utils.predictor import parse_nlp, predict

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_importances():
    path = os.path.join(MODEL_DIR, "feature_importances.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------- Routes ----------

@app.route("/")
def index():
    metrics     = load_metrics()
    importances = load_importances()
    return render_template("index.html", metrics=metrics, importances=importances)


@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        data = request.get_json(force=True)

        # Validation
        required = [
            "num_employees", "complexity", "team_experience",
            "num_modules", "working_hours_day", "deadline_pressure",
            "requirement_clarity",
        ]
        ranges = {
            "num_employees":      (1, 50),
            "complexity":         (1, 10),
            "team_experience":    (1, 10),
            "num_modules":        (1, 30),
            "working_hours_day":  (4, 12),
            "deadline_pressure":  (1, 10),
            "requirement_clarity":(1, 10),
        }

        features = {}
        for key in required:
            if key not in data:
                return jsonify({"error": f"Missing field: {key}"}), 400
            try:
                val = float(data[key])
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid value for {key}"}), 400
            lo, hi = ranges[key]
            if not (lo <= val <= hi):
                return jsonify({"error": f"{key} must be between {lo} and {hi}"}), 400
            features[key] = val

        result = predict(features)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_nlp", methods=["POST"])
def predict_nlp_route():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if len(text) > 2000:
            return jsonify({"error": "Text too long (max 2000 chars)"}), 400

        features = parse_nlp(text)
        result   = predict(features)
        result["parsed_features"] = features
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
def metrics_route():
    return jsonify(load_metrics())


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Train model if not already trained
    if not os.path.exists(os.path.join(MODEL_DIR, "rf_model.pkl")):
        print("Model not found. Training now …")
        from utils.train_model import train
        train()

    app.run(debug=False, host="0.0.0.0", port=5000)
