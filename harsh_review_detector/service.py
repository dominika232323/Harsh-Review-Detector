import random
import joblib
import json
from datetime import datetime

import pandas as pd
from flask import Flask, request, jsonify

from harsh_review_detector.config import BASE_MODEL, ADVANCED_MODEL, SERVICE_LOGS

base_model = joblib.load(BASE_MODEL)
advanced_model = joblib.load(ADVANCED_MODEL)

app = Flask(__name__)

@app.route("/predict/advanced-model", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Missing data"}), 400
    if "review" not in data:
        return jsonify({"error": "Missing 'review' field in input"}), 400

    review_text = data["review"]
    review_length = len(review_text)
    true_label = data.get("true_label", None)

    model = advanced_model

    input_df = pd.DataFrame([{
        "comments": review_text,
        "review_length": review_length
    }])

    prediction = int(model.predict(input_df)[0])

    log_entry = {
        "timestamp": str(datetime.now()),
        "model_used": "advanced-model",
        "review": review_text,
        "review_length": review_length,
        "prediction": prediction,
        "true_label": true_label
    }

    SERVICE_LOGS.parent.mkdir(parents=True, exist_ok=True)

    with open(SERVICE_LOGS, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return jsonify({
        "prediction": prediction
    })


@app.route("/predict/base-model", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Missing data"}), 400
    if "review" not in data:
        return jsonify({"error": "Missing 'review' field in input"}), 400

    review_text = data["review"]
    review_length = len(review_text)
    true_label = data.get("true_label", None)

    model = base_model

    input_df = pd.DataFrame([{
        "comments": review_text,
        "review_length": review_length
    }])

    prediction = int(model.predict(input_df)[0])

    log_entry = {
        "timestamp": str(datetime.now()),
        "model_used": "base-model",
        "review": review_text,
        "review_length": review_length,
        "prediction": prediction,
        "true_label": true_label
    }

    SERVICE_LOGS.parent.mkdir(parents=True, exist_ok=True)

    with open(SERVICE_LOGS, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return jsonify({
        "prediction": prediction
    })


@app.route("/experiment_ab", methods=["POST"])
def experiment_ab():
    data = request.get_json()

    if not data or "review" not in data:
        return jsonify({"error": "Missing 'review' field in input"}), 400

    review_text = data["review"]
    review_length = len(review_text)
    true_label = data.get("true_label", None)

    model_used = random.choice(["baseline", "advanced"])
    model = base_model if model_used == "baseline" else advanced_model

    input_df = pd.DataFrame([{
        "comments": review_text,
        "review_length": review_length
    }])

    prediction = int(model.predict(input_df)[0])

    log_entry = {
        "timestamp": str(datetime.now()),
        "model_used": model_used,
        "review": review_text,
        "review_length": review_length,
        "prediction": prediction,
        "true_label": true_label
    }

    SERVICE_LOGS.parent.mkdir(parents=True, exist_ok=True)

    with open(SERVICE_LOGS, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return jsonify({
        "prediction": prediction,
        "model_used": model_used
    })



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
