import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Response, jsonify

from harsh_review_detector.config import SERVICE_LOGS
from harsh_review_detector.service import base_model, advanced_model


def predict(data: dict[str, str], model_used: str) -> tuple[Response, int] | None | Response:
    validated = validate_data(data)

    if validated is not None:
        return validate_data(data)

    input_df = get_df_from_data(data)
    true_label = data.get("true_label", None)

    model = base_model if model_used == "base-model" else advanced_model
    prediction = int(model.predict(input_df)[0])

    log_entry = get_log_entry(data, "base-model", prediction, true_label)
    save_log(log_entry)

    return jsonify({
        "prediction": prediction
    })


def validate_data(data: dict[str, str] | None = None) -> tuple[Response, int] | None:
    if not data:
        return jsonify({"error": "Missing data"}), 400
    if "review" not in data:
        return jsonify({"error": "Missing 'review' field in input"}), 400
    return None


def get_df_from_data(data: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame([{
        "comments": data["review"],
        "review_length": len(data["review"])
    }])


def get_log_entry(data: dict[str, str], model_used: str, prediction: int, true_label: int | None) -> dict[str, str]:
    return {
        "timestamp": str(datetime.now()),
        "model_used": model_used,
        "review": data["review"],
        "review_length": len(data["review"]),
        "prediction": prediction,
        "true_label": true_label
    }


def save_log(log_entry: dict[str, str], path: Path = SERVICE_LOGS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
