import random
import logging

from flask import Flask, request, jsonify

from harsh_review_detector.modeling.naive_bayes_train_and_evaluate import transform_function
from harsh_review_detector.service_utils import predict


ab_experiment_users = {
    "advanced-model": [],
    "base-model": []
}

app = Flask(__name__)

@app.route("/predict/advanced-model", methods=["POST"])
def predict_advanced_model():
    data = request.get_json()
    return predict(data, "advanced-model")


@app.route("/predict/base-model", methods=["POST"])
def predict_base_model():
    data = request.get_json()
    return predict(data, "base-model")


@app.route("/experiment_ab", methods=["POST"])
def experiment_ab():
    data = request.get_json()

    user_id = data.get("user_id", None)

    if user_id is None:
        return jsonify({"error": "Missing 'user_id' field in input"}), 400

    if user_id in ab_experiment_users["advanced-model"]:
        model_used = "advanced-model"
    elif user_id in ab_experiment_users["base-model"]:
        model_used = "base-model"
    else:
        if len(ab_experiment_users["advanced-model"]) <= len(ab_experiment_users["base-model"]):
            model_used = "advanced-model"
        else:
            model_used = "base-model"

        ab_experiment_users[model_used].append(user_id)

    return predict(data, model_used, user_id)

def run_in_background():
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=8080)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)