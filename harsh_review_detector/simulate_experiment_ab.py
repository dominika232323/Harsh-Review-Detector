import random
from datetime import datetime
from json import dumps
from pathlib import Path
import requests
import pandas as pd
from tqdm import tqdm

from harsh_review_detector.modeling.naive_bayes_train_and_evaluate import transform_function
from harsh_review_detector.service_utils import get_log_entry, save_log
from harsh_review_detector.config import PROCESSED_HOTEL_REVIEWS_DATASET, SERVICE_LOGS, AB_TEST_LOGS


#ENDPOINT = "http://localhost:8080/experiment_ab"
BASE_MODEL_ENDPOINT = "http://localhost:8080/predict/base-model"
ADVANCED_MODEL_ENDPOINT = "http://localhost:8080/predict/advanced-model"
TEXT_COLUMN = "comments"
LABEL_COLUMN = "label"

def save_log(log: str, path = AB_TEST_LOGS) -> None:
    #path.parent.mkdir(parents=True, exist_ok=True)
    print(log)
    with open(path, "w+", encoding="utf-8") as f:
        f.write(log)

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_HOTEL_REVIEWS_DATASET)

    network_success_count = 0
    network_failures = []
    log = ""
    base_sucesses = 0
    base_failures = 0
    advanced_sucesses = 0
    advanced_failures = 0

    print(f"Sending {len(df)} samples to {BASE_MODEL_ENDPOINT}...\n")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        user_id = random.randint(1, 100)

        payload = {
            "review": row[TEXT_COLUMN],
            "user_id": user_id,
            "true_label": row[LABEL_COLUMN]
        }

        try:
            response = requests.post(BASE_MODEL_ENDPOINT, json=payload)
            if response.status_code == 200:
                network_success_count += 1
                prediction = response.json()["prediction"]
                report = {
                    "timestamp" : str(datetime.now()),
                    "model" : "base model",
                    "review" : payload["review"],
                    "prediction" : prediction,
                    "true_label" : payload["true_label"],
                    "correct" : prediction == payload["true_label"]
                }
                if prediction == payload["true_label"]:
                    base_sucesses += 1
                else:
                    base_failures += 1
                log += dumps(report) + "\n"

            response = requests.post(ADVANCED_MODEL_ENDPOINT, json=payload)
            if response.status_code == 200:
                network_success_count += 1
                prediction = response.json()["prediction"]
                report = {
                    "timestamp" : str(datetime.now()),
                    "model" : "advanced model",
                    "review" : payload["review"],
                    "prediction" : prediction,
                    "true_label" : payload["true_label"],
                    "correct" : prediction == payload["true_label"]
                }
                if prediction == payload["true_label"]:
                    advanced_sucesses += 1
                else:
                    advanced_failures += 1
                log += dumps(report) + "\n"


            else:
                network_failures.append((idx, response.status_code, response.text))

        except Exception as e:
            network_failures.append((idx, "exception", str(e)))


    print(f"\nSuccess: {network_success_count}")
    print(f"Failures: {len(network_failures)}")
    base_string = f"Base model succeeded: {base_sucesses} Base model failed: {base_failures}. Success rate: {base_sucesses}/{base_sucesses + base_failures}\n"
    print(base_string)
    advanced_string = f"Advanced model succeeded: {advanced_sucesses} Base model failed: {advanced_failures}. Success rate: {advanced_sucesses}/{advanced_sucesses + advanced_failures}\n"
    print(advanced_string)
    log = base_string + advanced_string + log
    save_log(log)
    if network_failures:
        print("\nSome failures:")

        for fail in network_failures[:5]:
            print(f"Row {fail[0]} → {fail[1]}: {fail[2]}")

