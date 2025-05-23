import random

import requests
import pandas as pd
from tqdm import tqdm

from harsh_review_detector.config import PROCESSED_HOTEL_REVIEWS_DATASET


ENDPOINT = "http://localhost:8080/experiment_ab"
TEXT_COLUMN = "comments"
LABEL_COLUMN = "label"

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_HOTEL_REVIEWS_DATASET)

    success_count = 0
    failures = []

    print(f"Sending {len(df)} samples to {ENDPOINT}...\n")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        user_id = random.randint(1, 100)

        payload = {
            "review": row[TEXT_COLUMN],
            "user_id": user_id,
            "true_label": row[LABEL_COLUMN]
        }

        try:
            response = requests.post(ENDPOINT, json=payload)

            if response.status_code == 200:
                success_count += 1
            else:
                failures.append((idx, response.status_code, response.text))

        except Exception as e:
            failures.append((idx, "exception", str(e)))


    print(f"\nSuccess: {success_count}")
    print(f"Failures: {len(failures)}")

    if failures:
        print("\nSome failures:")

        for fail in failures[:5]:
            print(f"Row {fail[0]} → {fail[1]}: {fail[2]}")
