from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from harsh_review_detector.config import PROCESSED_REVIEWS_DATASET


def load_dataset(path: Path = PROCESSED_REVIEWS_DATASET):
    return pd.read_csv(path)


def split_dataset(df: pd.DataFrame, features: list[str], target: str = "label", test_size: float = 0.2):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size)

    return X_train, X_test, y_train, y_test
