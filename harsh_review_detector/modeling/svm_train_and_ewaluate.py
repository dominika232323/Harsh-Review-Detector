from pathlib import Path
from typing import Any

import joblib
import wandb

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from harsh_review_detector.config import MODELS_DIR
from harsh_review_detector.modeling.dataset_operations import load_dataset, split_dataset


def get_preprocessor(
        config: dict[str, Any],
        numerical_features_columns: list[str],
        text_column: str,
        strategy: str="mean"
) -> ColumnTransformer:

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=config["tfidf_max_features"],
                                     ngram_range=config["tfidf_ngram_range"]), text_column),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy=strategy)),
                ("scaler", StandardScaler())
            ]), numerical_features_columns)
        ]
    )
    return preprocessor


def get_pipeline(config: dict[str, Any], preprocessor: ColumnTransformer) -> Pipeline:
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", LinearSVC(C=config["svm_C"], class_weight=config["svm_class_weight"]))
    ])
    return pipeline


def get_metrics(y_test: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred)
    }


def save_svm_weights(pipeline: ColumnTransformer, run_name: str | None, directory: Path = MODELS_DIR) -> None:
    model_path = directory / f"svm_model_{run_name}.pkl"
    joblib.dump(pipeline, model_path)

    artifact = wandb.Artifact(f"svm_model_{run_name}", type="model")
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)


def main():
    df = load_dataset()

    text_column = "comments"
    numerical_features_columns = ["review_length"]
    features_columns = [text_column] + numerical_features_columns
    target_column = "label"

    X_train, X_test, y_train, y_test = split_dataset(df, features_columns, target_column)

    config = {
        "model": "SVM",
        "tfidf_max_features": 5000,
        "tfidf_ngram_range": (1, 2),
        "svm_C": 1.0,
        "svm_class_weight": "balanced"
    }

    run = wandb.init(project="ium-harsh-reviews", config=config)

    preprocessor = get_preprocessor(config, numerical_features_columns, text_column)
    pipeline = get_pipeline(config, preprocessor)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = get_metrics(y_test, y_pred)

    wandb.log(metrics)
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test.tolist(),
        preds=y_pred.tolist(),
        class_names=["Not Harsh", "Harsh"]
    )})

    run_name = run.name
    save_svm_weights(pipeline, run_name)

    wandb.finish()


if __name__ == "__main__":
    main()
