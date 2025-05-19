import joblib
import wandb

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from harsh_review_detector.config import MODELS_DIR
from harsh_review_detector.modeling.dataset_operations import load_dataset, split_dataset


if __name__ == "__main__":
    df = load_dataset()

    text_column = "comments"
    numerical_features_columns = ["numerical_review", "sentiment", "review_length"]
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

    wandb.init(project="ium-harsh-reviews", name="svm-with-numeric", config=config)

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=config["tfidf_max_features"],
                                     ngram_range=config["tfidf_ngram_range"]), text_column),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), numerical_features_columns)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", LinearSVC(C=config["svm_C"], class_weight=config["svm_class_weight"]))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred)
    }

    wandb.log(metrics)

    cm = confusion_matrix(y_test, y_pred)
    labels = ["Not Harsh", "Harsh"]
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})
    plt.close()

    print(metrics)

    timestamp = wandb.run.id
    joblib.dump(pipeline, MODELS_DIR / f"svm_model_{timestamp}.pkl")
    wandb.save(f"svm_model_{timestamp}.pkl")

    wandb.finish()
