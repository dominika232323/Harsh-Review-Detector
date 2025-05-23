from pathlib import Path


PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

RAW_REVIEWS_DATASET = RAW_DATA_DIR / "reviews.csv"
PROCESSED_REVIEWS_DATASET = PROCESSED_DATA_DIR / "reviews.csv"

RAW_HOTEL_REVIEWS_DATASET = RAW_DATA_DIR / "Datafiniti_Hotel_Reviews.csv"
PROCESSED_HOTEL_REVIEWS_DATASET = PROCESSED_DATA_DIR / "hotel_reviews.csv"

MODELS_DIR = PROJ_ROOT / "models"
BASE_MODEL = MODELS_DIR / "base_model.pkl"
ADVANCED_MODEL = MODELS_DIR / "advanced_model.pkl"

LOGS_DIR = PROJ_ROOT / "logs"
SERVICE_LOGS = LOGS_DIR / "service_log.jsonl"
