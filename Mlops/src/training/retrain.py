import os
import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.preprocess import preprocess_text
from src.utils.logger import training_logger


# -------------------------------
# PATHS
# -------------------------------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_PATH = os.path.join(BASE, "data", "processed", "emails_1000.csv")

REGISTRY_DIR = os.path.join(BASE, "model_registry")
LATEST_DIR = os.path.join(REGISTRY_DIR, "latest")

os.makedirs(REGISTRY_DIR, exist_ok=True)


# -------------------------------
# Load dataset
# -------------------------------
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).apply(preprocess_text)
    return df


# -------------------------------
# Train model
# -------------------------------
def train(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=500, solver="saga")
    model.fit(X_vec, y_train)

    return model, vectorizer


# -------------------------------
# Evaluate model
# -------------------------------
def evaluate(model, vectorizer, X_test, y_test):
    X_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_vec)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4)
    }


# -------------------------------
# Load latest model metrics
# -------------------------------
def load_latest_metrics():
    metrics_path = os.path.join(LATEST_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)


# -------------------------------
# Compare models
# -------------------------------
def is_better(new_metrics, old_metrics):
    if old_metrics is None:
        return True  # No previous model → automatically accept

    return new_metrics["f1_score"] > old_metrics["f1_score"]


# -------------------------------
# Save new version
# -------------------------------
def save_new_version(model, vectorizer, metrics):
    import shutil
    
    existing = [d for d in os.listdir(REGISTRY_DIR) if d.startswith("version_")]
    new_version = f"version_{len(existing) + 1}"

    VERSION_DIR = os.path.join(REGISTRY_DIR, new_version)
    os.makedirs(VERSION_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(VERSION_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(VERSION_DIR, "vectorizer.pkl"))
    with open(os.path.join(VERSION_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Update latest
    if os.path.exists(LATEST_DIR):
        shutil.rmtree(LATEST_DIR)
    shutil.copytree(VERSION_DIR, LATEST_DIR)

    return new_version


# -------------------------------
# MAIN Retraining Logic
# -------------------------------
def main():
    training_logger.info("=== AUTOMATED RETRAINING STARTED ===")

    df = load_dataset()

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    training_logger.info("Training new model...")
    model, vectorizer = train(X_train, y_train)

    new_metrics = evaluate(model, vectorizer, X_test, y_test)
    training_logger.info(f"New model metrics: {new_metrics}")

    old_metrics = load_latest_metrics()
    training_logger.info(f"Previous model metrics: {old_metrics}")

    # Compare
    if is_better(new_metrics, old_metrics):
        version = save_new_version(model, vectorizer, new_metrics)
        training_logger.info(f"🎉 New model promoted as {version}")
    else:
        training_logger.info("❌ New model rejected — worse performance")

    training_logger.info("=== RETRAINING FINISHED ===\n")


if __name__ == "__main__":
    main()
