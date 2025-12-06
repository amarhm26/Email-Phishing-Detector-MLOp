import os
import sys
import json
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import shutil

# --------------------------
# FIX PYTHON PATH
# --------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.preprocess import preprocess_text
from src.utils.logger import training_logger


# --------------------------
# PATHS
# --------------------------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE, "data", "processed", "emails_1000.csv")

REGISTRY_DIR = os.path.join(BASE, "model_registry")
LATEST_DIR = os.path.join(REGISTRY_DIR, "latest")

os.makedirs(REGISTRY_DIR, exist_ok=True)


# --------------------------
# LOAD DATASET
# --------------------------
def load_dataset():
    df = pd.read_csv(DATA_PATH)

    if not {"subject", "body", "label"}.issubset(df.columns):
        raise ValueError("CSV must include: subject, body, label")

    return df


# --------------------------
# PREPROCESS DATASET
# --------------------------
def preprocess_dataset(df):
    df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).apply(preprocess_text)
    return df


# --------------------------
# TRAIN MODEL
# --------------------------
def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=500, solver="saga")
    model.fit(X_train_vec, y_train)

    return model, vectorizer


# --------------------------
# EVALUATE MODEL
# --------------------------
def evaluate_model(model, vectorizer, X_test, y_test):
    X_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_vec)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4)
    }


# --------------------------
# SAFE DELETE (WINDOWS + ONEDRIVE FRIENDLY)
# --------------------------
def safe_clear_folder(path):
    if not os.path.exists(path):
        return

    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except:
                pass
        for d in dirs:
            try:
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
            except:
                pass


# --------------------------
# SAVE VERSIONED ARTIFACTS
# --------------------------
def save_versioned_artifacts(model, vectorizer, metrics, baseline):
    # List existing versions
    existing = [d for d in os.listdir(REGISTRY_DIR) if d.startswith("version_")]
    new_version = f"version_{len(existing) + 1}"

    VERSION_DIR = os.path.join(REGISTRY_DIR, new_version)
    os.makedirs(VERSION_DIR, exist_ok=True)

    # Save version artifacts
    joblib.dump(model, os.path.join(VERSION_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(VERSION_DIR, "vectorizer.pkl"))

    with open(os.path.join(VERSION_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(VERSION_DIR, "baseline.json"), "w") as f:
        json.dump(baseline, f, indent=4)

    training_logger.info(f"📦 Saved model as {new_version}")

    # --------------------------
    # Update latest folder safely
    # --------------------------
    if os.path.exists(LATEST_DIR):
        safe_clear_folder(LATEST_DIR)

    if not os.path.exists(LATEST_DIR):
        os.makedirs(LATEST_DIR)

    shutil.copytree(VERSION_DIR, LATEST_DIR, dirs_exist_ok=True)
    training_logger.info("🔄 Latest model updated.")

    return new_version


# --------------------------
# MAIN
# --------------------------
def main():
    training_logger.info("=== TRAINING STARTED ===")

    df = load_dataset()
    training_logger.info(f"Loaded {len(df)} emails.")

    df = preprocess_dataset(df)

    X = df["text"]
    y = df["label"]

    training_logger.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    training_logger.info("Training model...")
    model, vectorizer = train_model(X_train, y_train)

    training_logger.info("Evaluating model...")
    metrics = evaluate_model(model, vectorizer, X_test, y_test)
    training_logger.info(f"Metrics: {metrics}")

    baseline = {
        "phishing_rate": round(float(y_train.mean()), 4),
        "n_train": int(len(y_train)),
        "timestamp": datetime.utcnow().isoformat()
    }

    version = save_versioned_artifacts(model, vectorizer, metrics, baseline)

    training_logger.info(f"Version {version} saved successfully.")
    training_logger.info("=== TRAINING COMPLETE ===\n")


if __name__ == "__main__":
    main()
