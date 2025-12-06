# src/models/retrain.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Absolute imports for your project structure
from src.features.tfidf_vectorizer import get_vectorizer, save_vectorizer
from src.preprocessing.text_preprocess import preprocess

# Paths
DATA_PATH = "data/processed/emails_1000.csv"  # Your CSV with 1000 emails
MODEL_PATH = "artifacts/model.pkl"
VECTORIZER_PATH = "artifacts/vectorizer.pkl"

def retrain_model():
    # Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)

    # Combine subject + body into 'text' column and preprocess
    df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).apply(preprocess)

    # Features and labels
    X = df["text"]
    y = df["label"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Get TF-IDF vectorizer
    vectorizer = get_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    save_vectorizer(vectorizer, VECTORIZER_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")
    print(f"[INFO] Vectorizer saved to {VECTORIZER_PATH}")

if __name__ == "__main__":
    retrain_model()
