# src/features/tfidf_vectorizer.py

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

VECTORIZER_PATH = "artifacts/vectorizer.pkl"

def create_vectorizer():
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )
    return vectorizer

def save_vectorizer(vectorizer, path=VECTORIZER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)

def load_vectorizer(path=VECTORIZER_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectorizer not found at {path}")
    return joblib.load(path)

def get_vectorizer():
    if os.path.exists(VECTORIZER_PATH):
        return load_vectorizer()
    return create_vectorizer()
