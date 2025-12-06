import pytest
import os
import joblib

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE, "artifacts", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE, "artifacts", "vectorizer.pkl")

@pytest.fixture(scope="module")
def model_and_vectorizer():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def test_model_prediction(model_and_vectorizer):
    model, vectorizer = model_and_vectorizer
    sample_text = "Urgent: Update your account information now"
    X = vectorizer.transform([sample_text])
    pred = model.predict(X)
    assert pred[0] in [0, 1]  # Should return legitimate (0) or phishing (1)
