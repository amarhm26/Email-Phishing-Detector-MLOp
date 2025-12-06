from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(BASE, "artifacts", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE, "artifacts", "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

class Email(BaseModel):
    subject: str
    body: str

@app.post("/predict")
async def predict(email: Email):
    text = (email.subject or "") + " " + (email.body or "")
    X = vectorizer.transform([text])
    score = model.predict_proba(X)[:, 1][0]
    label = "phishing" if score > 0.495 else "legitimate"
    return {"label": label, "score": score}
