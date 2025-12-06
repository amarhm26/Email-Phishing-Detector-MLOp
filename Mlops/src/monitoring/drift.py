import os
import json
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASELINE_PATH = os.path.join(ROOT, "artifacts", "baseline.json")

def load_baseline():
    if not os.path.exists(BASELINE_PATH):
        return None
    with open(BASELINE_PATH, "r") as f:
        return json.load(f)

def compute_phishing_rate(predictions):
    if not predictions:
        return 0.0
    phishing = sum(1 for p in predictions if p.get("label") == "phishing")
    return phishing / len(predictions)

def monitor_drift(predictions, threshold=0.10):
    """
    Compare current phishing rate to baseline.
    Returns None if no drift, otherwise returns a message describing drift.
    """
    baseline = load_baseline()
    if not baseline:
        # no baseline available; record and return no drift
        return None

    baseline_rate = float(baseline.get("phishing_rate", 0.0))
    current_rate = compute_phishing_rate(predictions)
    delta = abs(current_rate - baseline_rate)

    # relative change could be used as well; keep absolute for simplicity
    if delta >= threshold:
        msg = {
            "baseline_rate": baseline_rate,
            "current_rate": round(current_rate, 4),
            "delta": round(delta, 4),
            "threshold": threshold,
            "timestamp": datetime.utcnow().isoformat()
        }
        # we'll return the dict (caller will log it)
        return msg
    return None
