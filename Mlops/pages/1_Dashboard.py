import streamlit as st
import json
import os
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

MODEL_REGISTRY = BASE / "model_registry"
LOG_DIR = BASE / "logs"

st.title("📊 MLOps Monitoring Dashboard")

# -----------------------------
# Model Registry Section
# -----------------------------
st.header("🗂 Model Registry Overview")

versions = sorted([v for v in MODEL_REGISTRY.iterdir() if v.is_dir() and v.name.startswith("version_")])

if not versions:
    st.warning("No model versions found.")
else:
    version_data = []
    for v in versions:
        metrics_path = v / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            version_data.append({
                "Version": v.name,
                **metrics
            })

    st.dataframe(pd.DataFrame(version_data))

# -----------------------------
# Latest Model Metrics
# -----------------------------
st.header("📈 Latest Model Metrics")

if versions:
    latest = versions[-1]
    metrics_file = latest / "metrics.json"

    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        st.json(metrics)
    else:
        st.info("metrics.json not found for latest version.")


# -----------------------------
# Drift Monitoring
# -----------------------------
st.header("🔥 Drift Monitoring")

drift_path = LOG_DIR / "drift_logs.jsonl"

if drift_path.exists():
    drift_entries = []
    with open(drift_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                drift_entries.append(json.loads(line))
            except:
                pass

    if drift_entries:
        drift_df = pd.DataFrame(drift_entries)
        st.dataframe(drift_df.tail(20))
    else:
        st.info("No drift entries yet.")
else:
    st.info("drift_logs.jsonl not found.")

# -----------------------------
# Prediction Logs
# -----------------------------
st.header("📜 Prediction Logs")

prediction_path = LOG_DIR / "predictions.log"

if prediction_path.exists():
    st.text(prediction_path.read_text(encoding="utf-8", errors="replace"))
else:
    st.info("No prediction logs yet.")
