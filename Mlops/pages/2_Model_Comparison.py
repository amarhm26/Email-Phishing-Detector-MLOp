import os
import json
import pandas as pd
import streamlit as st

# -------------------------
# PATHS
# -------------------------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY = os.path.join(BASE, "model_registry")

# -------------------------
# STREAMLIT PAGE
# -------------------------
st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("📊 Model Comparison Dashboard")
st.write("Track the performance of every model version over time.")


# ------------------------------------------------
# Helper: Load metrics from version folder
# ------------------------------------------------
def load_version_metrics(version_folder):
    metrics_path = os.path.join(version_folder, "metrics.json")
    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, "r") as f:
        return json.load(f)


# ------------------------------------------------
# Read all version folders
# ------------------------------------------------
versions = sorted(
    [v for v in os.listdir(REGISTRY) if v.startswith("version_")],
    key=lambda x: int(x.split("_")[1])
)

if not versions:
    st.warning("No models found in model_registry/")
    st.stop()


# -------------------------
# Load metrics for all versions
# -------------------------
records = []

for v in versions:
    v_path = os.path.join(REGISTRY, v)
    met = load_version_metrics(v_path)

    if met:
        records.append({
            "version": v,
            "accuracy": met.get("accuracy"),
            "precision": met.get("precision"),
            "recall": met.get("recall"),
            "f1_score": met.get("f1_score")
        })


df = pd.DataFrame(records)


# -------------------------
# Version Table
# -------------------------
st.subheader("📄 All Model Versions")
st.dataframe(df, use_container_width=True)


# -------------------------
# Line Chart Comparison
# -------------------------
st.subheader("📈 Performance Over Versions")

metric = st.selectbox(
    "Select a metric to compare:",
    ["accuracy", "precision", "recall", "f1_score"]
)

st.line_chart(df.set_index("version")[metric])


# -------------------------
# Best Model Highlight
# -------------------------
best_version = df.loc[df["f1_score"].idxmax()]

st.subheader("🏆 Best Model")
st.success(
    f"""
**Best Version:** {best_version['version']}  
**F1 Score:** {best_version['f1_score']}  
**Accuracy:** {best_version['accuracy']}  
**Precision:** {best_version['precision']}  
**Recall:** {best_version['recall']}  
    """
)
