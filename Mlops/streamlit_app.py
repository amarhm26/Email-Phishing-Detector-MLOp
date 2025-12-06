import os
import sys

# -----------------------------
# FIX PYTHON IMPORT PATH
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st
import pandas as pd

# Import IMAP fetch + prediction function
from src.ingestion.fetch_and_predict_imap import fetch_and_predict


# -----------------------------
# STREAMLIT PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Email Phishing Detector",
    page_icon="📧",
    layout="wide",
)


# -----------------------------
# PAGE UI
# -----------------------------
st.title("📧 Email Phishing Detection Dashboard")
st.write(
    """
This tool automatically fetches your recent emails and classifies them as  
**phishing** or **legitimate** using your trained machine learning model.
"""
)

st.markdown("---")


# -----------------------------
# FETCH & PREDICT BUTTON
# -----------------------------
if st.button("🔄 Fetch & Analyze Emails", use_container_width=True):
    with st.spinner("Connecting to mailbox and analyzing emails... Please wait."):
        results = fetch_and_predict()

    # No results
    if not results:
        st.warning("⚠️ No emails found or IMAP connection issue. Check your .env settings.")
    else:
        # Convert results to DataFrame
        df = pd.DataFrame(results)

        st.success(f"✅ Analysis complete! {len(df)} emails processed.")

        # Show table
        st.subheader("📊 Prediction Results")
        st.dataframe(df, use_container_width=True, height=600)

        # CSV Download
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Results (CSV)",
            data=csv,
            file_name="email_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.info("Click **Fetch & Analyze Emails** to start retrieving and classifying your emails.")
