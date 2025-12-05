import numpy as np
import pandas as pd
import requests
import streamlit as st

from bs4 import BeautifulSoup
from joblib import load
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning

from my_features import extract_features_from_html

# suppress SSL warnings for sites with bad certificates
disable_warnings(InsecureRequestWarning)


# -----------------------------
# Cached helpers
# -----------------------------
@st.cache_resource
def load_trained_model():
    """Load the best phishing detection model from disk."""
    model_path = "models/best_phishing_model.joblib"
    return load(model_path)


@st.cache_data
def load_metrics_table():
    """Load the training metrics if available."""
    try:
        return pd.read_csv("results/model_performance.csv")
    except FileNotFoundError:
        return None


# -----------------------------
# Utility functions
# -----------------------------
def normalize_url(url: str) -> str:
    """Ensure the URL has a scheme (http/https)."""
    url = url.strip()
    if not url:
        return url
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url
    return url


def fetch_html(url: str, timeout: float = 4.0) -> str | None:
    """Download page HTML; return None on failure."""
    try:
        resp = requests.get(url, timeout=timeout, verify=False)
        if resp.status_code == 200 and resp.text:
            return resp.text
        return None
    except Exception:
        return None


def model_predict(model, features: list[float | int]):
    """Run the model on a single feature vector."""
    X = np.array(features, dtype=float).reshape(1, -1)

    # Some models have predict_proba, some don't (e.g., LinearSVC)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        label = int(np.argmax(proba))
        confidence = float(np.max(proba))
    else:
        label = int(model.predict(X)[0])
        confidence = None

    return label, confidence


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️")
    st.title("🛡️ Phishing Website Detection")

    st.markdown(
        """
This tool uses a machine learning model that **you trained** to classify web pages as
either **legitimate (0)** or **phishing (1)**, based on content-based HTML features
(e.g., forms, inputs, images, scripts, structure, etc.).
        """
    )

    # Show training summary
    with st.expander("Training summary & model performance"):
        metrics = load_metrics_table()
        if metrics is None:
            st.info("No metrics file found at `results/model_performance.csv` yet.")
        else:
            st.write("Performance of each candidate model on the held-out test set:")
            st.dataframe(metrics)

            best_row = metrics.iloc[0]
            st.write(
                f"**Best model used in this app:** `{best_row['model']}`  "
                f"(accuracy = {best_row['accuracy']:.3f})"
            )

    st.subheader("Classify a website URL")

    url_input = st.text_input(
        "Enter a URL (with or without https://)", placeholder="example.com"
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        analyze = st.button("Analyze")

    if analyze:
        url = normalize_url(url_input)

        if not url:
            st.error("Please enter a URL first.")
            return

        with st.spinner("Fetching page and extracting features..."):
            html = fetch_html(url)
            if html is None:
                st.error("Could not fetch this URL (network error or non-200 status).")
                return

            # extract features using *your* feature extractor
            features = extract_features_from_html(html)
            model = load_trained_model()
            label, confidence = model_predict(model, features)

        # Interpretation: 0 = legit, 1 = phishing
        if label == 0:
            st.success("This page is classified as **LEGITIMATE**.")
        else:
            st.warning("This page is classified as **POTENTIAL PHISHING**.")

        if confidence is not None:
            st.write(f"Model confidence (for the predicted class): **{confidence:.3f}**")

        # Debug / transparency
        with st.expander("Show raw feature vector (for debugging)"):
            st.write(f"Vector length: {len(features)}")
            st.write(features)


if __name__ == "__main__":
    main()
