import streamlit as st
import requests

from src.schema import FEATURE_COLUMNS

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Thunderstorm Prediction")

st.title("🌩 Thunderstorm Prediction")

model_name = st.selectbox(
    "Select Model",
    ["RandomForest", "XGBoost", "LogisticRegression"]
)

st.subheader("Input Features")

payload = {}
for feature in FEATURE_COLUMNS:
    payload[feature] = st.number_input(
        feature,
        value=0.0,
        format="%.4f"
    )

if st.button("Predict"):
    response = requests.post(
        API_URL,
        params={"model_name": model_name},
        json=payload
    )

    if response.status_code == 200:
        result = response.json()

        st.metric(
            "Thunderstorm Probability",
            f"{result['probability']:.3f}"
        )

        if result["prediction"] == 1:
            st.error("⚠️ Thunderstorm Likely")
        else:
            st.success("✅ No Thunderstorm Expected")
    else:
        st.error(response.text)
