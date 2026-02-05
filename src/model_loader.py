import mlflow
import streamlit as st
from src.config import MODEL_REGISTRY


@st.cache_resource
def load_model(model_name: str):
    model_info = MODEL_REGISTRY[model_name]
    model_path = model_info["path"]

    if model_info["type"] == "xgboost":
        return mlflow.xgboost.load_model(model_path.as_posix())
    else:
        return mlflow.sklearn.load_model(model_path.as_posix())
