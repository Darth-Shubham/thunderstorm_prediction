from fastapi import FastAPI
import pandas as pd

from src.schema import (
    PredictionRequest,
    PredictionResponse,
    FEATURE_COLUMNS
)
from src.model_loader import load_model
from src.prediction import predict_with_threshold

app = FastAPI(title="Thunderstorm Prediction API")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    model_name: str = "RandomForest"
):
    model = load_model(model_name)

    input_dict = payload.dict(by_alias=True)
    input_df = pd.DataFrame([input_dict])[FEATURE_COLUMNS]

    result = predict_with_threshold(model, input_df)
    return result
