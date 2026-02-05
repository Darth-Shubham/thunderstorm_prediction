import pandas as pd
import numpy as np
from src.config import THRESHOLD


def predict_with_threshold(model, input_df: pd.DataFrame):
    probs = model.predict_proba(input_df)[:, 1]
    pred = (probs >= THRESHOLD).astype(int)

    return {
        "probability": float(probs[0]),
        "prediction": int(pred[0])
    }
