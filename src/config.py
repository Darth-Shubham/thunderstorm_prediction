from pathlib import Path

MODEL_DIR = Path("models")

MODEL_REGISTRY = {
    "RandomForest": {
        "path": MODEL_DIR / "randomforest" / "best_model",
        "type": "sklearn"
    },
    "XGBoost": {
        "path": MODEL_DIR / "xgboost" / "best_model",
        "type": "xgboost"
    },
    "LogisticRegression": {
        "path": MODEL_DIR / "logisticregression" / "best_model",
        "type": "sklearn"
    }
}

THRESHOLD = 0.6
