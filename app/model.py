"""Load the trained XGBoost model and score raw customer records."""
from functools import lru_cache
from pathlib import Path

import pandas as pd
import xgboost as xgb

from src import dataset

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgb_model.json"


@lru_cache
def get_model() -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def predict(records: list[dict]) -> list[dict]:
    probs = get_model().predict_proba(dataset.encode(pd.DataFrame(records)))
    out = []
    for p in probs:
        best = int(p.argmax())
        out.append({
            "segment": dataset.CLASSES[best],
            "confidence": round(float(p[best]), 4),
            "probabilities": {c: round(float(v), 4) for c, v in zip(dataset.CLASSES, p)},
        })
    return out
