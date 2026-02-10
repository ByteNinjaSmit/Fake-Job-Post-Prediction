"""
Inference module — single and batch predictions.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocess import clean_text
from src.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """Loads a saved ML pipeline and performs predictions."""

    def __init__(self, model_name: str = "logistic_regression"):
        model_path = MODELS_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No saved model at {model_path}. Train first.")

        bundle = joblib.load(model_path)
        self.transformer = bundle["transformer"]
        self.model = bundle["model"]
        self.model_name = model_name
        logger.info(f"Loaded model: {model_name}")

    def predict_single(self, text: str) -> dict:
        """Predict a single text string."""
        cleaned = clean_text(text)
        X = self.transformer.transform([cleaned])
        pred = int(self.model.predict(X)[0])

        result = {
            "prediction": "Fraudulent" if pred == 1 else "Real",
            "label": pred,
        }

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            result["probability_real"] = float(probs[0])
            result["probability_fraudulent"] = float(probs[1])
        elif hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function(X)[0])
            result["decision_score"] = score

        return result

    def predict_batch(self, texts: list) -> list:
        """Predict a batch of text strings."""
        cleaned = [clean_text(t) for t in texts]
        X = self.transformer.transform(cleaned)
        preds = self.model.predict(X)

        results = []
        for i, pred in enumerate(preds):
            r = {
                "text_preview": texts[i][:80] + "...",
                "prediction": "Fraudulent" if pred == 1 else "Real",
                "label": int(pred),
            }
            results.append(r)

        return results


if __name__ == "__main__":
    predictor = Predictor("logistic_regression")
    result = predictor.predict_single(
        "Earn $5000/week from home! No experience needed. Contact us on WhatsApp."
    )
    print(result)
