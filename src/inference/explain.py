"""
Explainability module — SHAP and LIME explanations for model predictions.
"""
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def explain_with_lime(predictor, text: str, num_features: int = 10):
    """
    Generates a LIME explanation for a single prediction.

    Args:
        predictor: Predictor instance with transformer + model.
        text: Raw text input.
        num_features: Number of top features to show.

    Returns:
        List of (feature, weight) tuples.
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        logger.warning("LIME not installed. Run: pip install lime")
        return []

    explainer = LimeTextExplainer(class_names=["Real", "Fraudulent"])

    def predict_fn(texts):
        from src.data.preprocess import clean_text
        cleaned = [clean_text(t) for t in texts]
        X = predictor.transformer.transform(cleaned)
        if hasattr(predictor.model, "predict_proba"):
            return predictor.model.predict_proba(X)
        else:
            preds = predictor.model.predict(X)
            probs = np.column_stack([1 - preds, preds])
            return probs

    explanation = explainer.explain_instance(text, predict_fn, num_features=num_features)
    return explanation.as_list()


def explain_with_shap(predictor, texts: list, max_display: int = 10):
    """
    Generates SHAP explanations for ML models.

    Args:
        predictor: Predictor instance with transformer + model.
        texts: List of text inputs.
        max_display: Max features to display.

    Returns:
        SHAP values object.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Run: pip install shap")
        return None

    from src.data.preprocess import clean_text

    cleaned = [clean_text(t) for t in texts]
    X = predictor.transformer.transform(cleaned)

    if hasattr(predictor.model, "predict_proba"):
        explainer = shap.Explainer(predictor.model.predict_proba, X)
    else:
        explainer = shap.Explainer(predictor.model.predict, X)

    shap_values = explainer(X)
    logger.info(f"SHAP explanation computed for {len(texts)} samples.")
    return shap_values
