"""
Classical ML model factory — Logistic Regression, SVM, Random Forest, XGBoost, LightGBM.
All models use class_weight='balanced' for imbalanced fraud detection.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import RANDOM_SEED

# ─── Model Registry ──────────────────────────────────────────────
MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED, C=1.0
    ),
    "svm": lambda: LinearSVC(
        class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
    ),
    "xgboost": lambda: XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=10,  # approximate ratio for imbalance
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
    "lightgbm": lambda: LGBMClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    ),
}


def get_model(model_name: str):
    """
    Factory function — returns a fresh model instance.

    Args:
        model_name: One of 'logistic_regression', 'svm', 'random_forest',
                    'xgboost', 'lightgbm'.

    Raises:
        ValueError if model_name is unknown.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]()


def list_models() -> list:
    """Returns list of available model names."""
    return list(MODEL_REGISTRY.keys())
