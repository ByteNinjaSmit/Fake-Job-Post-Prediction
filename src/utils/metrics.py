"""
Custom metrics utilities (extends sklearn metrics).
"""
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
import numpy as np


def compute_all_metrics(y_true, y_pred, y_proba=None, model_name: str = "Model") -> dict:
    """
    Computes a comprehensive set of evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for the positive class (optional).
        model_name: Name of the model for display.

    Returns:
        Dictionary of metric names to values.
    """
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")

    return metrics


def print_report(y_true, y_pred, model_name: str = "Model"):
    """Prints classification report and confusion matrix."""
    print(f"\n{'='*60}")
    print(f"  Evaluation Report — {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fraudulent"], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"{'='*60}\n")
