from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculates evaluation metrics.
    """
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

def print_evaluation_report(y_true, y_pred, model_name="Model"):
    """
    Prints a detailed evaluation report.
    """
    print(f"\n--- Evaluation Report for {model_name} ---")
    metrics = calculate_metrics(y_true, y_pred, model_name)
    for k, v in metrics.items():
        if k != "Model":
            print(f"{k}: {v:.4f}")
            
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
