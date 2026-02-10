"""
Visualization utilities for EDA and model evaluation plots.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Style defaults
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12})


def plot_class_distribution(y, title="Class Distribution", save_path=None):
    """Bar plot of target class distribution."""
    fig, ax = plt.subplots()
    labels, counts = np.unique(y, return_counts=True)
    names = ["Real" if l == 0 else "Fraudulent" for l in labels]
    colors = ["#2ecc71", "#e74c3c"]
    ax.bar(names, counts, color=colors, edgecolor="black", alpha=0.85)

    for i, (name, count) in enumerate(zip(names, counts)):
        ax.text(i, count + 50, str(count), ha="center", fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot → {save_path}")

    return fig


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """Heatmap of confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fraudulent"],
        yticklabels=["Real", "Fraudulent"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curve(y_true, y_proba, model_name="Model", save_path=None):
    """ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_precision_recall_curve(y_true, y_proba, model_name="Model", save_path=None):
    """Precision-Recall curve with AUC."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, color="#e67e22", lw=2, label=f"PR (AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_comparison(results_df: pd.DataFrame, metric="f1_score", save_path=None):
    """Horizontal bar chart comparing models on a metric."""
    fig, ax = plt.subplots()
    results_sorted = results_df.sort_values(metric, ascending=True)
    ax.barh(results_sorted["model"], results_sorted[metric], color="#9b59b6", edgecolor="black", alpha=0.85)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
