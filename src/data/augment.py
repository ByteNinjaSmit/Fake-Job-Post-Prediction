"""
Optional data augmentation strategies for handling class imbalance.
Includes SMOTE oversampling for tabular features.
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import RANDOM_SEED, SMOTE_SAMPLING_STRATEGY
from src.utils.logger import get_logger

logger = get_logger(__name__)


def apply_smote(X, y):
    """
    Applies SMOTE oversampling to balance classes.

    Args:
        X: Feature matrix (dense numpy array or sparse matrix).
        y: Target vector.

    Returns:
        (X_resampled, y_resampled)
    """
    logger.info(f"Before SMOTE → Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    smote = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=RANDOM_SEED)
    X_res, y_res = smote.fit_resample(X, y)

    logger.info(f"After SMOTE  → Class distribution: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res
