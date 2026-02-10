"""
Feature engineering — TF-IDF, metadata encoding, and custom fraud indicators.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    CATEGORICAL_COLS,
    BOOLEAN_COLS,
)


# ─── Engineered numeric features created during preprocessing ────
ENGINEERED_NUM_COLS = [
    "email_count",
    "url_count",
    "word_count",
    "char_count",
    "upper_ratio",
    "exclamation_count",
    "company_profile_len",
]


def build_feature_transformer(
    text_col: str = "clean_text",
    cat_cols: list = None,
    num_cols: list = None,
):
    """
    Builds a ColumnTransformer that handles:
      • TF-IDF on clean_text
      • One-hot encoding on categorical columns
      • Pass-through for boolean + engineered numeric columns

    Returns:
        sklearn ColumnTransformer
    """
    if cat_cols is None:
        cat_cols = CATEGORICAL_COLS
    if num_cols is None:
        num_cols = BOOLEAN_COLS + ENGINEERED_NUM_COLS

    transformers = [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=TFIDF_NGRAM_RANGE,
                stop_words="english",
                sublinear_tf=True,
            ),
            text_col,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            cat_cols,
        ),
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ]
            ),
            num_cols,
        ),
    ]

    # Filter out columns that may not exist in the DataFrame
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_tfidf_only(text_col: str = "clean_text"):
    """Returns a simple TF-IDF-only feature extractor (for baseline pipelines)."""
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
        sublinear_tf=True,
    )
