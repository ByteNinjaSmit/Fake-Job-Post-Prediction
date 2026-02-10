"""
Main training script — runs end-to-end pipeline for classical ML models.
Supports: baseline, logistic_regression, svm, random_forest, xgboost, lightgbm.

Usage:
    python src/training/train.py --model logistic_regression
    python src/training/train.py --model xgboost --smote
    python src/training/train.py --all
"""
import argparse
import time
import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import load_hf_data
from src.data.preprocess import preprocess_dataframe
from src.data.split import split_data
from src.data.augment import apply_smote
from src.features.featurize import build_tfidf_only, build_feature_transformer
from src.features.utils import get_available_columns
from src.models.baseline import get_baseline_model
from src.models.ml_models import get_model, list_models
from src.utils.metrics import compute_all_metrics, print_report
from src.utils.logger import get_logger
from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TARGET_COL,
    CATEGORICAL_COLS,
    BOOLEAN_COLS,
)

logger = get_logger(__name__)


def load_or_prepare_data():
    """Loads cached processed splits or runs the full preprocessing pipeline."""
    train_path = PROCESSED_DATA_DIR / "train.csv"

    if train_path.exists():
        logger.info("Loading cached processed splits...")
        train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
        val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
        test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    else:
        logger.info("No cached data found. Running full pipeline...")
        df = load_hf_data()
        df = preprocess_dataframe(df)
        train_df, val_df, test_df = split_data(df)

    return train_df, val_df, test_df


def train_single_model(model_name: str, use_smote: bool = False, use_full_features: bool = False):
    """Trains a single model and evaluates on validation + test sets."""
    start = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"  Training: {model_name}")
    logger.info(f"{'='*60}")

    # 1. Load data
    train_df, val_df, test_df = load_or_prepare_data()

    # 2. Choose features
    if use_full_features and model_name not in ("baseline",):
        # Use ColumnTransformer with TF-IDF + metadata + engineered features
        cat_cols = get_available_columns(train_df, CATEGORICAL_COLS)
        num_cols = get_available_columns(
            train_df,
            BOOLEAN_COLS + [
                "email_count", "url_count", "word_count",
                "char_count", "upper_ratio", "exclamation_count", "company_profile_len",
            ],
        )

        transformer = build_feature_transformer(
            text_col="clean_text", cat_cols=cat_cols, num_cols=num_cols
        )

        X_train = transformer.fit_transform(train_df)
        X_val = transformer.transform(val_df)
        X_test = transformer.transform(test_df)
    else:
        # TF-IDF only
        tfidf = build_tfidf_only()
        X_train = tfidf.fit_transform(train_df["clean_text"].fillna(""))
        X_val = tfidf.transform(val_df["clean_text"].fillna(""))
        X_test = tfidf.transform(test_df["clean_text"].fillna(""))

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    # 3. Optional SMOTE
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    # 4. Get model
    if model_name == "baseline":
        clf = get_baseline_model()
    else:
        clf = get_model(model_name)

    # 5. Train
    logger.info("Fitting model...")
    clf.fit(X_train, y_train)

    # 6. Evaluate
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    # Probabilities (if supported)
    y_val_proba = None
    y_test_proba = None
    if hasattr(clf, "predict_proba"):
        y_val_proba = clf.predict_proba(X_val)[:, 1]
        y_test_proba = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_val_proba = clf.decision_function(X_val)
        y_test_proba = clf.decision_function(X_test)

    val_metrics = compute_all_metrics(y_val, y_val_pred, y_val_proba, f"{model_name} (val)")
    test_metrics = compute_all_metrics(y_test, y_test_pred, y_test_proba, f"{model_name} (test)")

    print_report(y_val, y_val_pred, f"{model_name} (Validation)")
    print_report(y_test, y_test_pred, f"{model_name} (Test)")

    # 7. Save model
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if use_full_features:
        # Save the transformer + model together
        full_pipeline = {"transformer": transformer if use_full_features else tfidf, "model": clf}
        joblib.dump(full_pipeline, model_path)
    else:
        full_pipeline = {"transformer": tfidf, "model": clf}
        joblib.dump(full_pipeline, model_path)

    elapsed = time.time() - start
    logger.info(f"Model saved → {model_path}  (took {elapsed:.1f}s)")

    return val_metrics, test_metrics


def train_all_models(use_smote: bool = False):
    """Trains all available models and prints comparison table."""
    results = []

    # Baseline
    _, test_m = train_single_model("baseline")
    results.append(test_m)

    # All ML models
    for name in list_models():
        _, test_m = train_single_model(name, use_smote=use_smote)
        results.append(test_m)

    # Comparison table
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON (Test Set)")
    print("=" * 80)
    comparison = pd.DataFrame(results)
    print(comparison.to_string(index=False))
    comparison.to_csv(MODELS_DIR / "comparison.csv", index=False)
    logger.info(f"Comparison saved → {MODELS_DIR / 'comparison.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fake Job Post classifiers")
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        help=f"Model name: baseline, {', '.join(list_models())}",
    )
    parser.add_argument("--all", action="store_true", help="Train all models")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE oversampling")
    parser.add_argument("--full-features", action="store_true", help="Use TF-IDF + metadata features")
    args = parser.parse_args()

    if args.all:
        train_all_models(use_smote=args.smote)
    else:
        train_single_model(args.model, use_smote=args.smote, use_full_features=args.full_features)
