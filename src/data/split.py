"""
Stratified train / validation / test splitting with CSV persistence.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import RANDOM_SEED, VAL_SIZE, TEST_SIZE, PROCESSED_DATA_DIR, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    save: bool = True,
) -> tuple:
    """
    Stratified split into train (70%), validation (15%), test (15%).

    Args:
        df: Preprocessed DataFrame.
        target_col: Name of the target column.
        save: If True, persists CSVs to data/processed/.

    Returns:
        (train_df, val_df, test_df)
    """
    # First split → train 70% | temp 30%
    train_df, temp_df = train_test_split(
        df,
        test_size=VAL_SIZE + TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df[target_col],
    )

    # Second split → val 50% of temp | test 50% of temp
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_df[target_col],
    )

    logger.info(
        f"Split sizes → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
    )
    logger.info(
        f"Fraud rates → Train: {train_df[target_col].mean():.2%} | "
        f"Val: {val_df[target_col].mean():.2%} | Test: {test_df[target_col].mean():.2%}"
    )

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
        val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
        logger.info(f"Saved splits to {PROCESSED_DATA_DIR}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    from src.data.dataset import load_hf_data
    from src.data.preprocess import preprocess_dataframe

    df = load_hf_data()
    df = preprocess_dataframe(df)
    train_df, val_df, test_df = split_data(df)
