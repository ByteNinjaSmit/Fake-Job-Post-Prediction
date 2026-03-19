"""
Dataset loader — fetches data from Hugging Face and caches locally.
"""
import pandas as pd
from datasets import load_dataset
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import HF_DATASET_NAME, RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_hf_data(force_download: bool = False) -> pd.DataFrame:
    """
    Loads the Fake Job Posting dataset from Hugging Face.

    Returns a single pandas DataFrame containing all records.
    Caches to disk for faster subsequent loads.
    """
    cache_path = RAW_DATA_DIR / "huggingface_dataset" / "full_dataset.csv"

    if cache_path.exists() and not force_download:
        logger.info(f"Loading cached dataset from {cache_path}")
        return pd.read_csv(cache_path)

    logger.info(f"Downloading dataset: {HF_DATASET_NAME}")
    ds = load_dataset(HF_DATASET_NAME)

    # Merge all splits into a single DataFrame
    frames = []
    for split_name in ds:
        df_split = pd.DataFrame(ds[split_name])
        df_split["_split"] = split_name
        frames.append(df_split)

    df = pd.concat(frames, ignore_index=True)

    # Cache locally
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info(f"Dataset cached → {cache_path}  ({len(df)} rows)")

    return df


def inspect_dataset(df: pd.DataFrame):
    """Prints key statistics about the dataset."""
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"\nTarget distribution:\n{df['fraudulent'].value_counts()}")
    logger.info(f"\nMissing values:\n{df.isnull().sum()}")
    logger.info(f"\nFraud rate: {df['fraudulent'].mean():.2%}")


if __name__ == "__main__":
    df = load_hf_data()
    inspect_dataset(df)
