import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models" # This wasn't in original structure but useful for saving models

# Parameters
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# HF Dataset
HF_DATASET_NAME = "victor/real-or-fake-fake-jobposting-prediction"
