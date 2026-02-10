"""
Configuration module for Fake Job Post Prediction.
Centralizes all hyperparameters, paths, and constants.
"""
import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Dataset ─────────────────────────────────────────────────────
HF_DATASET_NAME = "victor/real-or-fake-fake-jobposting-prediction"
TARGET_COL = "fraudulent"

# ─── Text Columns ────────────────────────────────────────────────
TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
CATEGORICAL_COLS = ["employment_type", "required_experience", "required_education", "industry", "function"]
BOOLEAN_COLS = ["telecommuting", "has_company_logo", "has_questions"]

# ─── Splitting ────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# ─── TF-IDF ──────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# ─── BERT / Transformer ──────────────────────────────────────────
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LENGTH = 256
BERT_BATCH_SIZE = 16
BERT_LEARNING_RATE = 2e-5
BERT_EPOCHS = 4
BERT_WARMUP_STEPS = 100

# ─── Class Imbalance ─────────────────────────────────────────────
SMOTE_SAMPLING_STRATEGY = "auto"  # minority oversampling
