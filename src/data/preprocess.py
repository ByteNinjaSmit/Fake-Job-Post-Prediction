"""
Text cleaning and data preprocessing pipeline.
Handles HTML removal, emoji stripping, stopword removal, and missing values.
"""
import re
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import TEXT_COLS, BOOLEAN_COLS, CATEGORICAL_COLS, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Download NLTK data silently
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

STOP_WORDS = set(nltk.corpus.stopwords.words("english"))

# ─── Regex patterns ──────────────────────────────────────────────
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def clean_text(text: str) -> str:
    """
    Cleans a single text string:
      1. Strip HTML tags
      2. Remove URLs
      3. Remove emails
      4. Remove emojis
      5. Remove non-alphabetic chars
      6. Lowercase
      7. Remove stopwords
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # HTML
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    # URLs & emails
    text = URL_PATTERN.sub("", text)
    text = EMAIL_PATTERN.sub("", text)
    # Emojis
    text = EMOJI_PATTERN.sub("", text)
    # Non-alpha
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Lowercase + stopword removal
    tokens = [w for w in text.lower().split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for the raw DataFrame.

    Steps:
      1. Fill missing text with empty string
      2. Fill missing booleans with 0
      3. Fill missing categoricals with 'Unknown'
      4. Combine text columns → 'text'
      5. Clean combined text → 'clean_text'
      6. Create fraud-indicator features (email count, url count, etc.)
    """
    df = df.copy()
    logger.info("Starting preprocessing...")

    # ── Fill missing values ──
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("")

    for col in BOOLEAN_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # ── Combine text ──
    text_parts = [df[c] for c in TEXT_COLS if c in df.columns]
    df["text"] = text_parts[0]
    for part in text_parts[1:]:
        df["text"] = df["text"] + " " + part

    # ── Clean text ──
    logger.info("Cleaning text (this may take a minute)...")
    df["clean_text"] = df["text"].apply(clean_text)

    # ── Fraud indicator features (from raw text before cleaning) ──
    df["email_count"] = df["text"].apply(lambda x: len(EMAIL_PATTERN.findall(x)))
    df["url_count"] = df["text"].apply(lambda x: len(URL_PATTERN.findall(x)))
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    df["char_count"] = df["clean_text"].apply(len)
    df["upper_ratio"] = df["text"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )
    df["exclamation_count"] = df["text"].apply(lambda x: x.count("!"))
    df["company_profile_len"] = df["company_profile"].apply(len) if "company_profile" in df.columns else 0

    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    from src.data.dataset import load_hf_data

    df = load_hf_data()
    df = preprocess_dataframe(df)
    print(df[["clean_text", "word_count", "email_count", "fraudulent"]].head())
