"""
General helper utilities.
"""
import re
from typing import List


def combine_text_columns(row, columns: List[str], separator: str = " ") -> str:
    """Combines multiple text columns into a single string."""
    parts = [str(row.get(col, "")) if row.get(col) is not None else "" for col in columns]
    return separator.join(parts).strip()


def count_pattern(text: str, pattern: str) -> int:
    """Counts regex pattern occurrences in text."""
    if not isinstance(text, str):
        return 0
    return len(re.findall(pattern, text))


def has_pattern(text: str, pattern: str) -> int:
    """Returns 1 if pattern exists in text, else 0."""
    return 1 if count_pattern(text, pattern) > 0 else 0
