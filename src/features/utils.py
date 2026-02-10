"""
Feature utility functions.
"""
import pandas as pd


def get_available_columns(df: pd.DataFrame, desired_cols: list) -> list:
    """Returns the subset of desired_cols that actually exist in df."""
    return [c for c in desired_cols if c in df.columns]
