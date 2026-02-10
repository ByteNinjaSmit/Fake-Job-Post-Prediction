"""
Baseline model — DummyClassifier (majority-class sanity check).
"""
from sklearn.dummy import DummyClassifier


def get_baseline_model():
    """Returns a DummyClassifier that predicts the most frequent class."""
    return DummyClassifier(strategy="most_frequent")
