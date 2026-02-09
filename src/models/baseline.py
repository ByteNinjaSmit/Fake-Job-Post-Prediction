from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
# from src.features.featurize import get_feature_pipeline # If needed later

def get_baseline_model():
    """Returns a Dummy Classifier pipeline."""
    # Simple baseline that predicts the most frequent class
    clf = DummyClassifier(strategy="most_frequent")
    return clf
