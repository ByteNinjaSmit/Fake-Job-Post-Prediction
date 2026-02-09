from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_logistic_regression():
    return LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

def get_linear_svm():
    return LinearSVC(class_weight='balanced', random_state=42)

def get_random_forest():
    return RandomForestClassifier(class_weight='balanced', random_state=42)

def get_xgboost():
    return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

def get_model(model_name):
    """Factory function to get a model by name."""
    if model_name == 'logistic_regression':
        return get_logistic_regression()
    elif model_name == 'svm':
        return get_linear_svm()
    elif model_name == 'random_forest':
        return get_random_forest()
    elif model_name == 'xgboost':
        return get_xgboost()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
