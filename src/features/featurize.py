import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class TextMetaFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts meta-features from text: word count, char count, etc."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Expecting X to be a Series of text
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
            
        features = pd.DataFrame()
        features['word_count'] = X.apply(lambda x: len(str(x).split()))
        features['char_count'] = X.apply(len)
        features['avg_word_len'] = features['char_count'] / (features['word_count'] + 1)
        features['upper_case_ratio'] = X.apply(lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1))
        
        return features

def get_feature_pipeline():
    """Builds a scikit-learn pipeline for feature extraction."""
    
    # Text pipeline
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)))
    ])
    
    # Text Meta features
    meta_text_pipeline = Pipeline([
        ('meta_text', TextMetaFeatureExtractor())
    ])
    
    # Categorical/Metadata pipeline
    # categorical_features = ['employment_type', 'required_experience', 'required_education']
    # numeric_boolean_features = ['telecommuting', 'has_company_logo', 'has_questions']
    
    # Since we need to define columns in the ColumnTransformer, we'll return a function/object that takes the dataframe 
    # and splits it, or we can use ColumnTransformer if we know the column names.
    
    # Let's assume the input to the main pipeline is the whole dataframe.
    # We need a custom transformer to select columns.
    
    return text_pipeline # For now, let's just return the text pipeline or a composed one in the training script
    
def build_transformer(categorical_cols, numerical_cols, text_col='clean_text'):
    """
    Constructs a ColumnTransformer for all features.
    """
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_tfidf', TfidfVectorizer(max_features=5000, stop_words='english'), text_col),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', SimpleImputer(strategy='constant', fill_value=0), numerical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor

if __name__ == "__main__":
    pass
