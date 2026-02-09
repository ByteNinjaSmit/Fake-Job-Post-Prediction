import argparse
import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import load_hf_data
from src.data.preprocess import preprocess_dataframe
from src.data.split import split_data
from src.models.baseline import get_baseline_model
from src.models.ml_models import get_model
from src.training.evaluate import print_evaluation_report
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def train_ml_model(model_name):
    print(f"Starting training for {model_name}...")
    
    # 1. Load Data
    print("Loading data...")
    ds = load_hf_data()
    df = pd.DataFrame(ds['train']) # The dataset might already have splits, but prompt said load from HF then split 70/15/15. 
    # Let's check the dataset structure. If it has splits, we might merge or just use train.
    # The prompt user request said: "Split data: train / val / test = 70% / 15% / 15%"
    # So I will assume I need to do the split myself from the full dataset.
    
    # 2. Preprocess
    print("Preprocessing data...")
    df = preprocess_dataframe(df)
    
    # 3. Split
    print("Splitting data...")
    train_df, val_df, test_df = split_data(df, target_col='fraudulent')
    
    X_train = train_df['clean_text']
    y_train = train_df['fraudulent']
    X_val = val_df['clean_text']
    y_val = val_df['fraudulent']
    
    # 4. Build Pipeline
    print("Building pipeline...")
    # Using TF-IDF as the main feature for now as per "Text Classical" approach
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    
    if model_name == 'baseline':
        clf = get_baseline_model()
    else:
        clf = get_model(model_name)
        
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', clf)
    ])
    
    # 5. Train
    print(f"Training {model_name}...")
    pipeline.fit(X_train, y_train)
    
    # 6. Evaluate
    print("Evaluating on Validation Set...")
    y_pred = pipeline.predict(X_val)
    print_evaluation_report(y_val, y_pred, model_name)
    
    # 7. Save Model
    model_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

def train_bert_model():
    print("BERT training not fully implemented in this script yet. Use a separate script or notebook for Deep Learning training loop.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", help="Model to train: baseline, logistic_regression, svm, random_forest, xgboost")
    args = parser.parse_args()
    
    train_ml_model(args.model)
