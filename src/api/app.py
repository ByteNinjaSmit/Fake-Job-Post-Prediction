from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os
from src.api.schemas import JobPostingRequest, PredictionResponse
from src.data.preprocess import clean_text
from src.config import MODELS_DIR

app = FastAPI(
    title="Fake Job Inspector API",
    description="API to detect fraudulent job postings.",
    version="1.0.0"
)

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = MODELS_DIR / "baseline.joblib" # Default to baseline
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("Warning: No model found. Predictions will fail.")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(job: JobPostingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Preprocess input
    # Combine text fields similarly to training
    full_text = f"{job.title} {job.company_profile or ''} {job.description} {job.requirements or ''}"
    cleaned_text = clean_text(full_text)
    
    # Create DataFrame or Series for the pipeline
    # The pipeline expects a Series or iterable of text if using TfidfVectorizer directly on text column
    # BUT, our training script used `X_train = train_df['clean_text']` which is a Series.
    # So we should pass an iterable containing the text.
    
    try:
        prediction = model.predict([cleaned_text])[0]
        # prediction is 0 (Real) or 1 (Fake)
        
        # Get probability if supported
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([cleaned_text])[0]
            fraud_prob = probs[1]
        else:
            fraud_prob = float(prediction)
            
        label = "Fraudulent" if prediction == 1 else "Real"
        
        return PredictionResponse(
            prediction=label,
            probability=1 - fraud_prob if label == "Real" else fraud_prob,
            fraudulent_score=fraud_prob
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain(job: JobPostingRequest):
    return {"message": "Explainability not yet implemented."}
