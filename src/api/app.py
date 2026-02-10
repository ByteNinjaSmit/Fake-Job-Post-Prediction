"""
FastAPI application for Fake Job Post Prediction.

Endpoints:
    GET  /health     — Health check
    POST /predict    — Classify a single job posting
    POST /batch      — Classify multiple job postings
    POST /explain    — Classify + explain top contributing features
"""
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.schemas import (
    JobPostingRequest,
    PredictionResponse,
    BatchRequest,
    BatchResponse,
    ExplainResponse,
    HealthResponse,
)
from src.data.preprocess import clean_text
from src.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Global state ────────────────────────────────────────────────
_state = {"transformer": None, "model": None, "model_name": "none"}

PREFERRED_MODELS = ["logistic_regression", "xgboost", "lightgbm", "random_forest", "svm", "baseline"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    for name in PREFERRED_MODELS:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            bundle = joblib.load(path)
            _state["transformer"] = bundle["transformer"]
            _state["model"] = bundle["model"]
            _state["model_name"] = name
            logger.info(f"Loaded model: {name}")
            break
    else:
        logger.warning("No trained model found. Train a model first.")
    yield


app = FastAPI(
    title="Fake Job Inspector API",
    description="Detects fraudulent job postings using ML models.",
    version="1.0.0",
    lifespan=lifespan,
)


def _prepare_text(job: JobPostingRequest) -> str:
    """Combines and cleans job posting fields."""
    raw = " ".join(
        filter(None, [job.title, job.company_profile, job.description, job.requirements, job.benefits])
    )
    return clean_text(raw)


def _predict_one(cleaned_text: str) -> PredictionResponse:
    """Runs prediction on a single cleaned text."""
    model = _state["model"]
    transformer = _state["transformer"]

    X = transformer.transform([cleaned_text])
    pred = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        fraud_score = float(probs[1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        # Normalize decision function to [0, 1]
        import numpy as np
        fraud_score = float(1 / (1 + np.exp(-score)))
    else:
        fraud_score = float(pred)

    return PredictionResponse(
        prediction="Fraudulent" if pred == 1 else "Real",
        confidence=max(fraud_score, 1 - fraud_score),
        fraudulent_score=fraud_score,
    )


# ─── Endpoints ───────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=_state["model"] is not None,
        model_name=_state["model_name"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(job: JobPostingRequest):
    if _state["model"] is None:
        raise HTTPException(503, "No model loaded. Train a model first.")

    cleaned = _prepare_text(job)
    return _predict_one(cleaned)


@app.post("/batch", response_model=BatchResponse)
def batch_predict(batch: BatchRequest):
    if _state["model"] is None:
        raise HTTPException(503, "No model loaded.")

    results = []
    for job in batch.posts:
        cleaned = _prepare_text(job)
        results.append(_predict_one(cleaned))

    return BatchResponse(results=results, total=len(results))


@app.post("/explain", response_model=ExplainResponse)
def explain(job: JobPostingRequest):
    if _state["model"] is None:
        raise HTTPException(503, "No model loaded.")

    cleaned = _prepare_text(job)
    response = _predict_one(cleaned)

    # LIME explanation
    try:
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(class_names=["Real", "Fraudulent"])

        def predict_fn(texts):
            import numpy as np
            cleaned_texts = [clean_text(t) for t in texts]
            X = _state["transformer"].transform(cleaned_texts)
            model = _state["model"]
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            else:
                preds = model.predict(X)
                return np.column_stack([1 - preds, preds])

        raw_text = " ".join(filter(None, [job.title, job.company_profile, job.description, job.requirements]))
        exp = explainer.explain_instance(raw_text, predict_fn, num_features=10)
        top_features = [{"word": w, "weight": round(s, 4)} for w, s in exp.as_list()]

    except Exception as e:
        logger.warning(f"Explain failed: {e}")
        top_features = [{"word": "explanation_not_available", "weight": 0.0}]

    return ExplainResponse(
        prediction=response.prediction,
        confidence=response.confidence,
        top_features=top_features,
    )
