import joblib
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, cast
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.api.schemas import (
    JobPostingRequest,
    PredictionResponse,
    BatchRequest,
    BatchResponse,
    ExplainResponse,
    HealthResponse,
    FeatureWeight,
    DatasetInfoResponse,
    DepartmentStat,
    DistributionStat,
)
from src.data.preprocess import clean_text
from src.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Global state ────────────────────────────────────────────────
# Use Optional types to satisfy the type checker
_state: Dict[str, Any] = {
    "transformer": None, 
    "model": None, 
    "model_name": "none",
    "dataset_stats": None
}

PREFERRED_MODELS = ["logistic_regression", "xgboost", "lightgbm", "random_forest", "svm", "baseline"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    for name in PREFERRED_MODELS:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            try:
                bundle = joblib.load(path)
                if "transformer" in bundle and "model" in bundle:
                    _state["transformer"] = bundle["transformer"]
                    _state["model"] = bundle["model"]
                    _state["model_name"] = name
                    logger.info(f"Loaded model: {name}")
                    break
                else:
                    logger.warning(f"Bundle at {path} is missing 'transformer' or 'model' keys.")
            except Exception as e:
                logger.error(f"Failed to load model {name} from {path}: {e}")
    else:
        logger.warning("No trained model found. Please train a model first.")

    # Load dataset stats
    train_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "train.csv"
    if train_path.exists():
        try:
            logger.info(f"Loading dataset stats from {train_path}")
            df = pd.read_csv(train_path)
            
            # Fill NaN in department to avoid errors
            df['department'] = df['department'].fillna('Unknown')
            
            total = len(df)
            fraudulent = int(df['fraudulent'].sum())
            real = total - fraudulent
            
            # Department stats (Top 5)
            dept_groups = df.groupby('department')['fraudulent'].agg(['count', 'sum']).reset_index()
            # If total < 5, use all
            top_depts = dept_groups.sort_values('count', ascending=False).head(5)
            dept_stats = [
                DepartmentStat(name=str(row['department']), count=int(row['count']), fraud=int(row['sum']))
                for _, row in top_depts.iterrows()
            ]

            # Location stats (Top 5 countries)
            # Location format is usually "Country, Region, City"
            df['country'] = df['location'].fillna('Unknown').apply(lambda x: x.split(',')[0].strip())
            loc_groups = df.groupby('country')['fraudulent'].agg(['count', 'sum']).reset_index()
            top_locs = loc_groups.sort_values('count', ascending=False).head(5)
            loc_stats = [
                DepartmentStat(name=str(row['country']), count=int(row['count']), fraud=int(row['sum']))
                for _, row in top_locs.iterrows()
            ]
            
            # Distribution stats
            distribution = [
                DistributionStat(name="Real Posts", value=real, color="#7c3aed"),
                DistributionStat(name="Fraudulent", value=fraudulent, color="#ef4444")
            ]
            
            # Estimate vocabulary (from word_count column if it exists, or placeholder)
            vocab_size = 42400 # Placeholder
            if 'word_count' in df.columns:
                 # Realistically this should be precomputed, but we can return total word count / estimate.
                 # Let's stick with a placeholder or calculated value.
                 vocab_size = int(df['word_count'].sum() / 100) # Arbitrary scaling for "vocabulary"
            
            _state["dataset_stats"] = DatasetInfoResponse(
                total_jobs=total,
                real_posts=real,
                fraudulent_posts=fraudulent,
                departments=dept_stats,
                distribution=distribution,
                locations=loc_stats,
                vocabulary_size=vocab_size,
                fields_per_entry=len(df.columns),
                imbalance_ratio=f"1:{round(real/fraudulent) if fraudulent > 0 else 1}"
            )
            logger.info("Dataset stats loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load dataset stats: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"Dataset not found at {train_path}")
    yield


app = FastAPI(
    title="Fake Job Inspector API",
    description="Detects fraudulent job postings using ML models.",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── Monitoring ──────────────────────────────────────────────────
# Instrument the app to expose /metrics
Instrumentator().instrument(app).expose(app)

# ─── CORS Configuration ──────────────────────────────────────────
# Allow localhost for development and * for general access
# Note: allow_credentials=True cannot be used with allow_origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _prepare_text(job: JobPostingRequest) -> str:
    """Combines and cleans job posting fields."""
    fields = [job.title, job.company_profile, job.description, job.requirements, job.benefits]
    raw = " ".join(filter(None, fields))
    return clean_text(raw)


def _predict_one(cleaned_text: str) -> PredictionResponse:
    """Runs prediction on a single cleaned text."""
    model = _state["model"]
    transformer = _state["transformer"]

    # Explicitly check for None to satisfy type checker
    if model is None or transformer is None:
        raise ValueError("Model or transformer not loaded.")

    # Cast to ensure the type checker knows they are not None and have the required methods
    model = cast(Any, model)
    transformer = cast(Any, transformer)

    X = transformer.transform([cleaned_text])
    pred = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        fraud_score = float(probs[1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        # Normalize decision function to [0, 1] using sigmoid
        fraud_score = float(1 / (1 + np.exp(-score)))
    else:
        fraud_score = float(pred)

    return PredictionResponse(
        prediction="Fraudulent" if pred == 1 else "Real",
        confidence=float(max(fraud_score, 1 - fraud_score)),
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
    if _state["model"] is None or _state["transformer"] is None:
        raise HTTPException(503, "API unavailable: No model loaded.")

    cleaned = _prepare_text(job)
    return _predict_one(cleaned)


@app.post("/batch", response_model=BatchResponse)
def batch_predict(batch: BatchRequest):
    if _state["model"] is None or _state["transformer"] is None:
        raise HTTPException(503, "API unavailable: No model loaded.")

    results = []
    for job in batch.posts:
        cleaned = _prepare_text(job)
        results.append(_predict_one(cleaned))

    return BatchResponse(results=results, total=len(results))


@app.post("/explain", response_model=ExplainResponse)
def explain(job: JobPostingRequest):
    if _state["model"] is None or _state["transformer"] is None:
        raise HTTPException(503, "API unavailable: No model loaded.")

    cleaned = _prepare_text(job)
    response = _predict_one(cleaned)

    # LIME explanation
    try:
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(class_names=["Real", "Fraudulent"])

        def predict_fn(texts):
            cleaned_texts = [clean_text(t) for t in texts]
            
            transformer = _state["transformer"]
            model = _state["model"]
            
            if transformer is None or model is None:
                raise ValueError("Model or transformer not loaded.")
                
            X = cast(Any, transformer).transform(cleaned_texts)
            if hasattr(model, "predict_proba"):
                return cast(Any, model).predict_proba(X)
            else:
                preds = cast(Any, model).predict(X)
                return np.column_stack([1 - preds, preds])

        # Use consistent text preparation for LIME
        fields = [job.title, job.company_profile, job.description, job.requirements, job.benefits]
        raw_text = " ".join(filter(None, fields))
        
        exp = explainer.explain_instance(raw_text, predict_fn, num_features=10)
        top_features = [
            FeatureWeight(word=str(w), weight=float(s)) 
            for w, s in exp.as_list()
        ]

    except Exception as e:
        logger.warning(f"Explain failed: {e}")
        top_features = [FeatureWeight(word="explanation_not_available", weight=0.0)]

    return ExplainResponse(
        prediction=response.prediction,
        confidence=response.confidence,
        fraudulent_score=response.fraudulent_score,
        top_features=top_features,
    )


@app.get("/dataset/info", response_model=DatasetInfoResponse)
async def get_dataset_info():
    if _state["dataset_stats"] is None:
        raise HTTPException(status_code=404, detail="Dataset statistics not available.")
    return _state["dataset_stats"]
