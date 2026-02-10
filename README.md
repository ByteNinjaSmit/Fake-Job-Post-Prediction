# 🔍 Fake Job Post Prediction

> **Industry-ready ML system** for detecting fraudulent job postings using classical ML and Transformer models.  
> Built with Python, scikit-learn, XGBoost, LightGBM, BERT (Transformers), and FastAPI.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📊 Model Performance Results

All 6 models were trained and evaluated on the [HuggingFace Fake Job Posting dataset](https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction) (17,880 records).  
Evaluation was performed on a held-out **15% stratified test set**.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (DummyClassifier) | 95.2% | — | — | — | 0.50 |
| Logistic Regression | 96.5% | 0.59 | **0.89** | 0.71 | **0.99** |
| **Linear SVM** | **98.2%** | 0.80 | 0.83 | **0.82** | 0.98 |
| Random Forest | 97.7% | **0.99** | 0.53 | 0.69 | 0.98 |
| XGBoost | 97.8% | 0.78 | 0.77 | 0.77 | 0.98 |
| LightGBM | 98.1% | 0.86 | 0.74 | 0.79 | 0.98 |

### Key Takeaways

- **Best overall (F1):** Linear SVM — 0.82 F1 with 98.2% accuracy
- **Best recall (catch fraud):** Logistic Regression — 0.89 recall (misses fewest fake posts)
- **Best precision (fewest false alarms):** Random Forest — 0.99 precision
- **Priority metric:** F1 Score and Recall — minimizing missed fraud is critical

---

## 📁 Project Structure

```
Fake-Job-Post-Prediction/
│
├── data/
│   ├── raw/
│   │   └── huggingface_dataset/           # Cached raw dataset from HF
│   ├── processed/
│   │   ├── train.csv                      # 70% stratified train split
│   │   ├── val.csv                        # 15% validation split
│   │   └── test.csv                       # 15% test split
│   └── external/                          # Optional augmentation data
│
├── notebooks/
│   ├── 01_eda.ipynb                       # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_baseline_models.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                          # Centralized hyperparameters & paths
│   │
│   ├── data/
│   │   ├── dataset.py                     # HuggingFace dataset loader + local cache
│   │   ├── preprocess.py                  # HTML/emoji/URL removal, stopwords, fraud indicators
│   │   ├── split.py                       # Stratified train/val/test splitting
│   │   └── augment.py                     # SMOTE oversampling for class imbalance
│   │
│   ├── features/
│   │   ├── featurize.py                   # TF-IDF + metadata ColumnTransformer
│   │   └── utils.py                       # Feature utility functions
│   │
│   ├── models/
│   │   ├── baseline.py                    # DummyClassifier (majority class)
│   │   ├── ml_models.py                   # Model registry: LR, SVM, RF, XGBoost, LightGBM
│   │   └── transformer.py                 # BERT fine-tuning wrapper (train/predict/save/load)
│   │
│   ├── training/
│   │   ├── train.py                       # Main training script (--all, --smote, --full-features)
│   │   ├── evaluate.py                    # Evaluation metrics (Accuracy, F1, ROC-AUC, PR-AUC)
│   │   └── callbacks.py                   # Early stopping callback
│   │
│   ├── inference/
│   │   ├── predict.py                     # Single + batch prediction with saved models
│   │   └── explain.py                     # SHAP & LIME explainability
│   │
│   ├── api/
│   │   ├── app.py                         # FastAPI app (4 endpoints)
│   │   └── schemas.py                     # Pydantic request/response schemas
│   │
│   ├── utils/
│   │   ├── helpers.py                     # Text combination, pattern matching
│   │   ├── metrics.py                     # Comprehensive metric computation
│   │   └── logger.py                      # Centralized logging
│   │
│   └── visualization/
│       └── plots.py                       # Confusion matrix, ROC, PR curves, model comparison
│
├── models/                                # Saved model artifacts (.joblib)
│   ├── baseline.joblib
│   ├── logistic_regression.joblib
│   ├── svm.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── lightgbm.joblib
│   └── comparison.csv                     # Model comparison results
│
├── requirements.txt
├── Dockerfile
├── .gitignore
├── README.md
└── LICENSE
```

---

## 🚀 Quick Start

### 1. Clone & Setup Virtual Environment

```bash
git clone https://github.com/your-username/Fake-Job-Post-Prediction.git
cd Fake-Job-Post-Prediction

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train a single model
.\venv\Scripts\python src/training/train.py --model logistic_regression

# Train ALL models and generate comparison table
.\venv\Scripts\python src/training/train.py --all

# Train with SMOTE oversampling (handles class imbalance)
.\venv\Scripts\python src/training/train.py --model xgboost --smote

# Train with full features (TF-IDF + metadata + engineered features)
.\venv\Scripts\python src/training/train.py --model xgboost --full-features
```

**Available models:** `baseline`, `logistic_regression`, `svm`, `random_forest`, `xgboost`, `lightgbm`

### 3. Run Inference

```python
from src.inference.predict import Predictor

predictor = Predictor("logistic_regression")

result = predictor.predict_single(
    "Earn $5000/week from home! No experience needed. Contact us on WhatsApp."
)
print(result)
# {'prediction': 'Fraudulent', 'label': 1, 'probability_fraudulent': 0.92, ...}
```

### 4. Start the API

```bash
uvicorn src.api.app:app --reload
```

Then visit: [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger documentation.

---

## 🌐 API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/health` | GET | Health check — model status |
| `/predict` | POST | Classify a single job posting |
| `/batch` | POST | Classify multiple job postings |
| `/explain` | POST | Classify + LIME feature explanation |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Marketing Intern",
    "description": "Earn money fast from home!",
    "company_profile": "",
    "requirements": "No experience needed"
  }'
```

### Example Response

```json
{
  "prediction": "Fraudulent",
  "confidence": 0.92,
  "fraudulent_score": 0.92
}
```

---

## 🧠 Models & Methodology

### Tier 1 — Classical ML (TF-IDF Features)

| Model | Library | Strategy |
|-------|---------|----------|
| Logistic Regression | scikit-learn | `class_weight='balanced'`, max_iter=1000 |
| Linear SVM | scikit-learn | `class_weight='balanced'` |

### Tier 2 — Ensemble Models (TF-IDF + Metadata)

| Model | Library | Strategy |
|-------|---------|----------|
| Random Forest | scikit-learn | 200 estimators, `class_weight='balanced'` |
| XGBoost | XGBoost | 200 estimators, `scale_pos_weight=10` |
| LightGBM | LightGBM | 200 estimators, `class_weight='balanced'` |

### Tier 3 — Deep Learning

| Model | Library | Strategy |
|-------|---------|----------|
| BERT | Hugging Face Transformers | `bert-base-uncased`, lr=2e-5, 4 epochs, AdamW |

### Class Imbalance Handling

The dataset is highly imbalanced (~5% fraud). We address this through:
- **Class weights** — `balanced` weighting in all classical models
- **Scale pos weight** — XGBoost positive class weighting
- **SMOTE** — Synthetic minority oversampling (optional via `--smote` flag)

---

## 🔧 Feature Engineering

### Text Features
- **TF-IDF vectors** — up to 5,000 features, bigrams, sublinear TF
- Combined text from: `title + company_profile + description + requirements + benefits`

### Engineered Fraud Indicators

| Feature | Rationale |
|---------|-----------|
| `email_count` | Fake posts often include personal emails |
| `url_count` | External link redirection |
| `exclamation_count` | Emotional manipulation ("Earn $$$!!!") |
| `upper_ratio` | ALL CAPS usage |
| `word_count` | Unusually short or long descriptions |
| `company_profile_len` | Fake companies have short/empty profiles |

### Metadata Features (One-Hot Encoded)
- `employment_type`, `required_experience`, `required_education`, `industry`, `function`

### Boolean Features
- `telecommuting`, `has_company_logo`, `has_questions`

---

## 📈 Evaluation Metrics

| Metric | Description | Priority |
|--------|-------------|----------|
| **F1 Score** | Harmonic mean of precision & recall | ⭐ Primary |
| **Recall** | Fraction of actual fraud detected | ⭐ Primary |
| **Precision** | Fraction of predicted fraud that is real | Secondary |
| **ROC-AUC** | Overall discrimination ability | Secondary |
| **PR-AUC** | Precision-Recall area under curve | Secondary |
| **Accuracy** | Overall correctness | Baseline |

> **Priority: F1 and Recall** — In fraud detection, missing a fake job post (false negative) is worse than a false alarm.

---

## 🧪 Data Pipeline

```
HuggingFace Dataset (17,880 records)
        ↓
   Text Cleaning (HTML, emoji, URL, stopword removal)
        ↓
   Fraud Indicator Feature Engineering
        ↓
   Stratified Split (70% train / 15% val / 15% test)
        ↓
   TF-IDF Vectorization + Metadata Encoding
        ↓
   Model Training & Evaluation
        ↓
   Model Comparison Table (models/comparison.csv)
```

---

## 🐳 Docker & Docker Compose

### Quick Start with Docker Compose

```bash
# Start the API server
docker compose up api

# Access API
curl http://localhost:8000/health
```

### Train Models Inside Docker

```bash
# Run all model training inside a container
docker compose --profile train up trainer
```

> Models and data are mounted as volumes — trained models persist on your host machine.

### Enable Monitoring (Prometheus + Grafana)

```bash
# Start API + Prometheus + Grafana
docker compose --profile monitoring up
```

- **API:** [http://localhost:8000](http://localhost:8000)
- **Prometheus:** [http://localhost:9090](http://localhost:9090)
- **Grafana:** [http://localhost:3000](http://localhost:3000) (admin/admin)

### Standalone Docker (without Compose)

```bash
# Build image
docker build -t fake-job-api .

# Run container
docker run -p 8000:8000 -v ./models:/app/models fake-job-api

# Access API
curl http://localhost:8000/health
```

### Docker Compose Services

| Service | Port | Profile | Description |
|---------|------|---------|-------------|
| `api` | 8000 | default | FastAPI prediction server |
| `trainer` | — | `train` | One-off model training |
| `prometheus` | 9090 | `monitoring` | Metrics collection |
| `grafana` | 3000 | `monitoring` | Dashboards |

---

## 🔬 Explainability

### LIME (Local Interpretable Model-agnostic Explanations)
- Explains individual predictions by highlighting contributing words
- Integrated into the `/explain` API endpoint

### SHAP (SHapley Additive exPlanations)
- Global feature importance for ML models
- Available via `src/inference/explain.py`

---

## 📚 Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data** | pandas, numpy, datasets (HuggingFace) |
| **ML** | scikit-learn, XGBoost, LightGBM, imbalanced-learn |
| **Deep Learning** | PyTorch, Transformers (HuggingFace) |
| **NLP** | NLTK, BeautifulSoup4 |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Explainability** | SHAP, LIME |
| **Visualization** | Matplotlib, Seaborn |
| **Testing** | pytest, httpx |

---

## 📝 Dataset

**Source:** [victor/real-or-fake-fake-jobposting-prediction](https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction)

| Field | Type | Description |
|-------|------|-------------|
| `title` | text | Job title |
| `company_profile` | text | Company description |
| `description` | text | Job description |
| `requirements` | text | Job requirements |
| `benefits` | text | Job benefits |
| `telecommuting` | binary | Remote work flag |
| `has_company_logo` | binary | Logo presence |
| `has_questions` | binary | Screening questions |
| `employment_type` | categorical | Full-time, Part-time, etc. |
| `required_experience` | categorical | Entry, Mid, Senior, etc. |
| `required_education` | categorical | Bachelor's, Master's, etc. |
| `industry` | categorical | Industry sector |
| **`fraudulent`** | **binary** | **Target — 0 (Real) / 1 (Fake)** |

---

## 🗂 Deliverables

- ✅ Clean, documented codebase (20+ source files)
- ✅ Reproducible training scripts with CLI arguments
- ✅ 6 trained models with comparison table
- ✅ Production-ready FastAPI with 4 endpoints
- ✅ SHAP & LIME explainability
- ✅ Dockerized deployment
- ✅ Comprehensive README with results

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
