# Fake Job Post Prediction

Comparison of classical ML and Deep Learning models for detecting fake job postings.

## Project Structure
- `data/`: Dataset storage
- `notebooks/`: Exploratory Data Analysis and experimentation
- `src/`: Source code for data processing, modeling, and API

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run data pipeline:
   ```bash
   python src/data/dataset.py
   ```
3. Train model:
   ```bash
   python src/training/train.py
   ```
4. Run API:
   ```bash
   uvicorn src.api.app:app --reload
   ```
