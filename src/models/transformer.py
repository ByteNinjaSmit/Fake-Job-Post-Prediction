"""
BERT / Transformer model for sequence classification.
Fine-tunes a pretrained model for binary fraud detection.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from pathlib import Path
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    BERT_MODEL_NAME,
    BERT_MAX_LENGTH,
    BERT_BATCH_SIZE,
    BERT_LEARNING_RATE,
    BERT_EPOCHS,
    BERT_WARMUP_STEPS,
    MODELS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Custom Dataset ──────────────────────────────────────────────
class JobPostDataset(Dataset):
    """PyTorch Dataset for job posting texts."""

    def __init__(self, texts, labels, tokenizer, max_length=BERT_MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─── Model wrapper ───────────────────────────────────────────────
class BERTClassifier:
    """Wrapper around HuggingFace Transformer for training and inference."""

    def __init__(self, model_name: str = BERT_MODEL_NAME, num_labels: int = 2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model.to(self.device)

    def train(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """Fine-tunes the transformer model."""
        train_dataset = JobPostDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=BERT_BATCH_SIZE, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=BERT_LEARNING_RATE, weight_decay=0.01)
        total_steps = len(train_loader) * BERT_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=BERT_WARMUP_STEPS, num_training_steps=total_steps
        )

        self.model.train()
        for epoch in range(BERT_EPOCHS):
            total_loss = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{BERT_EPOCHS}")

            for batch in progress:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} — Avg Loss: {avg_loss:.4f}")

            # Validation
            if val_texts is not None and val_labels is not None:
                val_acc = self.evaluate_accuracy(val_texts, val_labels)
                logger.info(f"Epoch {epoch + 1} — Val Accuracy: {val_acc:.4f}")

    def predict(self, texts):
        """Returns predicted labels."""
        self.model.eval()
        dataset = JobPostDataset(texts, [0] * len(texts), self.tokenizer)
        loader = DataLoader(dataset, batch_size=BERT_BATCH_SIZE)

        all_preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())

        return np.array(all_preds)

    def predict_proba(self, texts):
        """Returns probability scores for each class."""
        self.model.eval()
        dataset = JobPostDataset(texts, [0] * len(texts), self.tokenizer)
        loader = DataLoader(dataset, batch_size=BERT_BATCH_SIZE)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)

    def evaluate_accuracy(self, texts, labels):
        """Quick accuracy evaluation."""
        preds = self.predict(texts)
        return (preds == np.array(labels)).mean()

    def save(self, path=None):
        """Save model and tokenizer."""
        if path is None:
            path = MODELS_DIR / "bert_classifier"
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"BERT model saved → {path}")

    def load(self, path=None):
        """Load saved model and tokenizer."""
        if path is None:
            path = MODELS_DIR / "bert_classifier"
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        logger.info(f"BERT model loaded ← {path}")
