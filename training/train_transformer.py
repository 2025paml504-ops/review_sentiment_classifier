"""
Fine-tune DistilBERT for sentiment classification.
Compare printed metrics against your TF-IDF + LogisticRegression model.

Install: pip install transformers datasets torch scikit-learn --break-system-packages
Run:     python train_transformer_simple.py
"""

import sqlite3
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from feature_store.feature_store import REPO_ROOT

FEATURE_STORE_PATH = REPO_ROOT / "feature_store" / "feature_store.db"
LABEL_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# ─── Load Data ────────────────────────────────────────────────
conn = sqlite3.connect(FEATURE_STORE_PATH)
df = pd.read_sql("SELECT clean_review, sentiment FROM hotel_review_features", conn)
conn.close()
df = df.dropna(subset=["clean_review", "sentiment"]).reset_index(drop=True)

# Quick sanity check on a subsample first (full dataset takes ~7.5 hrs on MPS).
# Set to None once you're ready to run on the full dataset.
SAMPLE_SIZE = 100000
if SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

print(f'Loaded {len(df)} rows from feature store')

le = LabelEncoder()
y = le.fit_transform(df["sentiment"])
X = df["clean_review"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── Tokenize ─────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenize = lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()}).map(tokenize, batched=True)
test_ds = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()}).map(tokenize, batched=True)

# ─── Train Model ──────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="model_store/transformer_checkpoints",
        per_device_train_batch_size=16,
        num_train_epochs=3,  # bump to 3 once you're doing the real full-dataset run
        learning_rate=2e-5,
    ),
    train_dataset=train_ds,
)
trainer.train()

# ─── Evaluate ─────────────────────────────────────────────────
y_pred = np.argmax(trainer.predict(test_ds).predictions, axis=1)

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABEL_NAMES], columns=[f"pred_{l}" for l in LABEL_NAMES])
print("\nConfusion Matrix:")
print(cm_df)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# ─── Save Artifacts ───────────────────────────────────────────
trainer.save_model("model_store/sentiment_transformer")
tokenizer.save_pretrained("model_store/sentiment_transformer")
print("✅  Model saved to model_store/sentiment_transformer")