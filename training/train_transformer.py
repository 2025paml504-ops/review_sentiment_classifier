"""Fine-tune a pretrained transformer classifier on the same splits as the
linear baseline.

This is the heaviest of the three model families, next to the TF-IDF
linear baseline (`training/train_linear.py`) and the from-scratch
recurrent net (`training/train_rnn.py`): instead of hand-built features, a
pretrained language model is fine-tuned end-to-end on the review text
itself.

It deliberately reuses `data/processed/train_v1.csv` / `test_v1.csv`, so
the split is byte-identical to what the other two models were scored on,
and every metric is written in the same JSON shape as `train_linear.py`
and `train_rnn.py` - the point is that all three models can be compared
directly, not just individually understood.

`transformers`, `torch` and `datasets` are heavy dependencies and are
imported lazily inside the functions that need them, so importing this
module - or running the rest of the pipeline - never requires them.

How this file is organized, top to bottom, following the order the code
actually runs in:

    - Config              - which checkpoint, how long to train, hypothesis text
    - Data loading         - load_split()
    - Text -> tokens       - load_tokenizer(), build_dataset() (HuggingFace's
      equivalent of train_rnn.py's build_vocabulary()/encode() step)
    - Scoring helpers      - compute_metrics(), softmax(), full_metrics(),
      run_params()
    - Class-weighted loss  - _build_weighted_trainer_class() (HuggingFace's
      Trainer has no built-in class weighting, so this adds it)
    - train()              - the actual pipeline: load -> tokenize -> build
      model -> fit -> evaluate -> save
    - CLI                  - what happens when you run this file directly

Run it:

    python -m training.train_transformer --limit 5000 --epochs 1   # smoke test
    python -m training.train_transformer                           # full fine-tune

Note: a full fine-tune on CPU is much slower than a GPU would be. This
project went through two base-model choices before settling on one small
enough to be practical here - see decisions.md #15 for bert-mini's original
reasoning, and the Config section below for how it ended up on bert-tiny.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from features.build_features import SENTIMENT_LABELS
from features.vectorize import RANDOM_STATE, TEST_CSV, TRAIN_CSV
from training import tracking

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_transformer")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_STORE = REPO_ROOT / "model_store"


# ── Config ────────────────────────────────────────────────────────────

# Named after what is actually fine-tuned. This has changed twice:
# bert-mini -> distilbert-base-uncased (too slow on CPU, ~20h/epoch) ->
# bert-tiny (2 layers, hidden 128, ~4.4M params - smaller than bert-mini).
# The folder gets renamed each time, because an artifact name that lies
# about its weights is worse than no name at all (decisions.md #19).
MODEL_DIR = MODEL_STORE / "bert_tiny_v1"
CHECKPOINT_DIR = MODEL_STORE / "bert_tiny_v1_checkpoints"
METRICS_PATH = Path(__file__).resolve().parent / "metrics_transformer.json"

BASE_MODEL = "google/bert_uncased_L-2_H-128_A-2"
# Some older checkpoints only ship a legacy `vocab.txt` and no `tokenizer.json`;
# transformers 5.x can no longer convert those on the fly. They reuse the stock
# bert-base-uncased WordPiece vocab, so fall back to it.
TOKENIZER_FALLBACK = "google-bert/bert-base-uncased"

# Kept deliberately fast: this runs on CPU, with no GPU available.
MAX_LENGTH = 64
EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 0

# clean_review (not the raw full_review) is what gets tokenized - tested
# raw text once and it scored marginally worse, likely because MAX_LENGTH=64
# truncates unprocessed text harder than the pre-cleaned version.
TEXT_COLUMN = "clean_review"

LABEL2ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# What this config is expected to show, logged as an MLflow tag before
# training starts and checked against the real result afterward. The
# config is fixed run to run, so this is as much a reproducibility check
# as a fresh experiment.
HYPOTHESIS_TEMPLATE = (
    "Fine-tuning {base_model} for {epochs} epoch(s) on clean_review with "
    "class-weighted cross-entropy will reproduce closely if run again with "
    "the same config (base model, MAX_LENGTH={max_length}, learning rate) "
    "unchanged (decisions.md #15, #17)."
)


# ── Data loading ──────────────────────────────────────────────────────

def load_split(path: Path, limit: int | None = None) -> pd.DataFrame:
    """Read a train/test split written by features/vectorize.py."""
    df = pd.read_csv(path, nrows=limit)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df = df[df[TEXT_COLUMN].str.strip().astype(bool)]
    df["label"] = df["sentiment"].map(LABEL2ID)
    return df.reset_index(drop=True)


# ── Text -> tokens ────────────────────────────────────────────────────
# A pretrained model comes with its own tokenizer and its own fixed
# vocabulary - this is HuggingFace's equivalent of train_rnn.py building a
# vocabulary and encoding text into ids by hand.

def load_tokenizer(base_model: str):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(base_model)
    except (ValueError, OSError) as exc:
        logger.warning(
            "No usable tokenizer for %s (%s); falling back to %s",
            base_model, exc, TOKENIZER_FALLBACK,
        )
        return AutoTokenizer.from_pretrained(TOKENIZER_FALLBACK)


def build_dataset(df: pd.DataFrame, tokenizer):
    """Text column -> a HuggingFace Dataset of token ids, ready for the Trainer."""
    from datasets import Dataset
    ds = Dataset.from_pandas(df[[TEXT_COLUMN, "label"]], preserve_index=False)

    def _tokenize(batch):
        return tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=MAX_LENGTH)

    return ds.map(_tokenize, batched=True, remove_columns=[TEXT_COLUMN])


# ── Scoring helpers ───────────────────────────────────────────────────

def compute_metrics(eval_pred) -> dict:
    """Called by HuggingFace's Trainer after each eval pass, during training."""
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, so ROC-AUC sees probabilities rather than raw logits."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def full_metrics(y_true, y_pred, probabilities=None, confidence_threshold: float | None = None) -> dict:
    """Score the fine-tune in the *same shape* as `training/train_linear.py`
    and `training/train_rnn.py` - identical metric names are what makes all
    three model families sortable against each other in one MLflow view.

    `confidence_threshold` is optional abstention, mirroring the other two
    trainers - reuses the softmax probabilities already computed for
    ROC-AUC, so it's free.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    label_ids = [LABEL2ID[label] for label in SENTIMENT_LABELS]
    report = classification_report(
        y_true, y_pred, labels=label_ids, output_dict=True, zero_division=0
    )
    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    # Binary (2 classes) only needs the positive class's probability and no
    # averaging; 3+ classes need scikit-learn's one-vs-rest averaging
    # instead, which expects the full per-class probability matrix.
    if probabilities is not None:
        if len(label_ids) == 2:
            try:
                auc = float(roc_auc_score(y_true, probabilities[:, 1]))
                metrics["roc_auc_macro"] = auc
                metrics["roc_auc_weighted"] = auc
            except ValueError as exc:
                logger.warning("ROC-AUC not computable: %s", exc)
        else:
            for average in ("macro", "weighted"):
                try:
                    metrics[f"roc_auc_{average}"] = float(
                        roc_auc_score(
                            y_true, probabilities, multi_class="ovr",
                            average=average, labels=label_ids,
                        )
                    )
                except ValueError as exc:
                    logger.warning("ROC-AUC (%s) not computable: %s", average, exc)

        if confidence_threshold is not None:
            confidence = probabilities.max(axis=1)
            thresholded_pred = probabilities.argmax(axis=1)
            covered = confidence >= confidence_threshold
            y_true_arr = np.asarray(y_true)
            metrics["confidence_threshold"] = confidence_threshold
            metrics["coverage"] = float(covered.mean())
            metrics["accuracy_at_threshold"] = (
                float(accuracy_score(y_true_arr[covered], thresholded_pred[covered]))
                if covered.any()
                else None
            )
    metrics["per_class"] = {
        label: {
            "precision": float(report[str(LABEL2ID[label])]["precision"]),
            "recall": float(report[str(LABEL2ID[label])]["recall"]),
            "f1": float(report[str(LABEL2ID[label])]["f1-score"]),
            "support": int(report[str(LABEL2ID[label])]["support"]),
        }
        for label in SENTIMENT_LABELS
    }
    metrics["confusion_matrix"] = {
        "labels": SENTIMENT_LABELS,
        "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
    }
    return metrics


def run_params(
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    limit: int | None,
    confidence_threshold: float | None = None,
) -> dict:
    """The settings worth logging, named like the other trainers' so the UI can align them."""
    return {
        "model_name": base_model,
        "model.num_train_epochs": epochs,
        "model.per_device_train_batch_size": batch_size,
        "model.learning_rate": learning_rate,
        "model.weight_decay": WEIGHT_DECAY,
        "model.warmup_steps": WARMUP_STEPS,
        "model.max_length": MAX_LENGTH,
        "model.class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "limit": limit if limit is not None else "none",
        "train_csv": TRAIN_CSV.name,
        "test_csv": TEST_CSV.name,
        "confidence_threshold": confidence_threshold if confidence_threshold is not None else "none",
    }


# ── Class-weighted loss ───────────────────────────────────────────────

def _build_weighted_trainer_class(Trainer, nn):
    """HuggingFace's stock Trainer has no built-in class weighting - only
    plain unweighted cross-entropy. Overriding compute_loss is the standard
    way to give it a class-weighted loss, matching the balanced treatment
    the linear/RNN trainers already apply (decisions.md #17).

    Built inside train(), not at module level: Trainer itself is one of the
    heavy symbols imported lazily (see the module docstring), but a class
    statement needs its base class to already exist - so this factory just
    defers building the class until train() has done that import.
    """

    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


# ── train() - the actual pipeline ────────────────────────────────────

def train(
    base_model: str = BASE_MODEL,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    limit: int | None = None,
    confidence_threshold: float | None = None,
    output_dir: Path = MODEL_DIR,
) -> dict:
    """Load data -> tokenize -> build model -> fit -> evaluate -> save."""
    import torch
    from torch import nn
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(RANDOM_STATE)

    # Step 1: load the text splits.
    train_df = load_split(TRAIN_CSV, limit)
    test_df = load_split(TEST_CSV, limit)
    logger.info("train rows %d, test rows %d, base model %s", len(train_df), len(test_df), base_model)

    # Step 2: compute class weights (same balanced treatment as the
    # linear/RNN trainers, decisions.md #17), from the train split only.
    from sklearn.utils.class_weight import compute_class_weight

    label_ids = np.arange(len(SENTIMENT_LABELS))
    class_weights = compute_class_weight(
        "balanced", classes=label_ids, y=train_df["label"].to_numpy()
    )
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Step 3: load the pretrained tokenizer + model, and tokenize both splits.
    tokenizer = load_tokenizer(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(SENTIMENT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    train_ds = build_dataset(train_df, tokenizer)
    test_ds = build_dataset(test_df, tokenizer)

    # Step 4: set up the Trainer with the class-weighted loss from §5.
    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=200,
        seed=RANDOM_STATE,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    WeightedTrainer = _build_weighted_trainer_class(Trainer, nn)
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weight_tensor,
    )

    # Step 5: fit, evaluate, and log everything to MLflow in one tracked run.
    tags = {
        "stage": "train_transformer",
        "framework": "transformers",
        # A subsample run is a smoke test, tagged exactly as the other
        # trainers tag their own, so scored and unscored runs stay
        # distinguishable in the UI.
        "smoke_test": str(limit is not None),
    }
    params = run_params(base_model, epochs, batch_size, learning_rate, limit, confidence_threshold)
    hypothesis = HYPOTHESIS_TEMPLATE.format(base_model=base_model, epochs=epochs, max_length=MAX_LENGTH)

    with tracking.start_run("bert_tiny", params, tags, hypothesis=hypothesis) as run:
        trainer.train()

        # Step 6: score on the held-out test split.
        predictions = trainer.predict(test_ds)
        y_pred = np.argmax(predictions.predictions, axis=-1)
        y_true = np.asarray(test_ds["label"])
        metrics = full_metrics(y_true, y_pred, softmax(predictions.predictions), confidence_threshold)
        metrics["model"] = base_model
        metrics["n_train"] = len(train_df)
        metrics["n_test"] = len(test_df)
        metrics["epochs"] = epochs
        metrics["max_length"] = MAX_LENGTH
        logger.info(
            "macro F1: %.4f (accuracy %.4f, roc_auc %.4f)",
            metrics["macro_f1"], metrics["accuracy"], metrics.get("roc_auc_macro", float("nan")),
        )

        # Step 7: save the model. The tokenizer travels with the weights -
        # a fine-tuned model is useless without the exact tokenizer it was
        # trained against.
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logger.info("Wrote model: %s", output_dir)

        # A smoke run (--limit) must not overwrite the scored metrics file
        # DVC tracks - the run itself is still recorded, tagged smoke_test=True.
        if limit is None:
            METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
            logger.info("Wrote metrics: %s", METRICS_PATH)
            run.log_artifact(METRICS_PATH, "metrics")

        # Step 8: log the run's record.
        run.log_metrics(metrics)
        run.log_dict(metrics["confusion_matrix"], "confusion_matrix.json")
        run.log_artifact(output_dir, "model")
        run.set_conclusion(
            f"macro-F1 {metrics['macro_f1']:.4f}, accuracy {metrics['accuracy']:.4f} "
            f"after {epochs} epoch(s)."
        )

        metrics["mlflow_run_id"] = run.run_id
        logger.info("MLflow run id: %s (experiment %s)", run.run_id, tracking.EXPERIMENT_NAME)

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL, help="pretrained checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--limit", type=int, default=None, help="only read the first N rows of each split")
    parser.add_argument(
        "--confidence-threshold", type=float, default=None,
        help="abstain below this confidence; logs coverage and accuracy_at_threshold",
    )
    args = parser.parse_args()

    train(
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        limit=args.limit,
        confidence_threshold=args.confidence_threshold,
    )
