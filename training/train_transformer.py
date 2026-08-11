"""Fine-tune a transformer classifier on the same splits as the linear baseline.

    Written 10 Aug - Ankita. This is the heavier alternative deferred in
    [Decisions §4/§13](../docs/design/decisions.md): instead of TF-IDF + a linear
    model, a pretrained encoder (BERT-mini by default) is fine-tuned end-to-end on
    the review text.

    It deliberately reuses `data/processed/train_v1.csv` / `test_v1.csv`, so the
    split is byte-identical to the one the baseline was scored on and the macro-F1
    numbers are directly comparable. Metrics are written in the **same JSON shape**
    as `training/train_linear.py`, into `training/metrics_transformer.json`.

    The loss is the stock unweighted cross-entropy; unlike the linear baseline it
    does **not** apply `class_weight="balanced"` yet, so read its macro-F1 with
    that difference in mind.

    `transformers`, `torch` and `datasets` are heavy dependencies and are imported
    lazily, so importing this module (or running the other pipeline stages) never
    requires them. They are declared in `requirements.txt`.

    Like the linear baseline, every run is recorded as an MLflow experiment run
    (`training/tracking.py`) in the same `review_sentiment` experiment, with the
    same parameter names, the same metric names (macro-F1, accuracy, ROC-AUC,
    per-class scores) and the same artifacts, so the two model families are
    directly comparable side by side in `mlflow ui`.

    This is the `train_transformer` stage in `dvc.yaml`, defined like `train`: deps
    are the two split CSVs plus this script, the out is `model_store/bert_mini_v1`
    and `training/metrics_transformer.json` is a `cache: false` metric. Run it via
    `dvc repro`, or standalone from the repo root:

        python -m training.train_transformer --limit 5000 --epochs 1   # smoke test
        python -m training.train_transformer                           # full fine-tune

    Note: a full fine-tune over ~400k reviews is a GPU-scale job (hours). Start with
    `--limit`, and use `--subsample` for a stratified fraction of the training set.
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

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_STORE = REPO_ROOT / "model_store"

# Keep names to match dvc.yaml. Named after what is actually fine-tuned
# (BASE_MODEL, a BERT-mini) rather than "distilbert", which the stage stopped
# using; an artifact name that lies about its weights is worse than no name.
MODEL_DIR = MODEL_STORE / "bert_mini_v1"
CHECKPOINT_DIR = MODEL_STORE / "bert_mini_v1_checkpoints"

METRICS_PATH = Path(__file__).resolve().parent / "metrics_transformer.json"

# Simplified defaults for faster runs
# Google's official BERT-mini (4 layers, hidden 256). The equally sized
# `prajjwal1/bert-mini` mirror is not loadable on transformers 5.x: its config.json
# has no `model_type` and it ships no `tokenizer.json`.
BASE_MODEL = "google/bert_uncased_L-4_H-256_A-4"
# Some older checkpoints only ship a legacy `vocab.txt` and no `tokenizer.json`;
# transformers 5.x can no longer convert those on the fly. They reuse the stock
# bert-base-uncased WordPiece vocab, so fall back to it.
TOKENIZER_FALLBACK = "google-bert/bert-base-uncased"
# decisions.md #15 documents "max_length=256, 2 epochs, lr 2e-5" as the
# intended config, but that combination multiplies runtime past what's
# workable here (11 Aug - Ankita) - kept at the faster values that already
# produced a real, scored run; only the input text and the loss weighting
# change today, not these three.
MAX_LENGTH = 64
EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 0

LABEL2ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# TEXT_COLUMN = "full_review"
# Raw text tested in isolation 11 Aug - Ankita against the clean_review
# baseline (macro-F1 0.6461, accuracy 70.93%): full_review scored 0.6459 /
# 70.67% - a wash, marginally lower on both. The theory (a transformer
# pretrained on natural English shouldn't need stripped/underscore-glued
# text) didn't pay off under this fast config, likely because MAX_LENGTH=64
# truncates raw text harder - it tokenizes into more subword pieces
# (contractions split, punctuation becomes its own tokens) than the
# pre-normalized clean_review. Reverted to the proven, simpler option.
TEXT_COLUMN = "clean_review"


def load_split(path: Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=limit)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df = df[df[TEXT_COLUMN].str.strip().astype(bool)]
    df["label"] = df["sentiment"].map(LABEL2ID)
    return df.reset_index(drop=True)


def compute_metrics(eval_pred) -> dict:
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


# Added by Ankita 11 Aug: confidence_threshold param, matching train_linear.py's
# evaluate() and train_rnn.py's full_metrics() - same opt-in abstention,
# reusing the softmax probabilities already computed for ROC-AUC.
def full_metrics(y_true, y_pred, probabilities=None, confidence_threshold: float | None = None) -> dict:
    """Score the fine-tune in the *same shape* as `training/train_linear.py`.

    Identical metric names are what makes the model families sortable
    against each other in a single MLflow experiment view.
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
    if probabilities is not None:
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

        # Added by Ankita 11 Aug
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
    confidence_threshold: float | None = None,  # Added by Ankita 11 Aug
) -> dict:
    """The knobs worth logging, named like the baseline's so the UI can align them."""
    return {
        "model_name": base_model,
        "model.num_train_epochs": epochs,
        "model.per_device_train_batch_size": batch_size,
        "model.learning_rate": learning_rate,
        "model.weight_decay": WEIGHT_DECAY,
        "model.warmup_steps": WARMUP_STEPS,
        "model.max_length": MAX_LENGTH,
        # "model.class_weight": "none",
        "model.class_weight": "balanced",  # Added by Ankita 11 Aug
        "random_state": RANDOM_STATE,
        "limit": limit if limit is not None else "none",
        "train_csv": TRAIN_CSV.name,
        "test_csv": TEST_CSV.name,
        # Added by Ankita 11 Aug
        "confidence_threshold": confidence_threshold if confidence_threshold is not None else "none",
    }


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
    from datasets import Dataset
    ds = Dataset.from_pandas(df[[TEXT_COLUMN, "label"]], preserve_index=False)

    def _tokenize(batch):
        return tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=MAX_LENGTH)

    ds = ds.map(_tokenize, batched=True, remove_columns=[TEXT_COLUMN])
    return ds


# Added by Ankita 11 Aug: built inside train(), not at module level - Trainer
# is one of the lazily-imported transformers symbols (module docstring: the
# heavy deps are imported inside the functions that need them so the rest of
# the pipeline runs without them), and a class statement needs its base class
# at definition time, so this factory defers the class body until train() has
# already done that import.
def _build_weighted_trainer_class(Trainer, nn):
    """The stock Trainer uses unweighted cross-entropy: a documented gap
    (decisions.md #17 - "planned but never landed"). Overriding compute_loss
    is the standard way to give it a class-weighted loss, matching the
    balanced treatment the linear/RNN trainers already apply."""

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


def train(
    base_model: str = BASE_MODEL,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    limit: int | None = None,
    confidence_threshold: float | None = None,  # Added by Ankita 11 Aug
    output_dir: Path = MODEL_DIR,
) -> dict:
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

    train_df = load_split(TRAIN_CSV, limit)
    test_df = load_split(TEST_CSV, limit)
    logger.info("train rows %d, test rows %d, base model %s", len(train_df), len(test_df), base_model)

    # Added by Ankita 11 Aug: same balanced treatment as the linear/RNN
    # trainers (decisions.md #17), computed from the train split only.
    from sklearn.utils.class_weight import compute_class_weight

    label_ids = np.arange(len(SENTIMENT_LABELS))
    class_weights = compute_class_weight(
        "balanced", classes=label_ids, y=train_df["label"].to_numpy()
    )
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    tokenizer = load_tokenizer(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(SENTIMENT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = build_dataset(train_df, tokenizer)
    test_ds = build_dataset(test_df, tokenizer)

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

    # Added by Ankita 11 Aug
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

    # A subsample run is a smoke test, tagged exactly as the baseline tags its
    # own, so scored and unscored runs stay distinguishable in the UI.
    tags = {
        "stage": "train_transformer",
        "framework": "transformers",
        "smoke_test": str(limit is not None),
    }
    params = run_params(base_model, epochs, batch_size, learning_rate, limit, confidence_threshold)

    with tracking.start_run("bert_mini", params, tags) as run:
        trainer.train()

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

        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logger.info("Wrote model: %s", output_dir)

        if limit is None:
            METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
            logger.info("Wrote metrics: %s", METRICS_PATH)
            run.log_artifact(METRICS_PATH, "metrics")

        run.log_metrics(metrics)
        run.log_dict(metrics["confusion_matrix"], "confusion_matrix.json")
        run.log_artifact(output_dir, "model")

        metrics["mlflow_run_id"] = run.run_id
        logger.info("MLflow run id: %s (experiment %s)", run.run_id, tracking.EXPERIMENT_NAME)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL, help="pretrained checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--limit", type=int, default=None, help="only read the first N rows of each split")
    # Added by Ankita 11 Aug: same opt-in abstention as train_linear.py/train_rnn.py
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
