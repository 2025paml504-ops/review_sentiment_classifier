"""Recurrent (RNN / LSTM / BiLSTM) sentiment classifier (v1.2).

    The third model family, next to the TF-IDF linear baseline
    (`training/train_linear.py`) and the transformer fine-tune
    (`training/train_transformer.py`). It reads the *same* splits,
    `data/processed/train_v1.csv` / `test_v1.csv`, so its macro-F1 is directly
    comparable, and writes metrics in the same JSON shape.

    Why this is a separate module and not a branch of `train_linear.py`: a
    recurrent net consumes an **ordered sequence of token ids**, while
    `train_linear.py` works on the TF-IDF matrix, which is an unordered
    bag-of-ngrams weighting. Feeding TF-IDF rows to an `Embedding`/`LSTM` (even
    after padding) destroys the only thing a recurrent model adds - word order -
    and turns a 20k-dimensional sparse matrix into a dense one, which is
    hundreds of GB over the full split. So this trainer builds its own
    vocabulary from the training text and never touches the vectorizer.

    Built on **PyTorch**, not Keras: this project runs on Python 3.14, for which
    TensorFlow publishes no wheels, while torch is already a dependency of the
    transformer stage (decisions §20). `torch` is heavy and imported lazily, so
    the rest of the pipeline runs without it.

    Every run is recorded as an MLflow run (`training/tracking.py`) in the same
    `review_sentiment` experiment, with the same parameter names, metric names
    (macro-F1, weighted-F1, accuracy, ROC-AUC, per-class scores) and artifacts,
    so all three families rank side by side in `mlflow ui` and in
    `python -m training.compare_runs`.

    This is the `train_rnn` stage in `dvc.yaml`. Standalone:

        python -m training.train_rnn --limit 5000 --epochs 1   # smoke test
        python -m training.train_rnn                           # full run
        python -m training.train_rnn --arch rnn_bilstm         # other architectures
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from features.build_features import SENTIMENT_LABELS
from features.vectorize import RANDOM_STATE, TEST_CSV, TRAIN_CSV
from training import tracking

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_rnn")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_STORE = REPO_ROOT / "model_store"
METRICS_DIR = Path(__file__).resolve().parent

# Sequence encoding. 0 is reserved for padding and 1 for out-of-vocabulary, so
# real token ids start at 2 - the usual convention for text pipelines.
PAD_ID = 0
OOV_ID = 1
NUM_WORDS = 20000  # matches the TF-IDF vocabulary budget, for a fair comparison
MAX_LENGTH = 200

# Training defaults. Deliberately modest: this runs on CPU.
EMBEDDING_DIM = 128
RECURRENT_UNITS = 128
DROPOUT = 0.2
# 4 epochs tried (v1.2): val loss kept dropping (0.689 -> 0.687) but
# macro-F1 fell 0.6436 -> 0.6393 - epoch 4 overfits the metric that matters,
# even though raw loss looked fine. Reverted to 3.
EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.1

# The recurrent variants. The name is also the MLflow run name and the artifact
# stem, so a run can never be confused with another architecture's.
ARCHITECTURES = ("rnn_lstm", "rnn_bilstm", "rnn_simple")
DEFAULT_ARCH = "rnn_lstm"

# Added (v1.4): logged as a hypothesis tag before training (training/tracking.py).
HYPOTHESIS_TEMPLATE = (
    "Training for {epochs} epochs will keep macro-F1 at or above the proven "
    "3-epoch baseline - a 4th epoch previously overfit the metric that "
    "mattered even while validation loss kept falling (decisions.md #21)."
)

LABEL2ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def model_path(arch: str) -> Path:
    """Where the fitted network is saved (a torch `state_dict` checkpoint)."""
    return MODEL_STORE / f"{arch}_v1.pt"


def vocab_path(arch: str) -> Path:
    """The word index is part of the model: without it the weights are unusable."""
    return MODEL_STORE / f"{arch}_v1_vocab.json"


def metrics_path(arch: str) -> Path:
    return METRICS_DIR / f"metrics_{arch}.json"


def load_split(path: Path, limit: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `dvc repro vectorize` first")
    df = pd.read_csv(path, nrows=limit)
    df["clean_review"] = df["clean_review"].fillna("").astype(str)
    df = df[df["clean_review"].str.strip().astype(bool)]
    df["label"] = df["sentiment"].map(LABEL2ID)
    return df.reset_index(drop=True)


def build_vocabulary(texts, num_words: int = NUM_WORDS) -> dict[str, int]:
    """Word index of the `num_words` most frequent training tokens.

    Built from the **train split only**, for the same no-leakage reason the
    TF-IDF vectorizer is fit on train only. Ties are broken alphabetically so
    the vocabulary is deterministic across runs.
    """
    counts = Counter()
    for text in texts:
        counts.update(text.split())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {word: index for index, (word, _) in enumerate(ordered[:num_words], start=OOV_ID + 1)}


def encode(texts, vocab: dict[str, int], max_length: int = MAX_LENGTH) -> np.ndarray:
    """Text -> right-padded, right-truncated int64 id matrix.

    Right padding (rather than left) is what `pack_padded_sequence` expects, so
    the recurrent layer can be told the true length of every row and never reads
    a padding position.
    """
    encoded = np.full((len(texts), max_length), PAD_ID, dtype="int64")
    for row, text in enumerate(texts):
        ids = [vocab.get(word, OOV_ID) for word in text.split()[:max_length]]
        if ids:
            encoded[row, : len(ids)] = ids
    return encoded


def sequence_lengths(encoded: np.ndarray) -> np.ndarray:
    """Number of real (non-padding) tokens per row, floored at 1."""
    return np.maximum((encoded != PAD_ID).sum(axis=1), 1).astype("int64")


def build_model(arch: str, vocab_size: int):
    """Embedding -> recurrent layer -> dropout -> logits over the three classes."""
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence

    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch}', expected one of {list(ARCHITECTURES)}")

    class RecurrentClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.arch = arch
            # padding_idx keeps the pad embedding fixed at zero and out of the
            # gradient, the torch equivalent of Keras' mask_zero.
            self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=PAD_ID)
            bidirectional = arch == "rnn_bilstm"
            cell = nn.RNN if arch == "rnn_simple" else nn.LSTM
            self.rnn = cell(
                EMBEDDING_DIM,
                RECURRENT_UNITS,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.dropout = nn.Dropout(DROPOUT)
            self.output = nn.Linear(
                RECURRENT_UNITS * (2 if bidirectional else 1), len(SENTIMENT_LABELS)
            )

        def forward(self, inputs, lengths):
            embedded = self.embedding(inputs)
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, state = self.rnn(packed)
            hidden = state[0] if isinstance(state, tuple) else state
            # hidden: (num_directions, batch, units) -> concatenate directions.
            final = torch.cat([hidden[i] for i in range(hidden.shape[0])], dim=1)
            return self.output(self.dropout(final))

    return RecurrentClassifier()


def class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Balanced weights, so the loss matches the linear baseline's objective.

    The baseline uses `class_weight="balanced"` ([Decisions §17](../docs/design/decisions.md));
    without the same treatment the macro-F1 of the two families would be
    optimising different things.
    """
    from sklearn.utils.class_weight import compute_class_weight

    present = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=present, y=y_train)
    return {int(label): float(weight) for label, weight in zip(present, weights)}


# Added (v1.2): confidence_threshold param, mirroring train_linear.py's
# evaluate() - same abstention idea, deferring instead of forcing a guess.
# Reuses the softmax probabilities already computed for ROC-AUC, so it's free.
def full_metrics(y_true, y_pred, probabilities=None, confidence_threshold: float | None = None) -> dict:
    """Score in the *same shape* as the other two trainers."""
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

        # Added (v1.2)
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


def run_params(arch: str, epochs: int, batch_size: int, learning_rate: float,
               vocab_size: int, limit: int | None,
               confidence_threshold: float | None = None) -> dict:
    """The knobs worth logging, named like the other trainers' so the UI aligns them."""
    return {
        "model_name": arch,
        "model.framework": "pytorch",
        "model.embedding_dim": EMBEDDING_DIM,
        "model.recurrent_units": RECURRENT_UNITS,
        "model.dropout": DROPOUT,
        "model.max_length": MAX_LENGTH,
        "model.num_words": NUM_WORDS,
        "model.vocab_size": vocab_size,
        "model.epochs": epochs,
        "model.batch_size": batch_size,
        "model.learning_rate": learning_rate,
        "model.validation_split": VALIDATION_SPLIT,
        "model.class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "limit": limit if limit is not None else "none",
        "train_csv": TRAIN_CSV.name,
        "test_csv": TEST_CSV.name,
        # Added (v1.2)
        "confidence_threshold": confidence_threshold if confidence_threshold is not None else "none",
    }


def predict_proba(model, encoded: np.ndarray, batch_size: int) -> np.ndarray:
    """Softmax probabilities for the whole matrix, in eval mode and no-grad."""
    import torch

    lengths = sequence_lengths(encoded)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            chunk = torch.from_numpy(encoded[start:start + batch_size])
            chunk_lengths = torch.from_numpy(lengths[start:start + batch_size])
            logits = model(chunk, chunk_lengths)
            outputs.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(outputs) if outputs else np.zeros((0, len(SENTIMENT_LABELS)))


def train(
    arch: str = DEFAULT_ARCH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    limit: int | None = None,
    confidence_threshold: float | None = None,  # Added (v1.2)
) -> dict:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    # Same seed as every other stage, so the run is reproducible.
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    train_df = load_split(TRAIN_CSV, limit)
    test_df = load_split(TEST_CSV, limit)

    vocab = build_vocabulary(train_df["clean_review"])
    # +2 for the reserved padding and out-of-vocabulary ids.
    vocab_size = len(vocab) + 2

    X_train = encode(train_df["clean_review"], vocab)
    X_test = encode(test_df["clean_review"], vocab)
    y_train = train_df["label"].to_numpy(dtype="int64")
    y_test = test_df["label"].to_numpy(dtype="int64")

    logger.info(
        "train %s, test %s, vocab %d, arch %s", X_train.shape, X_test.shape, vocab_size, arch
    )

    model = build_model(arch, vocab_size)
    tags = {"stage": "train_rnn", "framework": "pytorch", "smoke_test": str(limit is not None)}
    params = run_params(arch, epochs, batch_size, learning_rate, vocab_size, limit, confidence_threshold)

    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(sequence_lengths(X_train)),
        torch.from_numpy(y_train),
    )
    validation_size = int(len(dataset) * VALIDATION_SPLIT)
    train_subset, validation_subset = random_split(
        dataset,
        [len(dataset) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_subset, batch_size=batch_size)

    weights = class_weights(y_train)
    weight_tensor = torch.tensor(
        [weights.get(i, 1.0) for i in range(len(SENTIMENT_LABELS))], dtype=torch.float32
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    hypothesis = HYPOTHESIS_TEMPLATE.format(epochs=epochs)
    with tracking.start_run(arch, params, tags, hypothesis=hypothesis) as run:
        for epoch in range(epochs):
            model.train()
            total, seen = 0.0, 0
            for inputs, lengths, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(inputs, lengths), labels)
                loss.backward()
                optimizer.step()
                total += float(loss) * len(labels)
                seen += len(labels)
            train_loss = total / max(seen, 1)

            model.eval()
            total, seen = 0.0, 0
            with torch.no_grad():
                for inputs, lengths, labels in validation_loader:
                    total += float(criterion(model(inputs, lengths), labels)) * len(labels)
                    seen += len(labels)
            validation_loss = total / max(seen, 1)

            logger.info(
                "epoch %d/%d - loss %.4f - val_loss %.4f",
                epoch + 1, epochs, train_loss, validation_loss,
            )
            # The learning curve is part of the record: it is what tells a
            # plateau apart from an under-trained net.
            run.log_metric_step("train.loss", train_loss, step=epoch)
            if seen:
                run.log_metric_step("val.loss", validation_loss, step=epoch)

        probabilities = predict_proba(model, X_test, batch_size)
        y_pred = np.argmax(probabilities, axis=1)

        metrics = full_metrics(y_test, y_pred, probabilities, confidence_threshold)
        metrics["model"] = arch
        metrics["n_train"] = int(X_train.shape[0])
        metrics["n_test"] = int(X_test.shape[0])
        metrics["n_features"] = int(vocab_size)

        MODEL_STORE.mkdir(parents=True, exist_ok=True)
        # The weights alone are meaningless without the shape that produced
        # them, so the checkpoint carries the architecture and vocab size too.
        torch.save(
            {"arch": arch, "vocab_size": vocab_size, "state_dict": model.state_dict()},
            model_path(arch),
        )
        vocab_path(arch).write_text(json.dumps(vocab), encoding="utf-8")

        run.log_metrics(metrics)
        run.log_dict(metrics["confusion_matrix"], "confusion_matrix.json")
        run.log_artifact(model_path(arch), "model")
        run.log_artifact(vocab_path(arch), "model")
        run.set_conclusion(
            f"macro-F1 {metrics['macro_f1']:.4f}, accuracy {metrics['accuracy']:.4f} "
            f"after {epochs} epoch(s)."
        )

        # Smoke runs (--limit) must not overwrite the scored metrics file that
        # DVC tracks; the run itself is still recorded, tagged smoke_test=True.
        if limit is None:
            metrics_path(arch).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            run.log_artifact(metrics_path(arch))

        metrics["mlflow_run_id"] = run.run_id
        logger.info("MLflow run id: %s (experiment %s)", run.run_id, tracking.EXPERIMENT_NAME)

    logger.info("macro-F1 %.4f, accuracy %.4f", metrics["macro_f1"], metrics["accuracy"])
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=DEFAULT_ARCH, choices=sorted(ARCHITECTURES))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--limit", type=int, default=None, help="only read the first N rows")
    # Added (v1.2): same opt-in abstention as train_linear.py
    parser.add_argument(
        "--confidence-threshold", type=float, default=None,
        help="abstain below this confidence; logs coverage and accuracy_at_threshold",
    )
    args = parser.parse_args()
    train(
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        limit=args.limit,
        confidence_threshold=args.confidence_threshold,
    )
