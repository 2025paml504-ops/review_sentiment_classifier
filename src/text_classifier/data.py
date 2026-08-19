from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import load_config, resolve_path
from .text import clean_text


class DataValidationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_and_validate(config: dict) -> tuple[pd.DataFrame, dict]:
    cfg = config["data"]
    path = resolve_path(cfg["path"])
    if not path.exists():
        raise DataValidationError(f"Dataset not found: {path}. Run scripts/make_demo_data.py.")
    frame = pd.read_csv(path)
    text_col, label_col = cfg["text_column"], cfg["label_column"]
    missing = {text_col, label_col} - set(frame.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")

    selected = frame[[text_col, label_col]].copy()
    source_rows = len(selected)
    null_rows = int(selected[[text_col, label_col]].isna().any(axis=1).sum())
    selected = selected.dropna()
    selected[text_col] = selected[text_col].map(clean_text)
    selected[label_col] = selected[label_col].astype(str).str.strip()
    empty_rows = int((selected[text_col].str.len() == 0).sum())
    selected = selected[selected[text_col].str.len() > 0]
    duplicate_rows = int(selected.duplicated([text_col]).sum())
    selected = selected.drop_duplicates([text_col]).reset_index(drop=True)

    allowed = set(cfg.get("allowed_labels") or selected[label_col].unique())
    invalid_labels = sorted(set(selected[label_col]) - allowed)
    if invalid_labels:
        raise DataValidationError(f"Unexpected labels: {invalid_labels}")
    if len(selected) < int(cfg["min_rows"]):
        raise DataValidationError(f"Only {len(selected)} valid rows; need {cfg['min_rows']}")
    counts = selected[label_col].value_counts()
    if (counts < 2).any():
        raise DataValidationError(f"Every label needs at least 2 rows: {counts.to_dict()}")

    report = {
        "source": str(path),
        "sha256": sha256_file(path),
        "source_rows": source_rows,
        "valid_rows": len(selected),
        "dropped_null_rows": null_rows,
        "dropped_empty_rows": empty_rows,
        "dropped_duplicate_rows": duplicate_rows,
        "class_counts": counts.sort_index().to_dict(),
    }
    return selected, report


def prepare_data(config: dict) -> dict:
    frame, report = ingest_and_validate(config)
    cfg = config["data"]
    train, test = train_test_split(
        frame,
        test_size=float(cfg["test_size"]),
        random_state=int(config["seed"]),
        stratify=frame[cfg["label_column"]],
    )
    output = resolve_path("data/processed")
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "train.csv", index=False)
    test.to_csv(output / "test.csv", index=False)
    report.update({"train_rows": len(train), "test_rows": len(test), "seed": config["seed"]})
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, clean, and split the text dataset")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    report = prepare_data(load_config(args.config))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
