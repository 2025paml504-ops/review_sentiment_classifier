from pathlib import Path

import pandas as pd
import pytest

from text_classifier.data import DataValidationError, ingest_and_validate


def config_for(path: Path):
    return {
        "data": {
            "path": str(path),
            "text_column": "text",
            "label_column": "label",
            "allowed_labels": ["a", "b"],
            "min_rows": 2,
        }
    }


def test_validation_rejects_wrong_schema(tmp_path):
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"feature": [1], "target": [2]}).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="Missing required columns"):
        ingest_and_validate(config_for(path))


def test_validation_cleans_and_deduplicates(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "text": [" hello ", "hello", "second hello", "world", "second world"],
            "label": ["a", "a", "a", "b", "b"],
        }
    ).to_csv(path, index=False)
    frame, report = ingest_and_validate(config_for(path))
    assert len(frame) == 4
    assert report["dropped_duplicate_rows"] == 1
