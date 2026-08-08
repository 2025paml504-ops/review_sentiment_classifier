"""Validate the hotel-review features before they feed the model.

Runs schema checks, placeholder detection, a score-floor sanity check, and an
empty-`full_review` warning. Hard errors (schema/score) fail validation;
placeholder counts and empty reviews are reported as warnings only.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

from features.build_features import SENTIMENT_LABELS, build_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("validate_data")

# Declarative feature schema lives in JSON so the column contract is data, not code.
FEATURE_COLUMN_PATH = Path(__file__).resolve().parent / "feature_column.json"

# dtype name (as written in feature_column.json) -> pandas dtype-kind predicate.
DTYPE_PREDICATES = {
    "string": pd.api.types.is_string_dtype,
    "integer": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_float_dtype,
}


def load_expected_columns(path=FEATURE_COLUMN_PATH) -> dict:
    """Load the feature schema JSON into a {column: dtype-predicate} map."""
    with open(path) as f:
        data = json.load(f)
    expected = {}
    for col, kind in data["columns"].items():
        if kind not in DTYPE_PREDICATES:
            raise ValueError(f"Unknown dtype '{kind}' for column '{col}' in {path}")
        expected[col] = DTYPE_PREDICATES[kind]
    return expected


# Expected column -> dtype-kind predicate, loaded from feature_column.json.
EXPECTED_COLUMNS = load_expected_columns()

SCORE_FLOOR = 2.5
SCORE_CEIL = 10.0

# Placeholder text the dataset uses when a reviewer left that side blank.
PLACEHOLDERS = {"Positive_Review": "No Positive", "Negative_Review": "No Negative"}


def check_schema(df: pd.DataFrame) -> list[str]:
    """Every expected column present with the right dtype; no unexpected columns."""
    issues = []
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    for col in missing:
        issues.append(f"missing expected column '{col}'")

    unexpected = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    for col in unexpected:
        issues.append(f"unexpected column '{col}'")

    for col, is_kind in EXPECTED_COLUMNS.items():
        if col in df.columns and not is_kind(df[col]):
            issues.append(
                f"column '{col}' has dtype {df[col].dtype}, which fails {is_kind.__name__}"
            )
    return issues


def check_placeholders(df: pd.DataFrame) -> list[str]:
    """Report placeholder-text counts (informational, not a hard failure)."""
    issues = []
    for col, placeholder in PLACEHOLDERS.items():
        if col in df.columns:
            count = int((df[col].astype("string").str.strip() == placeholder).sum())
            if count:
                logger.warning("%d rows have placeholder %r in '%s'", count, placeholder, col)
    return issues


def check_score_floor(df: pd.DataFrame) -> list[str]:
    """Flag Reviewer_Score values outside [SCORE_FLOOR, SCORE_CEIL] or NaN."""
    issues = []
    if "Reviewer_Score" not in df.columns:
        return issues
    scores = df["Reviewer_Score"]
    nan_count = int(scores.isna().sum())
    if nan_count:
        issues.append(f"{nan_count} rows have a missing Reviewer_Score")
    below = int((scores < SCORE_FLOOR).sum())
    above = int((scores > SCORE_CEIL).sum())
    if below:
        issues.append(f"{below} rows have Reviewer_Score below floor {SCORE_FLOOR}")
    if above:
        issues.append(f"{above} rows have Reviewer_Score above ceiling {SCORE_CEIL}")
    return issues


def check_empty_full_review(df: pd.DataFrame) -> list[str]:
    """Warn (not fail) when any full_review is empty after stripping."""
    if "full_review" not in df.columns:
        return []
    empty = int((df["full_review"].astype("string").str.strip() == "").sum())
    if empty:
        logger.warning("%d rows have an empty full_review", empty)
    return []


def check_sentiment(df: pd.DataFrame) -> list[str]:
    """Ensure sentiment holds only known labels; log the class distribution."""
    issues = []
    if "sentiment" not in df.columns:
        return issues
    valid = set(SENTIMENT_LABELS)
    counts = df["sentiment"].value_counts(dropna=False)
    unknown = int(counts[[lbl for lbl in counts.index if lbl not in valid]].sum())
    if unknown:
        issues.append(f"{unknown} rows have a sentiment outside {sorted(valid)}")
    total = len(df)
    for label in SENTIMENT_LABELS:
        n = int(counts.get(label, 0))
        logger.info("sentiment %s: %d (%.1f%%)", label, n, 100 * n / total if total else 0)
    return issues


def validate(df: pd.DataFrame) -> bool:
    """Run all checks. Return True if no hard errors were found."""
    errors = check_schema(df) + check_score_floor(df) + check_sentiment(df)
    # Warning-level checks (do not affect pass/fail).
    check_placeholders(df)
    check_empty_full_review(df)

    for err in errors:
        logger.error(err)

    if errors:
        logger.error("Validation FAILED with %d error(s)", len(errors))
        return False
    logger.info("Validation PASSED (%d rows checked)", len(df))
    return True


if __name__ == "__main__":
    df = build_features()
    ok = validate(df)
    sys.exit(0 if ok else 1)
