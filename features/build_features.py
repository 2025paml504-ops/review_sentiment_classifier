"""Load the raw hotel reviews dataset, clean/tokenize it, and write to data/interim."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root is two levels up from this file (features/build_features.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "Hotel_Reviews.csv"
INTERIM_PATH = REPO_ROOT / "data" / "interim" / "features_clean.csv"

# Placeholder phrases the dataset uses when a reviewer left that side blank.
_PLACEHOLDER_RE = re.compile(r"no positive|no negative")
# Anything that is not an ASCII letter or whitespace (digits, punctuation, specials).
_NON_LETTER_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")

# Sentiment label thresholds on Reviewer_Score (Scheme A): NEG < 6, 6 <= NEU < 8, POS >= 8.
SENTIMENT_NEUTRAL_FLOOR = 6.0
SENTIMENT_POSITIVE_FLOOR = 8.0
SENTIMENT_LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# Metadata columns that carry no signal for sentiment classification.
COLUMNS_TO_DROP = [
    "Hotel_Address",
    "Additional_Number_of_Scoring",
    "Review_Date",
    "Average_Score",
    "Hotel_Name",
    "Reviewer_Nationality",
    "Total_Number_of_Reviews",
    "Total_Number_of_Reviews_Reviewer_Has_Given",
    "Tags",
    "days_since_review",
    "lat",
    "lng",
]


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw hotel reviews CSV into a DataFrame."""
    return pd.read_csv(path)


def drop_columns(df: pd.DataFrame, columns=COLUMNS_TO_DROP) -> pd.DataFrame:
    """Drop the given columns, ignoring any that are missing."""
    return df.drop(columns=columns, errors="ignore")


def combine_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `full_review` column: positive and negative review text combined."""
    df = df.copy()
    df["full_review"] = (
        df["Positive_Review"].fillna("") + " " + df["Negative_Review"].fillna("")
    ).str.strip()
    return df


def clean_text(text: str) -> str:
    """Lowercase, strip placeholder phrases, drop non-letters, and collapse whitespace."""
    text = str(text).lower()
    text = _PLACEHOLDER_RE.sub(" ", text)
    text = _NON_LETTER_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split cleaned text into tokens on whitespace."""
    return text.split()


def clean_and_tokenize(df: pd.DataFrame) -> pd.DataFrame:
    """Add `clean_review` (cleaned text) and `tokens` (JSON-encoded token list)."""
    df = df.copy()
    df["clean_review"] = df["full_review"].map(clean_text)
    df["tokens"] = df["clean_review"].map(lambda t: json.dumps(tokenize(t)))
    return df


def label_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `sentiment` label derived from Reviewer_Score (Scheme A thresholds)."""
    df = df.copy()
    df["sentiment"] = pd.cut(
        df["Reviewer_Score"],
        bins=[-np.inf, SENTIMENT_NEUTRAL_FLOOR, SENTIMENT_POSITIVE_FLOOR, np.inf],
        labels=SENTIMENT_LABELS,
        right=False,
    ).astype(str)
    return df


def build_features(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw data, drop unused columns, combine reviews, clean/tokenize, and label."""
    df = load_raw_data(path)
    df = drop_columns(df)
    df = combine_reviews(df)
    df = clean_and_tokenize(df)
    return label_sentiment(df)


def write_interim(df: pd.DataFrame, path=INTERIM_PATH) -> Path:
    """Write the cleaned/tokenized features to the interim CSV; return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    df = build_features()
    out = write_interim(df)
    print("Shape:", df.shape)
    print("Wrote interim:", out)
    print("Sample clean_review:", df["clean_review"].iloc[0])
