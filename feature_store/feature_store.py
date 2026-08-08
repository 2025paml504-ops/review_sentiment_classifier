"""Persist engineered hotel-review features to a SQLite feature store."""

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from features.build_features import INTERIM_PATH, build_features, write_interim

# Repo root is two levels up from this file (feature_store/feature_store.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_STORE_PATH = REPO_ROOT / "feature_store" / "feature_store.db"
TABLE_NAME = "hotel_review_features"


def get_engine(db_path=FEATURE_STORE_PATH):
    """Return a SQLAlchemy engine for the SQLite feature store."""
    return create_engine(f"sqlite:///{db_path}")


def save_features(df: pd.DataFrame, table: str = TABLE_NAME, if_exists: str = "replace") -> int:
    """Write the feature DataFrame to the feature store. Returns rows written."""
    engine = get_engine()
    df.to_sql(table, engine, if_exists=if_exists, index=False)
    return len(df)


def load_features(table: str = TABLE_NAME) -> pd.DataFrame:
    """Read the feature table back from the feature store."""
    engine = get_engine()
    return pd.read_sql_table(table, engine)


def load_interim(path=INTERIM_PATH) -> pd.DataFrame:
    """Load the cleaned/tokenized interim CSV, generating it first if missing."""
    if not path.exists():
        write_interim(build_features(), path)
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_interim()
    rows = save_features(df)
    print(f"Wrote {rows} rows to table '{TABLE_NAME}' at {FEATURE_STORE_PATH}")
