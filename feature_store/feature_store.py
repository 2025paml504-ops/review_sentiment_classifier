"""Persist engineered hotel-review features to a SQLite feature store."""

# Added (v1.2): needed for run snapshotting
import datetime as dt
import hashlib
import sqlite3
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from features.build_features import INTERIM_PATH, build_features, write_interim

# Repo root is two levels up from this file (feature_store/feature_store.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_STORE_PATH = REPO_ROOT / "feature_store" / "feature_store.db"
TABLE_NAME = "hotel_review_features"
# The label column whose distribution is worth carrying in a run snapshot.
LABEL_COLUMN = "sentiment"
# The store is ~0.5 GB, far too large to attach to a run, so a run logs this
# metadata *snapshot* instead: enough to prove which feature rows it read.
HASH_CHUNK = 1 << 20


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


def file_digest(path: Path, chunk: int = HASH_CHUNK) -> str:
    """SHA-256 of a (possibly large) file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(db_path: Path = FEATURE_STORE_PATH, table: str = TABLE_NAME) -> dict | None:
    """Describe the feature store as it stands: identity, shape and label mix.

    Returns None when the store has not been built yet, so tracking a run never
    fails on a missing feature store.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return None

    stat = db_path.stat()
    snap = {
        "path": db_path.relative_to(REPO_ROOT).as_posix(),
        "table": table,
        "size_bytes": stat.st_size,
        "modified_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
        "sha256": file_digest(db_path),
    }

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        columns = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        if not columns:
            snap["error"] = f"table '{table}' not found"
            return snap
        snap["columns"] = {row[1]: row[2] for row in columns}
        snap["rows"] = int(conn.execute(f"SELECT COUNT(*) FROM '{table}'").fetchone()[0])
        if LABEL_COLUMN in snap["columns"]:
            counts = conn.execute(
                f"SELECT {LABEL_COLUMN}, COUNT(*) FROM '{table}' GROUP BY {LABEL_COLUMN}"
            ).fetchall()
            snap["label_distribution"] = {str(k): int(v) for k, v in counts}
    return snap


def load_interim(path=INTERIM_PATH) -> pd.DataFrame:
    """Load the cleaned/tokenized interim CSV, generating it first if missing."""
    if not path.exists():
        write_interim(build_features(), path)
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_interim()
    rows = save_features(df)
    print(f"Wrote {rows} rows to table '{TABLE_NAME}' at {FEATURE_STORE_PATH}")
