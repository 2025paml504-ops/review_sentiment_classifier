"""SQLite log of every prediction the API has served, plus every simulated
prediction run through the model - the shared foundation both monitoring and
drift detection below are built on.

Without a record of what the model was actually asked and what it answered,
neither "is something wrong" (monitoring) nor "why is it wrong" (diagnosing
which text/topic drifted) is answerable after the fact. Same reasoning as
the training-side feature store (feature_store/feature_store.py): SQLite,
because it's a single file with no server to run, and it's exactly enough
for a project running on one machine.

Each row is one prediction. `source` distinguishes a real API call ("api")
from a synthetic drift-simulation run ("drift_simulation") so the two never
get mixed into the same "current production traffic" window by mistake.
"""

import datetime as dt
import uuid
from pathlib import Path

from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "monitoring" / "predictions.db"
TABLE_NAME = "predictions"


def get_engine(db_path: Path = LOG_PATH):
    """Return a SQLAlchemy engine for the SQLite prediction log, creating the folder if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def init_table(engine=None) -> None:
    """Create the predictions table if it doesn't already exist. Safe to call every time."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    cleaned_text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    oov_token_count INTEGER NOT NULL,
                    sentiment TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    prob_negative REAL NOT NULL,
                    prob_positive REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    source TEXT NOT NULL,
                    true_sentiment TEXT
                )
                """
            )
        )


def log_prediction(
    *,
    cleaned_text: str,
    token_count: int,
    oov_token_count: int,
    sentiment: str,
    confidence: float,
    probabilities: dict,
    latency_ms: float,
    model_version: str,
    source: str = "api",
    true_sentiment: str | None = None,
) -> str:
    """Append one prediction to the log. Returns the row's generated id.

    `true_sentiment` is only ever populated by the drift simulation (§ the
    hand-labeled examples come with a known correct answer) - real API
    traffic has no ground truth at request time, which is exactly the
    ground-truth-delay problem monitor.py has to work around.
    """
    engine = get_engine()
    init_table(engine)
    row_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                INSERT INTO {TABLE_NAME}
                    (id, timestamp_utc, cleaned_text, token_count, oov_token_count,
                     sentiment, confidence, prob_negative, prob_positive,
                     latency_ms, model_version, source, true_sentiment)
                VALUES
                    (:id, :ts, :text, :tokens, :oov, :sentiment, :confidence,
                     :prob_neg, :prob_pos, :latency, :version, :source, :truth)
                """
            ),
            {
                "id": row_id,
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "text": cleaned_text,
                "tokens": token_count,
                "oov": oov_token_count,
                "sentiment": sentiment,
                "confidence": confidence,
                "prob_neg": probabilities.get("NEGATIVE", 0.0),
                "prob_pos": probabilities.get("POSITIVE", 0.0),
                "latency": latency_ms,
                "version": model_version,
                "source": source,
                "truth": true_sentiment,
            },
        )
    return row_id


def read_predictions(source: str | None = None, limit: int | None = None):
    """Read logged predictions back as a DataFrame, most recent first.

    Returns an empty DataFrame (not an error) when the log doesn't exist yet
    - a freshly cloned project with no traffic yet is a normal state, not a
    failure.
    """
    import pandas as pd

    if not LOG_PATH.exists():
        return pd.DataFrame()
    engine = get_engine()
    query = f"SELECT * FROM {TABLE_NAME}"
    if source is not None:
        query += " WHERE source = :source"
    query += " ORDER BY timestamp_utc DESC"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    params = {"source": source} if source is not None else {}
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)
