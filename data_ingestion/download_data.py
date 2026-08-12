"""Kaggle download entrypoint for the raw hotel-reviews CSV (added 11-Aug - Ankita).

Thin CLI wrapper around data_ingestion.download() -- kept as its own module so
it matches the rest of the pipeline's "one module per stage, runnable from the
repo root as python -m <package>.<module>" convention (see docs/contributing.md).
Not a DVC stage itself: it produces data/raw/Hotel_Reviews.csv, which
build_features consumes, but `dvc repro` expects that file to already exist
(see docs/dataset.md) or be restored via `dvc pull`.

    python -m data_ingestion.download_data
    python -m data_ingestion.download_data --dataset <owner>/<dataset> --path <dest.csv>
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from data_ingestion import KAGGLE_DATASET, RAW_DATA_PATH, download

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("download_data")


def report(path: Path) -> None:
    """Log size/shape so a successful download is visible, not silent."""
    size_mb = path.stat().st_size / (1024 * 1024)
    df = pd.read_csv(path)
    logger.info("%s: %.1f MB, %d rows, %d columns", path, size_mb, *df.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=KAGGLE_DATASET, help="Kaggle <owner>/<dataset> slug")
    parser.add_argument("--path", type=Path, default=RAW_DATA_PATH, help="destination CSV path")
    args = parser.parse_args()

    path = download(args.dataset, args.path)
    report(path)


if __name__ == "__main__":
    main()
