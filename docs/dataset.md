# Dataset

[← Back to README](../README.md) · Related: [Pipeline](pipeline.md) · [Versioning](versioning.md)

## Source

- **Kaggle — "515K Hotel Reviews Data in Europe"**: ~515,738 Booking.com reviews
  across European hotels (17 raw columns).
- Stored locally at `data/raw/Hotel_Reviews.csv` (~238 MB).
- The raw file is **large and not committed to git**. It is tracked by DVC
  (`data/raw/Hotel_Reviews.csv.dvc`). Get it one of two ways:
  - `dvc pull` if you have access to the DVC remote (see [Versioning](versioning.md)), or
  - download it from Kaggle and place it at `data/raw/Hotel_Reviews.csv`.

## Data layers

All data lives under `data/`, organized into three layers:

| Layer            | Path              | Role                                                    |
|------------------|-------------------|---------------------------------------------------------|
| **raw**          | `data/raw/`       | Immutable source (`Hotel_Reviews.csv`) — never edit     |
| **interim**      | `data/interim/`   | Cleaned/labeled, rebuildable (`features_clean.csv`)     |
| **processed**    | `data/processed/` | Model-ready splits (`train_v1.csv`)                     |

`interim/` and `processed/` are **outputs of the pipeline** — always rebuildable
from `raw/` via `dvc repro`. Never hand-edit them; change the code and re-run.
