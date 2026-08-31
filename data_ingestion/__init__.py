import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("data_ingestion")

KAGGLE_DATASET = "jiashenliu/515k-hotel-reviews-data-in-europe"
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "Hotel_Reviews.csv"

def download(dataset: str = KAGGLE_DATASET, path: Path = RAW_DATA_PATH) -> Path:
    if path.exists():
        logger.info("Already present: %s", path)
        return path

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    logger.info("Downloading %s to %s", dataset, path)
    api.dataset_download_files(dataset, path.parent, unzip=True)
    return path

if __name__ == "__main__":
    download()