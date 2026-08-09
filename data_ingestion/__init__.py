from pathlib import Path

KAGGLE_DATASET = "jiashenliu/515k-hotel-reviews-data-in-europe"
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "Hotel_Reviews.csv"

def download(dataset: str = KAGGLE_DATASET, path: Path = RAW_DATA_PATH) -> Path:
    if path.exists():
        print(f"Already present: {path}")
        return path

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {dataset} to {path}")
    api.dataset_download_files(dataset, path.parent, unzip=True)
    return path

if __name__ == "__main__":
    download()