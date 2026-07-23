"""Download the Home Credit Default Risk dataset via the Kaggle API.

Needs kaggle.json in ~/.kaggle or KAGGLE_CONFIG_DIR, and the competition
rules accepted on the Kaggle website. Data lands in data/raw and is never
committed.
"""

import argparse
import zipfile
from pathlib import Path

COMPETITION = "home-credit-default-risk"

EXPECTED_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
    "POS_CASH_balance.csv",
]


def download(dest: Path) -> None:
    """Pull the competition zip and extract the raw CSVs into dest."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)
    api.competition_download_files(COMPETITION, path=str(dest), quiet=False)
    archive = dest / f"{COMPETITION}.zip"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    archive.unlink()


def verify(dest: Path) -> list[str]:
    """Return the expected files that are missing from dest."""
    return [name for name in EXPECTED_FILES if not (dest / name).exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    missing = verify(args.dest)
    if not missing:
        print(f"All {len(EXPECTED_FILES)} raw files already present in {args.dest}")
        return

    download(args.dest)
    missing = verify(args.dest)
    if missing:
        raise SystemExit(f"Download finished but files are missing: {missing}")
    print(f"Downloaded {len(EXPECTED_FILES)} raw files to {args.dest}")


if __name__ == "__main__":
    main()
