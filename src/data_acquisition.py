"""
data_acquisition.py
====================
Downloads the EMSCAD dataset (Real or Fake Job Posting Prediction) from
Kaggle into  data/raw/emscad.csv.

Requires:
    - KAGGLE_USERNAME and KAGGLE_KEY environment variables, **or**
    - A valid ~/.kaggle/kaggle.json token file.

If the automated download fails the script prints step-by-step manual
download instructions.
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_SLUG = "shivamb/real-or-fake-fake-jobposting-prediction"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TARGET_CSV = RAW_DIR / "emscad.csv"

# The CSV filename inside the Kaggle archive
EXPECTED_CSV_NAME = "fake_job_postings.csv"


def _print_manual_instructions() -> None:
    """Print clear manual-download instructions."""
    print(
        "\n"
        "+--------------------------------------------------------------+\n"
        "|              MANUAL DOWNLOAD INSTRUCTIONS                    |\n"
        "+--------------------------------------------------------------+\n"
        "| 1. Go to:                                                    |\n"
        "|    https://www.kaggle.com/datasets/shivamb/                  |\n"
        "|    real-or-fake-fake-jobposting-prediction                   |\n"
        "|                                                              |\n"
        "| 2. Click 'Download' (you need a free Kaggle account).        |\n"
        "|                                                              |\n"
        "| 3. Extract the ZIP. You should get 'fake_job_postings.csv'.  |\n"
        "|                                                              |\n"
        f"| 4. Copy/move it to:                                          |\n"
        f"|    {TARGET_CSV}\n"
        "|                                                              |\n"
        "| To configure Kaggle API credentials instead, see:            |\n"
        "|    https://github.com/Kaggle/kaggle-api#api-credentials      |\n"
        "|                                                              |\n"
        "| Quick setup:                                                 |\n"
        "|   set KAGGLE_USERNAME=<your_username>                        |\n"
        "|   set KAGGLE_KEY=<your_api_key>                              |\n"
        "|   (or place kaggle.json in ~/.kaggle/)                       |\n"
        "+--------------------------------------------------------------+\n"
    )


def download_dataset() -> Path:
    """Download the dataset and return the path to the saved CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_CSV.exists():
        print(f"[INFO] Dataset already present at {TARGET_CSV} -- skipping download.")
        return TARGET_CSV

    # ------------------------------------------------------------------
    # Attempt 1: kagglehub (preferred, newer API)
    # ------------------------------------------------------------------
    try:
        import kagglehub

        print("[INFO] Downloading via kagglehub ...")
        downloaded_path = Path(kagglehub.dataset_download(DATASET_SLUG))
        print(f"[INFO] kagglehub downloaded to: {downloaded_path}")

        # kagglehub returns a directory; find the CSV inside it
        candidates = list(downloaded_path.rglob(EXPECTED_CSV_NAME))
        if not candidates:
            # Fall back to any CSV
            candidates = list(downloaded_path.rglob("*.csv"))

        if candidates:
            shutil.copy2(candidates[0], TARGET_CSV)
            print(f"[OK]   Saved dataset -> {TARGET_CSV}")
            return TARGET_CSV
        else:
            print("[WARN] kagglehub download succeeded but CSV not found in output.")

    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] kagglehub download failed: {exc}")

    # ------------------------------------------------------------------
    # Attempt 2: kaggle CLI (legacy API)
    # ------------------------------------------------------------------
    try:
        import subprocess

        print("[INFO] Trying kaggle CLI ...")
        subprocess.check_call(
            [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", DATASET_SLUG,
                "-p", str(RAW_DIR),
                "--unzip",
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        src = RAW_DIR / EXPECTED_CSV_NAME
        if src.exists():
            src.rename(TARGET_CSV)
            print(f"[OK]   Saved dataset -> {TARGET_CSV}")
            return TARGET_CSV
        # Maybe it was saved with a different name
        csvs = list(RAW_DIR.glob("*.csv"))
        if csvs:
            csvs[0].rename(TARGET_CSV)
            print(f"[OK]   Saved dataset -> {TARGET_CSV}")
            return TARGET_CSV

        print("[WARN] kaggle CLI ran but CSV not found.")

    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] kaggle CLI failed: {exc}")

    # ------------------------------------------------------------------
    # Both methods failed — show manual instructions
    # ------------------------------------------------------------------
    _print_manual_instructions()
    sys.exit(1)


if __name__ == "__main__":
    download_dataset()
