"""
splits.py
=========
Creates reproducible stratified train/test and cross-validation splits
for the cleaned EMSCAD dataset.

All seeds, fold counts, and repeat counts are read from configs/seeds.yaml
-- nothing is hardcoded.

Outputs:
    data/processed/splits.json
        {
            "test_indices": [...],
            "train_cv": {
                "seed_0": {
                    "fold_0": {"train": [...], "val": [...]},
                    ...
                },
                ...
            }
        }
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "emscad_clean.csv"
SEEDS_YAML = PROJECT_ROOT / "configs" / "seeds.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_JSON = PROCESSED_DIR / "splits.json"


def load_config() -> dict:
    """Load seeds.yaml configuration."""
    with open(SEEDS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_splits() -> None:
    """Create stratified splits and save to JSON."""
    # ------------------------------------------------------------------
    # Load data and config
    # ------------------------------------------------------------------
    cfg = load_config()
    random_seed = cfg["random_seed"]
    n_folds = cfg["n_folds"]
    n_seed_repeats = cfg["n_seed_repeats"]

    print(f"[CONFIG] random_seed={random_seed}, n_folds={n_folds}, "
          f"n_seed_repeats={n_seed_repeats}")

    df = pd.read_csv(CLEAN_CSV)
    y = df["fraudulent"].values
    print(f"[INFO]  Loaded {len(df):,} rows from emscad_clean.csv")

    # ------------------------------------------------------------------
    # 1. Stratified 80/20 train/test split
    # ------------------------------------------------------------------
    all_indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=random_seed,
        stratify=y,
    )

    # Convert to plain Python ints for JSON serialisation
    train_idx_list = [int(i) for i in train_idx]
    test_idx_list = [int(i) for i in test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    train_fraud_pct = y_train.mean() * 100
    test_fraud_pct = y_test.mean() * 100

    print(f"\n{'='*60}")
    print(f"  TRAIN / TEST SPLIT")
    print(f"{'='*60}")
    print(f"  Train : {len(train_idx_list):,} rows  "
          f"(fraud: {y_train.sum():,}, {train_fraud_pct:.2f}%)")
    print(f"  Test  : {len(test_idx_list):,} rows  "
          f"(fraud: {y_test.sum():,}, {test_fraud_pct:.2f}%)")
    print(f"  Stratification check: "
          f"train {train_fraud_pct:.2f}% vs test {test_fraud_pct:.2f}% "
          f"-- {'OK' if abs(train_fraud_pct - test_fraud_pct) < 0.5 else 'DRIFT > 0.5%'}")

    # ------------------------------------------------------------------
    # 2. 5-fold stratified CV on the train portion, repeated with
    #    n_seed_repeats different seeds
    # ------------------------------------------------------------------
    # Generate deterministic seeds for each repeat from the base seed
    rng = np.random.RandomState(random_seed)
    repeat_seeds = rng.randint(0, 2**31, size=n_seed_repeats).tolist()

    train_cv: dict[str, dict] = {}
    min_fraud_in_fold = float("inf")
    min_fold_label = ""

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION ({n_folds}-fold x {n_seed_repeats} seed repeats)")
    print(f"{'='*60}")
    print(f"  Repeat seeds: {repeat_seeds}")

    for seed_i, seed_val in enumerate(repeat_seeds):
        seed_key = f"seed_{seed_i}"
        train_cv[seed_key] = {}

        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed_val
        )

        fold_fraud_counts = []
        for fold_i, (cv_train_pos, cv_val_pos) in enumerate(
            skf.split(train_idx, y_train)
        ):
            fold_key = f"fold_{fold_i}"
            # cv_train_pos / cv_val_pos are positions within train_idx
            cv_train_indices = [int(train_idx[j]) for j in cv_train_pos]
            cv_val_indices = [int(train_idx[j]) for j in cv_val_pos]

            train_cv[seed_key][fold_key] = {
                "train": cv_train_indices,
                "val": cv_val_indices,
            }

            # Track fraud counts in validation fold
            val_fraud = int(y[cv_val_indices].sum())
            fold_fraud_counts.append(val_fraud)
            if val_fraud < min_fraud_in_fold:
                min_fraud_in_fold = val_fraud
                min_fold_label = f"{seed_key}/{fold_key}"

        print(f"  {seed_key} (seed={seed_val}): "
              f"val-fold fraud counts = {fold_fraud_counts}")

    # ------------------------------------------------------------------
    # 3. Flag if any fold has fewer than 20 fraud examples
    # ------------------------------------------------------------------
    print(f"\n  Smallest fraud count in any val fold: "
          f"{min_fraud_in_fold} ({min_fold_label})")
    if min_fraud_in_fold < 20:
        print(f"  ** WARNING: fold {min_fold_label} has only "
              f"{min_fraud_in_fold} fraud examples -- "
              f"precision/recall will be unstable!")
    else:
        print(f"  OK -- all folds have >= 20 fraud examples")

    # ------------------------------------------------------------------
    # 4. Save splits.json
    # ------------------------------------------------------------------
    splits = {
        "test_indices": test_idx_list,
        "train_cv": train_cv,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_JSON, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    size_mb = SPLITS_JSON.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Saved splits -> {SPLITS_JSON} ({size_mb:.1f} MB)")
    print(f"     Total fold assignments: "
          f"{n_folds} folds x {n_seed_repeats} repeats = "
          f"{n_folds * n_seed_repeats}")


if __name__ == "__main__":
    create_splits()
