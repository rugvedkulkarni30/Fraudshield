"""
preprocessing.py
=================
Cleans the raw EMSCAD dataset and saves to data/processed/emscad_clean.csv.

Steps:
  1. Deduplication (title + description hash, keep first)
  2. Missing-value handling — NO imputation / value invention
  3. Text cleaning (HTML strip, whitespace normalise, lowercase)
  4. Categorical standardisation
  5. Drop identifier column (job_id)

Design principle (Assignment 2 Part F):
  We never invent values.  Missing text gets an explicit "[MISSING]" token
  plus a binary indicator.  Missing categoricals become the literal category
  "missing".  salary_range (84% null) is reduced to a binary flag.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "emscad.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "emscad_clean.csv"

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
TEXT_FIELDS = ["company_profile", "description", "requirements", "benefits"]
CATEGORICAL_FIELDS = [
    "department", "required_education", "required_experience",
    "function", "industry", "employment_type", "location",
]

# HTML tag pattern (simple — covers all <...> tags)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Whitespace normalisation (collapse runs of whitespace into single space)
_MULTI_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _row_hash(row: pd.Series) -> str:
    """SHA-256 of title|||description for dedup."""
    combined = str(row.get("title", "")) + "|||" + str(row.get("description", ""))
    return hashlib.sha256(combined.encode("utf-8", errors="replace")).hexdigest()


def _clean_text(text: str) -> str:
    """Strip HTML, normalise whitespace, lowercase."""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text.lower()


def preprocess(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    """Run the full preprocessing pipeline and return the cleaned DataFrame."""
    df = pd.read_csv(csv_path)
    n_start = len(df)
    print(f"[INFO] Loaded {n_start:,} rows from {csv_path.name}")

    # ------------------------------------------------------------------
    # 1. Deduplication (title + description hash, keep first occurrence)
    # ------------------------------------------------------------------
    df["_hash"] = df.apply(_row_hash, axis=1)
    dup_mask = df.duplicated(subset=["_hash"], keep="first")
    n_dups = dup_mask.sum()
    df = df[~dup_mask].copy()
    df.drop(columns=["_hash"], inplace=True)
    print(f"[DEDUP] Dropped {n_dups:,} duplicate rows  "
          f"({n_start:,} -> {len(df):,})")

    # ------------------------------------------------------------------
    # 2. Missing values — no value invention
    # ------------------------------------------------------------------

    # 2a. Text fields: "[MISSING]" token + binary indicator
    for col in TEXT_FIELDS:
        indicator_col = f"{col}_missing"
        df[indicator_col] = df[col].isnull().astype(int)
        df[col] = df[col].fillna("[MISSING]")
        n_miss = df[indicator_col].sum()
        print(f"[MISSING] {col}: {n_miss:,} filled with [MISSING], "
              f"indicator -> {indicator_col}")

    # 2b. salary_range -> binary flag, then drop raw column
    df["has_salary_range"] = df["salary_range"].notna().astype(int)
    n_has = df["has_salary_range"].sum()
    print(f"[MISSING] salary_range: {n_has:,} non-null -> has_salary_range; "
          f"dropping raw column")
    df.drop(columns=["salary_range"], inplace=True)

    # 2c. Categorical fields: fill missing with literal "missing"
    for col in CATEGORICAL_FIELDS:
        n_miss = df[col].isnull().sum()
        df[col] = df[col].fillna("missing")
        if n_miss > 0:
            print(f"[MISSING] {col}: {n_miss:,} filled with 'missing' category")

    # ------------------------------------------------------------------
    # 3. Text cleaning (HTML strip, whitespace normalise, lowercase)
    #    Keep raw description copy for Streamlit demo (Stage 6)
    # ------------------------------------------------------------------
    df["description_raw"] = df["description"].copy()

    for col in TEXT_FIELDS:
        df[col] = df[col].apply(
            lambda x: _clean_text(x) if x != "[MISSING]" else x
        )
    print(f"[TEXT]  Cleaned {len(TEXT_FIELDS)} text columns "
          f"(HTML strip, whitespace norm, lowercase)")
    print(f"[TEXT]  Kept 'description_raw' for Stage 6 Streamlit demo")

    # ------------------------------------------------------------------
    # 4. Standardise categorical values
    # ------------------------------------------------------------------
    for col in ["employment_type", "required_experience", "required_education"]:
        df[col] = df[col].str.strip().str.lower()
    print("[CAT]  Standardised employment_type, required_experience, "
          "required_education (strip + lowercase)")

    # ------------------------------------------------------------------
    # 5. Drop job_id (identifier, not a feature)
    # ------------------------------------------------------------------
    df.drop(columns=["job_id"], inplace=True)
    print("[DROP] Removed job_id column")

    return df


def main() -> None:
    """Run preprocessing and save output."""
    df = preprocess()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"\n[OK] Saved {len(df):,} rows -> {CLEAN_CSV}")
    print(f"[OK] Final columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
