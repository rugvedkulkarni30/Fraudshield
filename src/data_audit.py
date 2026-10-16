"""
data_audit.py
=============
Loads  data/raw/emscad.csv  and produces a comprehensive audit report
covering schema validation, class balance, missing-value rates, duplicate
detection, and text-length statistics — then writes the report to
data/processed/audit_report.txt.
"""

from __future__ import annotations

import hashlib
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "emscad.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "audit_report.txt"

# ---------------------------------------------------------------------------
# Expected schema (from EMSCAD / Kaggle listing)
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "job_id", "title", "location", "department", "salary_range",
    "company_profile", "description", "requirements", "benefits",
    "telecommuting", "has_company_logo", "has_questions",
    "employment_type", "required_experience", "required_education",
    "industry", "function", "fraudulent",
]
EXPECTED_ROW_COUNT = 17_880
EXPECTED_FRAUD_COUNT = 866
EXPECTED_GENUINE_COUNT = 17_014


def _section(title: str) -> str:
    """Return a formatted section header."""
    bar = "=" * 60
    return f"\n{bar}\n  {title}\n{bar}\n"


def audit(csv_path: Path = RAW_CSV) -> str:
    """Run the full audit and return the report as a string."""
    buf = StringIO()

    def p(text: str = "") -> None:
        buf.write(text + "\n")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    if not csv_path.exists():
        p(f"[ERROR] CSV not found: {csv_path}")
        p("Run  data_acquisition.py  first.")
        return buf.getvalue()

    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------------
    # 1. Row count & column list
    # ------------------------------------------------------------------
    p(_section("1. SCHEMA VALIDATION"))
    n_rows, n_cols = df.shape
    p(f"Rows    : {n_rows:,}  (expected {EXPECTED_ROW_COUNT:,})")
    row_ok = "OK" if n_rows == EXPECTED_ROW_COUNT else "MISMATCH"
    p(f"          {row_ok}")
    p(f"Columns : {n_cols}")
    p()

    actual_cols = list(df.columns)
    missing_cols = set(EXPECTED_COLUMNS) - set(actual_cols)
    extra_cols = set(actual_cols) - set(EXPECTED_COLUMNS)

    if not missing_cols and not extra_cols:
        p("Column check: OK -- all expected columns present, no extras.")
    else:
        if missing_cols:
            p(f"Missing columns: {sorted(missing_cols)}")
        if extra_cols:
            p(f"Extra columns  : {sorted(extra_cols)}")

    p("\nColumn listing:")
    for i, col in enumerate(actual_cols, 1):
        p(f"  {i:>2}. {col}")

    # ------------------------------------------------------------------
    # 2. Class balance
    # ------------------------------------------------------------------
    p(_section("2. CLASS BALANCE (fraudulent)"))
    if "fraudulent" not in df.columns:
        p("[WARN] 'fraudulent' column not found -- skipping class balance.")
    else:
        counts = df["fraudulent"].value_counts().sort_index()
        total = len(df)
        for label, cnt in counts.items():
            pct = cnt / total * 100
            p(f"  fraudulent={label}  ->  {cnt:>6,}  ({pct:5.2f}%)")

        fraud_n = counts.get(1, 0)
        genuine_n = counts.get(0, 0)
        p()
        p(f"  Expected fraud   : {EXPECTED_FRAUD_COUNT:,}  | Actual: {fraud_n:,}  "
          f"{'OK' if fraud_n == EXPECTED_FRAUD_COUNT else 'MISMATCH'}")
        p(f"  Expected genuine : {EXPECTED_GENUINE_COUNT:,} | Actual: {genuine_n:,} "
          f"{'OK' if genuine_n == EXPECTED_GENUINE_COUNT else 'MISMATCH'}")
        p(f"  Imbalance ratio  : 1 : {genuine_n / max(fraud_n, 1):.1f}")

    # ------------------------------------------------------------------
    # 3. Missing-value rate per column
    # ------------------------------------------------------------------
    p(_section("3. MISSING VALUES"))
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"missing_count": miss, "missing_%": miss_pct})
    miss_df = miss_df.sort_values("missing_%", ascending=False)
    p(miss_df.to_string())

    # ------------------------------------------------------------------
    # 4. Exact duplicate rows (by title + description hash)
    # ------------------------------------------------------------------
    p(_section("4. DUPLICATE DETECTION (title + description hash)"))
    text_cols = ["title", "description"]
    if all(c in df.columns for c in text_cols):
        def _row_hash(row: pd.Series) -> str:
            combined = str(row.get("title", "")) + "|||" + str(row.get("description", ""))
            return hashlib.sha256(combined.encode("utf-8", errors="replace")).hexdigest()

        df["_hash"] = df.apply(_row_hash, axis=1)
        dup_mask = df.duplicated(subset=["_hash"], keep=False)
        n_dup_rows = dup_mask.sum()
        n_dup_groups = df.loc[dup_mask, "_hash"].nunique()
        p(f"  Rows involved in duplicates : {n_dup_rows:,}")
        p(f"  Distinct duplicate groups   : {n_dup_groups:,}")
        df.drop(columns=["_hash"], inplace=True)
    else:
        p("  [WARN] 'title' and/or 'description' column missing -- skipping.")

    # ------------------------------------------------------------------
    # 5. Text-length stats for 'description', split by label
    # ------------------------------------------------------------------
    p(_section("5. TEXT-LENGTH STATISTICS (description)"))
    if "description" in df.columns and "fraudulent" in df.columns:
        df["_desc_len"] = df["description"].fillna("").str.len()
        for label in sorted(df["fraudulent"].unique()):
            subset = df.loc[df["fraudulent"] == label, "_desc_len"]
            tag = "FRAUD" if label == 1 else "GENUINE"
            p(f"  [{tag}] (fraudulent={label}, n={len(subset):,})")
            p(f"    mean   : {subset.mean():,.1f} chars")
            p(f"    median : {subset.median():,.1f} chars")
            p(f"    std    : {subset.std():,.1f} chars")
            p(f"    min    : {subset.min():,}")
            p(f"    max    : {subset.max():,}")
            p()
        df.drop(columns=["_desc_len"], inplace=True)
    else:
        p("  [WARN] 'description' and/or 'fraudulent' column missing -- skipping.")

    return buf.getvalue()


def main() -> None:
    """Generate and save the audit report."""
    report = audit()

    # Print to console
    print(report)

    # Save to file
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n[OK] Report saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
