"""Audit raw data without overwriting the trained experiment's provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RAW_DATA_PATH, REPORTS_DIR
from src.data import (
    load_raw_dataset,
    clean_and_prepare_dataset,
    attach_label_availability,
)
from src.features import compute_schedule_and_calendar_features


def run_audit(data_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw_dataset(data_path)
    cleaned, profile = clean_and_prepare_dataset(raw)
    _, availability = attach_label_availability(
        compute_schedule_and_calendar_features(cleaned)
    )
    profile["historical_label_availability"] = availability
    profile["dataset_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    profile["raw_destination_count"] = int(raw.dest.nunique())
    (output_dir / "raw_dataset_audit.json").write_text(
        json.dumps(profile, indent=2) + "\n"
    )
    lines = [
        "# Dataset audit",
        "",
        f"Raw scheduled flights: **{len(raw):,}**. Observed arrival outcomes: **{len(cleaned):,}**.",
        f"Missing arrival outcomes excluded from supervision: **{profile['excluded_cancellations_or_missing_target']:,}**. Their individual exclusion reasons are not fully verified.",
        f"Target: arrival delay at least 15 minutes. Positive prevalence: **{cleaned.is_delayed.mean():.2%}**.",
        f"Duplicate flight keys: **{profile['duplicate_flight_keys']}**. Raw destinations: **{raw.dest.nunique()}**. Supervised destinations: **{cleaned.dest.nunique()}**.",
        "",
        "## Missingness",
        "",
        "| Column | Missing rows | Fraction |",
        "|---|---:|---:|",
    ]
    for col, count in profile["raw_missing_counts"].items():
        lines.append(f"| {col} | {count:,} | {count/len(raw):.2%} |")
    lines += [
        "",
        "## Historical label availability",
        "",
        pd.Series(availability).to_frame("value").to_markdown(),
        "",
        "Current-flight outcomes are never predictors. These timestamps govern release of historical labels only.",
        "The source lacks actual publication timestamps. Arrival plus 24 hours is a declared assumption.",
        "",
        "## Provenance",
        "",
        f"CSV SHA-256: `{profile['dataset_sha256']}`",
        "",
        "Source: https://www.kaggle.com/datasets/matinsajadi/flights",
        "Original schema: https://nycflights13.tidyverse.org/reference/flights.html",
        "",
        "This raw audit does not replace results/reports/dataset_profile.json, which belongs to the trained run.",
    ]
    (output_dir / "dataset_audit_report.md").write_text("\n".join(lines) + "\n")
    print(f"Audit complete: {len(raw):,} raw rows, {len(cleaned):,} observed outcomes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    run_audit(args.data_path, args.output_dir)
