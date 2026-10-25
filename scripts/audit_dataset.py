"""
Dataset Audit Script.
Inspects raw flight data, schema mappings, missingness, cancellations, target prevalence,
and produces comprehensive machine-readable and markdown audit reports.
"""

import json
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_PATH, REPORTS_DIR
from src.data import load_raw_dataset, clean_and_prepare_dataset


def run_audit(data_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading raw dataset from {data_path}...")
    df_raw = load_raw_dataset(data_path)

    print("Auditing schema and cleaning population...")
    df_cleaned, audit_stats = clean_and_prepare_dataset(df_raw)

    # Missing value analysis on raw dataframe
    missing_analysis = {col: int(df_raw[col].isna().sum()) for col in df_raw.columns}
    audit_stats["raw_missing_counts"] = missing_analysis

    # Save JSON profile
    json_path = output_dir / "dataset_profile.json"
    with open(json_path, "w") as f:
        json.dump(audit_stats, f, indent=2)
    print(f"Saved machine-readable profile to {json_path}")

    # Generate Markdown report
    md_content = f"""# Flight Delay Prediction - Dataset Audit Report

## 1. Executive Summary
- **Source File**: `{data_path.name}`
- **Total Raw Records**: {audit_stats['initial_raw_rows']:,}
- **Total Raw Columns**: {audit_stats['initial_raw_cols']}
- **Operated Flights with Observed Outcome**: {audit_stats['operated_flights_count']:,} ({audit_stats['operated_flights_count'] / audit_stats['initial_raw_rows'] * 100:.2f}%)
- **Excluded Flights (Cancellations & Diversions)**: {audit_stats['excluded_cancellations_or_missing_target']:,} ({audit_stats['excluded_cancellations_or_missing_target'] / audit_stats['initial_raw_rows'] * 100:.2f}%)

## 2. Operational Prediction Contract
- **Declared Prediction Timestamp**: 2 hours before scheduled departure (`CRSDepTime` / `sched_dep_time`).
- **Target Variable**: `{audit_stats['target_column']}` (Binary: 1 if `{audit_stats['raw_target_column']}` >= {audit_stats['delay_threshold_minutes']} mins, else 0).
- **Delayed Flights (Target = 1)**: {audit_stats['delayed_flights_count']:,}
- **On-Time Flights (Target = 0)**: {audit_stats['on_time_flights_count']:,}
- **Positive Class Prevalence**: **{audit_stats['positive_class_prevalence'] * 100:.2f}%**

## 3. Date & Geographic Coverage
- **Temporal Range**: {audit_stats['date_min']} to {audit_stats['date_max']} ({audit_stats['unique_flight_dates']} unique calendar dates)
- **Origin Airports ({audit_stats['unique_origins']})**: {', '.join(audit_stats['origin_list'])}
- **Unique Destination Airports**: {audit_stats['unique_destinations']}
- **Unique Airline Carriers ({audit_stats['unique_carriers']})**: {', '.join(audit_stats['carrier_list'])}
- **Unique Operational Routes**: {audit_stats['unique_routes']:,}

## 4. Missing Value Breakdown (Raw Data)
| Column Name | Missing Count | Missing Percentage |
| :--- | :--- | :--- |
"""
    for col, count in missing_analysis.items():
        pct = (count / audit_stats['initial_raw_rows']) * 100.0
        md_content += f"| `{col}` | {count:,} | {pct:.2f}% |\n"

    md_content += """
## 5. Leakage Prevention and Column Classification
- **Allowed Pre-Departure Features**: Scheduled departure/arrival hour and minute, cyclical transformations, distance, route, airline carrier, origin, destination, calendar date indicators, and past-only smoothed historical delay rates.
- **Prohibited Outcome Fields**: `dep_time`, `dep_delay`, `arr_time`, `arr_delay`, `air_time`, `taxi_out`, `taxi_in`, `wheels_off`, `wheels_on`, delay causal breakdowns. All post-departure and current-flight outcome fields are strictly excluded from model feature matrices.
"""

    md_path = output_dir / "dataset_audit_report.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Saved human-readable audit report to {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit flights dataset")
    parser.add_argument("--data-path", type=str, default=str(RAW_DATA_PATH), help="Path to raw CSV")
    parser.add_argument("--output-dir", type=str, default=str(REPORTS_DIR), help="Output directory for reports")
    args = parser.parse_args()

    run_audit(Path(args.data_path), Path(args.output_dir))
