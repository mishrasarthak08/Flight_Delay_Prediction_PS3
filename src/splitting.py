"""
Chronological dataset splitting module.
Enforces date-level temporal splitting (all flights from a given date belong to the same partition)
and generates reproducible split manifests.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.config import SplittingConfig, TARGET_COL, RESULTS_DIR


def chronological_date_split(
    df: pd.DataFrame,
    date_col: str = "flight_date",
    config: SplittingConfig = SplittingConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Splits the dataset chronologically based on calendar flight dates.
    Earliest ~70% dates -> Train,
    Next ~15% dates -> Validation,
    Final ~15% dates -> Test.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe.")
    ratios = [config.train_ratio, config.val_ratio, config.test_ratio]
    if any(r <= 0 for r in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must be positive and sum to one.")

    # Unique sorted dates
    dates = np.array(sorted(pd.to_datetime(df[date_col]).dt.date.unique()))
    total_dates = len(dates)
    if total_dates < 3:
        raise ValueError("At least three distinct flight dates are required.")

    n_train_dates = int(np.round(total_dates * config.train_ratio))
    n_val_dates = int(np.round(total_dates * config.val_ratio))
    # Test gets the rest to ensure exact coverage of all dates
    if min(n_train_dates, n_val_dates, total_dates - n_train_dates - n_val_dates) < 1:
        raise ValueError("Each chronological partition must contain at least one date.")

    train_date_set = set(dates[:n_train_dates])
    val_date_set = set(dates[n_train_dates : n_train_dates + n_val_dates])
    test_date_set = set(dates[n_train_dates + n_val_dates :])

    df_dates = pd.to_datetime(df[date_col]).dt.date
    train_mask = df_dates.isin(train_date_set)
    val_mask = df_dates.isin(val_date_set)
    test_mask = df_dates.isin(test_date_set)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    # Generate split manifest
    manifest: Dict[str, Any] = {
        "total_unique_dates": total_dates,
        "train": {
            "num_dates": len(train_date_set),
            "start_date": min(train_date_set).strftime("%Y-%m-%d"),
            "end_date": max(train_date_set).strftime("%Y-%m-%d"),
            "row_count": len(train_df),
            "positive_count": (
                int(train_df[TARGET_COL].sum()) if TARGET_COL in train_df else None
            ),
            "positive_prevalence": (
                float(train_df[TARGET_COL].mean()) if TARGET_COL in train_df else None
            ),
            "date_proportion": len(train_date_set) / total_dates,
            "row_proportion": len(train_df) / len(df),
        },
        "validation": {
            "num_dates": len(val_date_set),
            "start_date": min(val_date_set).strftime("%Y-%m-%d"),
            "end_date": max(val_date_set).strftime("%Y-%m-%d"),
            "row_count": len(val_df),
            "positive_count": (
                int(val_df[TARGET_COL].sum()) if TARGET_COL in val_df else None
            ),
            "positive_prevalence": (
                float(val_df[TARGET_COL].mean()) if TARGET_COL in val_df else None
            ),
            "date_proportion": len(val_date_set) / total_dates,
            "row_proportion": len(val_df) / len(df),
        },
        "test": {
            "num_dates": len(test_date_set),
            "start_date": min(test_date_set).strftime("%Y-%m-%d"),
            "end_date": max(test_date_set).strftime("%Y-%m-%d"),
            "row_count": len(test_df),
            "positive_count": (
                int(test_df[TARGET_COL].sum()) if TARGET_COL in test_df else None
            ),
            "positive_prevalence": (
                float(test_df[TARGET_COL].mean()) if TARGET_COL in test_df else None
            ),
            "date_proportion": len(test_date_set) / total_dates,
            "row_proportion": len(test_df) / len(df),
        },
        "split_logic": "Strict chronological date partition. All flights from any given date are assigned to the same partition.",
        "random_seed": config.random_seed,
    }

    return train_df, val_df, test_df, manifest


def save_split_manifest(
    manifest: Dict[str, Any],
    output_path: Path = RESULTS_DIR / "reports" / "split_manifest.json",
) -> None:
    """Saves the chronological split manifest to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
