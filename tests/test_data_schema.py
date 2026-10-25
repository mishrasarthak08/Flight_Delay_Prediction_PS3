"""
Unit tests for data schema discovery, mapping, and population filtering.
"""

import pytest
import pandas as pd
import numpy as np

from src.data import (
    discover_and_map_schema,
    clean_and_prepare_dataset,
    generate_synthetic_dataset,
)
from src.config import TARGET_COL, RAW_TARGET_COL


def test_schema_discovery_with_standard_aliases():
    """Verifies that schema discovery resolves standard BTS and NYC flights aliases."""
    raw_df = pd.DataFrame({
        "CRSDepTime": [515, 1200],
        "CRSArrTime": [819, 1430],
        "ArrDelay": [10.0, 35.0],
        "UniqueCarrier": ["UA", "DL"],
        "ORIGIN": ["EWR", "JFK"],
        "DEST": ["ORD", "MIA"],
        "Distance": [719, 1089],
        "YEAR": [2013, 2013],
        "MONTH": [1, 1],
        "DAY_OF_MONTH": [1, 2],
    })

    mapped_df, col_map = discover_and_map_schema(raw_df)
    assert "sched_dep_time" in mapped_df.columns
    assert "sched_arr_time" in mapped_df.columns
    assert "arr_delay" in mapped_df.columns
    assert "carrier" in mapped_df.columns
    assert "origin" in mapped_df.columns
    assert "dest" in mapped_df.columns
    assert "distance" in mapped_df.columns


def test_clean_and_prepare_filters_missing_target():
    """Verifies that cancelled or diverted flights without arrival delay are excluded from target population."""
    df_synth = generate_synthetic_dataset(n_samples=200, random_seed=42)
    # Ensure at least 5 missing arr_delay values
    df_synth.loc[:4, "arr_delay"] = np.nan

    cleaned_df, audit = clean_and_prepare_dataset(df_synth)
    assert cleaned_df[TARGET_COL].isna().sum() == 0
    assert audit["excluded_cancellations_or_missing_target"] >= 5
    assert len(cleaned_df) == len(df_synth) - audit["excluded_cancellations_or_missing_target"]
    assert set(cleaned_df[TARGET_COL].unique()).issubset({0, 1})
