"""
Future-label mutation test and forbidden column leakage verification.
PROVES that modifying future flight labels cannot change any earlier row's historical features.
"""

import pytest
import numpy as np
import pandas as pd

from src.features import HistoricalFeatureGenerator, compute_schedule_and_calendar_features
from src.config import SmoothingConfig, TARGET_COL, FORBIDDEN_COLUMNS
from src.validation import audit_feature_matrix_for_forbidden_columns, DataLeakageError


def test_future_label_mutation_invariance():
    """
    CRITICAL TEST: Proves that altering outcomes on future dates (d >= D)
    has ZERO effect on historical feature values calculated for earlier dates (d < D).
    """
    # Create multi-day synthetic flight records
    dates = pd.date_range("2013-01-01", periods=10, freq="D")
    records = []
    carriers = ["UA", "AA", "DL"]
    origins = ["EWR", "JFK"]
    dests = ["ORD", "MIA"]

    for d in dates:
        for _ in range(20):
            records.append({
                "flight_date": d,
                "sched_dep_time": 900,
                "sched_arr_time": 1200,
                "carrier": np.random.choice(carriers),
                "origin": np.random.choice(origins),
                "dest": np.random.choice(dests),
                "route": "EWR_ORD",
                "distance": 719,
                TARGET_COL: np.random.choice([0, 1]),
            })

    df_base = pd.DataFrame(records)
    df_base = compute_schedule_and_calendar_features(df_base)

    # 1. Compute historical rates on original dataset
    gen_orig = HistoricalFeatureGenerator(SmoothingConfig(carrier_m=10.0, default_prior_delay_rate=0.25))
    df_out_orig = gen_orig.fit_transform_chronological(df_base, date_col="flight_date", target_col=TARGET_COL)

    # 2. Mutate future flight outcomes on dates >= Day 5
    cutoff_date = dates[4].date()
    df_mutated = df_base.copy()
    future_mask = pd.to_datetime(df_mutated["flight_date"]).dt.date >= cutoff_date
    # Invert all future targets: 0 -> 1, 1 -> 0
    df_mutated.loc[future_mask, TARGET_COL] = 1 - df_mutated.loc[future_mask, TARGET_COL]

    # Re-run historical feature generator on mutated dataset
    gen_mut = HistoricalFeatureGenerator(SmoothingConfig(carrier_m=10.0, default_prior_delay_rate=0.25))
    df_out_mut = gen_mut.fit_transform_chronological(df_mutated, date_col="flight_date", target_col=TARGET_COL)

    # 3. Assert earlier rows (dates < Day 5) have EXACT same historical features
    past_mask = pd.to_datetime(df_base["flight_date"]).dt.date < cutoff_date
    hist_cols = [
        "carrier_hist_delay_rate",
        "origin_hist_delay_rate",
        "dest_hist_delay_rate",
        "route_hist_delay_rate",
        "hist_carrier_flight_count",
    ]

    for col in hist_cols:
        orig_vals = df_out_orig.loc[past_mask, col].values
        mut_vals = df_out_mut.loc[past_mask, col].values
        np.testing.assert_array_almost_equal(
            orig_vals,
            mut_vals,
            decimal=7,
            err_msg=f"Leakage detected! Column '{col}' changed for past flights when future labels mutated."
        )


def test_forbidden_columns_raise_leakage_error():
    """Verifies that attempting to include post-departure outcome fields raises DataLeakageError."""
    features_with_leakage = ["sched_dep_hour", "carrier", "distance", "dep_delay"]
    with pytest.raises(DataLeakageError, match="DATA LEAKAGE DETECTED"):
        audit_feature_matrix_for_forbidden_columns(features_with_leakage, strict=True)

    features_with_target = ["sched_dep_hour", "carrier", "distance", TARGET_COL]
    with pytest.raises(DataLeakageError, match="DATA LEAKAGE DETECTED"):
        audit_feature_matrix_for_forbidden_columns(features_with_target, strict=True)
