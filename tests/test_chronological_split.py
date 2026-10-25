"""
Unit tests for chronological date splitting and partition boundaries.
"""

import pandas as pd
import numpy as np

from src.splitting import chronological_date_split
from src.config import SplittingConfig, TARGET_COL
from src.validation import verify_no_future_leakage_in_splits


def test_chronological_split_disjoint_dates():
    """Verifies that all flights from any date belong to the same split and splits are disjoint."""
    dates = pd.date_range("2013-01-01", periods=100, freq="D")
    records = []
    for d in dates:
        n_flights = np.random.randint(10, 30)
        for _ in range(n_flights):
            records.append(
                {
                    "flight_date": d,
                    "sched_dep_time": 800,
                    "sched_arr_time": 1000,
                    "carrier": "UA",
                    "origin": "EWR",
                    "dest": "ORD",
                    "distance": 719,
                    TARGET_COL: np.random.choice([0, 1]),
                }
            )

    df = pd.DataFrame(records)
    cfg = SplittingConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df, manifest = chronological_date_split(df, config=cfg)

    train_dates = set(pd.to_datetime(train_df["flight_date"]).dt.date)
    val_dates = set(pd.to_datetime(val_df["flight_date"]).dt.date)
    test_dates = set(pd.to_datetime(test_df["flight_date"]).dt.date)

    # 1. Disjoint partitions
    assert len(train_dates.intersection(val_dates)) == 0
    assert len(val_dates.intersection(test_dates)) == 0
    assert len(train_dates.intersection(test_dates)) == 0

    # 2. Complete date coverage
    assert len(train_dates) + len(val_dates) + len(test_dates) == 100

    # 3. Monotonic ordering
    assert max(train_dates) < min(val_dates)
    assert max(val_dates) < min(test_dates)

    # 4. Verify via validation helper
    verify_no_future_leakage_in_splits(
        train_df["flight_date"], val_df["flight_date"], test_df["flight_date"]
    )
