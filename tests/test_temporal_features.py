"""
Unit tests for temporal schedule and calendar feature extraction.
"""

import pytest
import numpy as np
import pandas as pd

from src.features import compute_schedule_and_calendar_features


def test_cyclical_and_schedule_decomposition():
    """Tests proper decomposition of HHMM times into hours, minutes, and cyclical vectors."""
    df = pd.DataFrame({
        "flight_date": pd.to_datetime(["2013-01-05", "2013-07-20"]),  # Saturday (weekend)
        "sched_dep_time": [600, 1830],  # 06:00 and 18:30
        "sched_arr_time": [845, 2115],  # 08:45 and 21:15
        "origin": ["EWR", "JFK"],
        "dest": ["ORD", "MIA"],
        "carrier": ["UA", "AA"],
        "distance": [719, 1089],
    })

    df_out = compute_schedule_and_calendar_features(df)

    # Check departure time features
    assert df_out["sched_dep_hour"].iloc[0] == 6
    assert df_out["sched_dep_min"].iloc[0] == 0
    assert df_out["sched_dep_min_of_day"].iloc[0] == 360

    assert df_out["sched_dep_hour"].iloc[1] == 18
    assert df_out["sched_dep_min"].iloc[1] == 30
    assert df_out["sched_dep_min_of_day"].iloc[1] == 1110

    # Check mathematical identity: sin^2 + cos^2 == 1.0
    sin_sq_plus_cos_sq = df_out["sin_sched_dep"]**2 + df_out["cos_sched_dep"]**2
    np.testing.assert_allclose(sin_sq_plus_cos_sq, 1.0, rtol=1e-5)

    sin_arr_sq_plus_cos_arr_sq = df_out["sin_sched_arr"]**2 + df_out["cos_sched_arr"]**2
    np.testing.assert_allclose(sin_arr_sq_plus_cos_arr_sq, 1.0, rtol=1e-5)

    # Check weekend flag (2013-01-05 and 2013-07-20 are both Saturdays)
    assert (df_out["is_weekend"] == 1).all()
    assert (df_out["day_of_week"] == 5).all()
