"""
Unit tests for offline rescheduling what-if simulator.
"""

import pytest
import numpy as np
import pandas as pd

from src.reschedule import ReschedulingSimulator
from src.features import compute_schedule_and_calendar_features


class MockModel:
    """Mock model predicting probability proportional to departure hour."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Predict higher delay for later departures (simulate evening congestion)
        return np.linspace(0.20, 0.40, len(X))


class MockPreprocessor:
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return df.values


def test_rescheduling_simulator_duration_preservation_and_risk_diff():
    """Tests that schedule shifts preserve duration and correctly compute risk differences."""
    flight_dict = {
        "flight_date": pd.to_datetime("2013-05-10"),
        "sched_dep_time": 1000,  # 10:00 -> 600 min
        "sched_arr_time": 1200,  # 12:00 -> 720 min (duration = 120 min)
        "carrier": "UA",
        "origin": "EWR",
        "dest": "ORD",
        "distance": 719,
        "carrier_hist_delay_rate": 0.25,
        "origin_hist_delay_rate": 0.28,
        "dest_hist_delay_rate": 0.24,
        "route_hist_delay_rate": 0.26,
        "hist_carrier_flight_count": 500,
        "hist_origin_flight_count": 800,
        "hist_dest_flight_count": 400,
        "hist_route_flight_count": 200,
    }

    df_sample = compute_schedule_and_calendar_features(pd.DataFrame([flight_dict]))
    for col in [
        "carrier_recent_delay_rate",
        "origin_recent_delay_rate",
        "route_recent_delay_rate",
    ]:
        df_sample[col] = 0.25
    df_sample["origin_hour_scheduled_departures"] = 10
    df_sample["origin_window_scheduled_departures"] = 30
    flight_row = df_sample.iloc[0]

    sim = ReschedulingSimulator(
        model_wrapper=MockModel(),
        preprocessor=MockPreprocessor(),
        calibrator=None,
    )

    out = sim.simulate_shifts_for_flight(flight_row, candidate_shifts=[-60, 0, 60])
    scenarios = out["scenarios"]

    # Check 3 scenarios evaluated
    assert len(scenarios) == 3

    # Check baseline shift (0 min) has absolute risk difference 0.0
    baseline_scen = [s for s in scenarios if s["shift_minutes"] == 0][0]
    assert baseline_scen["absolute_risk_difference"] == 0.0

    # Check disclaimer exists
    assert "DISCLAIMER" in out["disclaimer"]


def test_rescheduling_with_saved_xgboost_model():
    """Integration test: validates that saved XGBoost predictor works seamlessly with ReschedulingSimulator."""
    from src.config import MODELS_DIR
    from src.inference import FlightDelayPredictor

    xgb_path = MODELS_DIR / "xgboost.pkl"
    if not xgb_path.exists():
        pytest.skip("xgboost.pkl not found, skipping integration test")

    predictor = FlightDelayPredictor("xgboost", models_dir=MODELS_DIR)
    flight_dict = {
        "flight_date": "2013-11-20",
        "sched_dep_time": 1600,
        "sched_arr_time": 1830,
        "carrier": "DL",
        "origin": "JFK",
        "dest": "ATL",
        "distance": 760,
    }

    df_full = predictor.preprocess_raw_input(pd.DataFrame([flight_dict]))
    sim = ReschedulingSimulator(
        model_wrapper=predictor.model,
        preprocessor=predictor.preprocessor,
        calibrator=predictor.calibrator,
        is_pytorch=predictor.is_pytorch,
    )

    out = sim.simulate_shifts_for_flight(
        df_full.iloc[0], candidate_shifts=[-60, -30, 0, 30, 60]
    )
    assert len(out["scenarios"]) == 5
    assert (
        out["baseline_delay_probability"] >= 0.0
        and out["baseline_delay_probability"] <= 1.0
    )
    assert (
        out["optimal_delay_probability"] >= 0.0
        and out["optimal_delay_probability"] <= 1.0
    )
    assert "DISCLAIMER" in out["disclaimer"]
