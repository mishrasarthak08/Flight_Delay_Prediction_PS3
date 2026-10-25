"""
Unit tests for end-to-end inference pipeline and unseen category handling.
"""

import pytest
import numpy as np
import pandas as pd
import json
import pickle

from src.inference import FlightDelayPredictor
from src.features import (
    compute_schedule_and_calendar_features,
    HistoricalFeatureGenerator,
    ScheduleContext,
)
from src.config import SmoothingConfig, TARGET_COL, ModelHyperparams
from src.preprocessing import FlightTabularPreprocessor
from src.models import LogisticRegressionBaseline
from src.validation import DataLeakageError


@pytest.fixture
def mock_inference_environment(tmp_path):
    """Creates a temporary models directory with fitted mock artifacts for inference testing."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    # 1. Historical generator
    hist_gen = HistoricalFeatureGenerator(
        SmoothingConfig(default_prior_delay_rate=0.25)
    )
    hist_gen.is_fitted = True
    hist_gen.latest_carrier_stats = {"UA": (25, 100), "AA": (15, 80)}
    hist_gen.latest_origin_stats = {"EWR": (40, 150), "JFK": (35, 140)}
    hist_gen.latest_dest_stats = {"ORD": (20, 90), "MIA": (10, 50)}
    hist_gen.latest_route_stats = {"EWR_ORD": (15, 60)}
    with open(models_dir / "historical_generator.json", "w") as f:
        json.dump(hist_gen.to_dict(), f)
    with open(models_dir / "schedule_context.json", "w") as f:
        json.dump(ScheduleContext({}, "2013-01-01", "2013-12-31").to_dict(), f)

    # 2. Fit Tabular Preprocessor & Mock Model
    from src.config import (
        CATEGORICAL_FEATURES,
        NUMERIC_SCHEDULE_FEATURES,
        HISTORICAL_FEATURES,
    )

    all_features = (
        CATEGORICAL_FEATURES + NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES
    )

    dummy_df = pd.DataFrame(
        {
            "flight_date": pd.date_range("2013-01-01", periods=10),
            "sched_dep_time": [800] * 10,
            "sched_arr_time": [1000] * 10,
            "carrier": ["UA", "AA"] * 5,
            "origin": ["EWR", "JFK"] * 5,
            "dest": ["ORD", "MIA"] * 5,
            "distance": [700] * 10,
            TARGET_COL: [0, 1] * 5,
        }
    )
    dummy_feat = compute_schedule_and_calendar_features(dummy_df)
    dummy_feat["origin_hour_scheduled_departures"] = 0
    dummy_feat["origin_window_scheduled_departures"] = 0
    dummy_full = hist_gen.transform_inference(dummy_feat)

    prep = FlightTabularPreprocessor(sparse_output=False)
    X = prep.fit_transform(dummy_full[all_features])
    y = dummy_full[TARGET_COL].values

    model = LogisticRegressionBaseline(ModelHyperparams())
    model.fit(X, y)

    prep.save(models_dir / "tabular_preprocessor.pkl")
    with open(models_dir / "logistic_regression.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(models_dir / "logistic_regression_threshold.json", "w") as f:
        json.dump({"selected_threshold": 0.45, "target_precision_met": True}, f)
    from src.calibration import ModelCalibrator

    ModelCalibrator("none").fit(np.array([0.2, 0.8]), np.array([0, 1])).save(
        models_dir / "logistic_regression_calibrator.pkl"
    )
    with open(models_dir / "bundle_metadata.json", "w") as f:
        json.dump({"policy_available_at": "2013-09-01T00:00:00+00:00"}, f)

    return models_dir


def test_inference_single_and_batch(mock_inference_environment):
    """Tests that inference runs cleanly and outputs bounded probabilities."""
    predictor = FlightDelayPredictor(
        model_name="logistic_regression", models_dir=mock_inference_environment
    )

    flight_input = {
        "flight_date": "2013-10-15",
        "sched_dep_time": 1430,
        "sched_arr_time": 1745,
        "carrier": "UA",
        "origin": "EWR",
        "dest": "ORD",
        "distance": 719,
    }

    result = predictor.predict_single(flight_input)
    assert 0.0 <= result["predicted_delay_prob_raw"] <= 1.0
    assert result["predicted_is_delayed"] in (0, 1)
    assert result["risk_tier"] in ("Low Risk", "Moderate Risk", "High Risk")


def test_inference_handles_unknown_categories(mock_inference_environment):
    """Verifies that novel/unseen airlines or destinations do not crash inference."""
    predictor = FlightDelayPredictor(
        model_name="logistic_regression", models_dir=mock_inference_environment
    )

    unseen_flight = {
        "flight_date": "2013-11-20",
        "sched_dep_time": 900,
        "sched_arr_time": 1200,
        "carrier": "NEW_AIRLINE",  # Unseen
        "origin": "NEW_AIRPORT",  # Unseen
        "dest": "UNKNOWN_DEST",  # Unseen
        "distance": 500,
    }

    result = predictor.predict_single(unseen_flight)
    assert 0.0 <= result["predicted_delay_prob_raw"] <= 1.0


def test_inference_rejects_forbidden_columns(mock_inference_environment):
    """Verifies that attempting inference with post-departure actual fields raises DataLeakageError."""
    predictor = FlightDelayPredictor(
        model_name="logistic_regression", models_dir=mock_inference_environment
    )

    leaked_input = pd.DataFrame(
        [
            {
                "flight_date": "2013-10-15",
                "sched_dep_time": 1430,
                "sched_arr_time": 1745,
                "carrier": "UA",
                "origin": "EWR",
                "dest": "ORD",
                "distance": 719,
                "dep_delay": 45.0,  # FORBIDDEN
            }
        ]
    )

    with pytest.raises(DataLeakageError):
        predictor.predict_dataframe(leaked_input)


def test_alias_loads_the_same_calibration_and_threshold(mock_inference_environment):
    canonical = FlightDelayPredictor(
        "logistic_regression", models_dir=mock_inference_environment
    )
    alias = FlightDelayPredictor("lr", models_dir=mock_inference_environment)
    assert alias.model_name == canonical.model_name
    assert alias.calibrator is not None
    assert alias.calibrator.method == canonical.calibrator.method
    assert alias.operational_threshold == canonical.operational_threshold


def test_inference_rejects_dates_before_policy_or_outside_timetable(
    mock_inference_environment,
):
    predictor = FlightDelayPredictor("lr", models_dir=mock_inference_environment)
    flight = {
        "flight_date": "2013-01-01",
        "sched_dep_time": 900,
        "sched_arr_time": 1200,
        "carrier": "UA",
        "origin": "EWR",
        "dest": "ORD",
        "distance": 719,
    }
    with pytest.raises(ValueError, match="policy cutoff"):
        predictor.predict_single(flight)
    flight["flight_date"] = "2026-10-25"
    with pytest.raises(ValueError, match="historical 2013"):
        predictor.predict_single(flight)
