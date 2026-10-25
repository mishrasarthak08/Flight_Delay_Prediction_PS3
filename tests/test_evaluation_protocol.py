"""Regression tests for availability, score scales, timetable demand and replay."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_score, recall_score
from src.features import (
    HistoricalFeatureGenerator,
    ScheduleContext,
    compute_schedule_and_calendar_features,
)
from src.config import HISTORICAL_FEATURES
from src.calibration import ModelCalibrator, select_operational_threshold
from src.experiments import prepare_partitions, development_blocks
from src.reschedule import ReschedulingSimulator


def flights():
    return compute_schedule_and_calendar_features(
        pd.DataFrame(
            {
                "flight_date": pd.to_datetime(
                    ["2013-01-01", "2013-01-02", "2013-01-03"]
                ),
                "sched_dep_time": [800] * 3,
                "sched_arr_time": [1000] * 3,
                "carrier": ["UA"] * 3,
                "origin": ["EWR"] * 3,
                "dest": ["ORD"] * 3,
                "distance": [719] * 3,
                "is_delayed": [0, 1, 1],
                "label_available_at": pd.to_datetime(
                    ["2013-01-01T20:00Z", "2013-01-02T20:00Z", "2013-01-03T20:00Z"]
                ),
            }
        )
    )


def test_future_labels_cannot_change_earlier_features_including_global_prior():
    frame = flights()
    before = HistoricalFeatureGenerator().fit_transform_chronological(frame)
    mutated = frame.copy()
    mutated.loc[1:, "is_delayed"] = 1 - mutated.loc[1:, "is_delayed"]
    after = HistoricalFeatureGenerator().fit_transform_chronological(mutated)
    np.testing.assert_allclose(
        before.loc[0, HISTORICAL_FEATURES].astype(float),
        after.loc[0, HISTORICAL_FEATURES].astype(float),
    )


def test_labels_must_be_available_strictly_before_prediction():
    frame = flights()
    frame.loc[0, "label_available_at"] = frame.loc[1, "decision_time"]
    output = HistoricalFeatureGenerator().fit_transform_chronological(frame)
    assert output.loc[1, "hist_carrier_flight_count"] == 0
    assert output.loc[2, "hist_carrier_flight_count"] == 2


def test_frozen_history_continues_across_partitions_and_rejects_old_dates():
    frame = flights()
    train, future = frame.iloc[:2], frame.iloc[2:]
    _, output, gen, info = prepare_partitions(train, future)
    assert output.iloc[0].hist_carrier_flight_count == 2
    assert info["fitted_rows"] == 2
    mutated = future.copy()
    mutated["is_delayed"] = 0
    pd.testing.assert_frame_equal(
        output[HISTORICAL_FEATURES],
        gen.transform_inference(mutated)[HISTORICAL_FEATURES],
    )
    restored = HistoricalFeatureGenerator.from_dict(gen.to_dict())
    pd.testing.assert_frame_equal(
        output[HISTORICAL_FEATURES],
        restored.transform_inference(future)[HISTORICAL_FEATURES],
    )
    with pytest.raises(ValueError, match="precedes"):
        restored.transform_inference(train)


def test_model_fit_purges_unavailable_training_labels():
    frame = flights()
    frame.loc[1, "label_available_at"] = pd.Timestamp("2013-01-04T00:00Z")
    # Retain another available positive to keep the fitting population two-class.
    extra = frame.iloc[[0]].copy()
    extra["is_delayed"] = 1
    train = pd.concat([frame.iloc[:2], extra], ignore_index=True)
    fitted, _, _, info = prepare_partitions(train, frame.iloc[2:])
    assert info["purged_unavailable_or_invalid_labels"] == 1
    assert len(fitted) == 2


def test_timetable_counts_include_missing_outcomes_and_ignore_label_mutation():
    roster = flights()
    extra = roster.iloc[[0]].copy()
    extra["is_delayed"] = np.nan
    roster = pd.concat([roster, extra], ignore_index=True)
    context = ScheduleContext.from_roster(roster)
    assert context.transform(roster).iloc[0].origin_hour_scheduled_departures == 2
    changed = roster.copy()
    changed["is_delayed"] = 1
    assert context.to_dict() == ScheduleContext.from_roster(changed).to_dict()


def test_threshold_is_selected_on_the_calibrated_scale():
    raw = np.linspace(0.1, 0.9, 400)
    y = (np.arange(400) % 4 == 0).astype(int)
    cal = ModelCalibrator("sigmoid").fit(raw, y)
    calibrated = cal.predict_proba(raw)
    info = select_operational_threshold(
        calibrated, y, target_precision=0.7, min_alerts=20
    )
    alarms = calibrated >= info["selected_threshold"]
    assert info["validation_precision_at_threshold"] == pytest.approx(
        precision_score(y, alarms)
    )
    assert info["validation_recall_at_threshold"] == pytest.approx(
        recall_score(y, alarms)
    )
    assert info["validation_alert_count"] == int(alarms.sum())
    assert info["target_precision_met"] is False
    assert int(alarms.sum()) >= 20


def test_identity_calibration_preserves_scores():
    p = np.array([0.1, 0.4, 0.8])
    np.testing.assert_array_equal(
        ModelCalibrator("none").fit(p, np.array([0, 0, 1])).predict_proba(p), p
    )


def test_daily_replay_cannot_read_future_test_outcomes():
    frame = flights()
    first = (
        HistoricalFeatureGenerator()
        .enable_rolling()
        .fit(frame)
        .transform_inference(frame)
    )
    changed = frame.copy()
    changed.loc[2, "is_delayed"] = 0
    second = (
        HistoricalFeatureGenerator()
        .enable_rolling()
        .fit(changed)
        .transform_inference(changed)
    )
    pd.testing.assert_frame_equal(
        first[HISTORICAL_FEATURES], second[HISTORICAL_FEATURES]
    )
    # A prior day's released outcome may affect subsequent dates, as in deployment.
    changed.loc[0, "is_delayed"] = 1
    third = (
        HistoricalFeatureGenerator()
        .enable_rolling()
        .fit(changed)
        .transform_inference(changed)
    )
    assert (
        first.loc[1, "origin_hist_delay_rate"] != third.loc[1, "origin_hist_delay_rate"]
    )


@pytest.mark.parametrize("clock", [1260, 2500, -1, 1234.5, np.nan])
def test_invalid_schedule_clocks_are_rejected(clock):
    frame = flights()
    frame["sched_dep_time"] = clock
    with pytest.raises(ValueError, match="HHMM"):
        compute_schedule_and_calendar_features(frame)


class Recorder:
    def transform(self, frame):
        self.frame = frame.copy()
        return np.zeros((len(frame), 1))


class Constant:
    def predict_proba(self, x):
        return np.full(len(x), 0.25)


def test_rescheduling_crosses_month_and_preserves_utc_duration():
    row = flights().iloc[0].copy()
    row["flight_date"], row["sched_dep_time"], row["sched_arr_time"] = (
        pd.Timestamp("2013-01-31"),
        2345,
        145,
    )
    row = compute_schedule_and_calendar_features(pd.DataFrame([row])).iloc[0]
    for col in HISTORICAL_FEATURES:
        row[col] = 0.25 if "rate" in col else 10
    (
        row["origin_hour_scheduled_departures"],
        row["origin_window_scheduled_departures"],
    ) = (10, 30)
    recorder = Recorder()
    sim = ReschedulingSimulator(Constant(), recorder)
    out = sim.simulate_shifts_for_flight(row, [0, 60])
    assert recorder.frame.iloc[1]["month"] == 2
    assert recorder.frame.iloc[1]["day"] == 1
    assert (
        out["scenarios"][0]["scheduled_duration_minutes"]
        == out["scenarios"][1]["scheduled_duration_minutes"]
    )
    assert out["optimal_shift_minutes"] == 0
    with pytest.raises(ValueError, match="zero-shift"):
        sim.simulate_shifts_for_flight(row, [30, 60])
    with pytest.raises(ValueError, match="15-minute"):
        sim.simulate_shifts_for_flight(row, [0, 90])


def test_development_decisions_use_disjoint_dates_and_available_labels():
    template = flights().iloc[[0]]
    frames = []
    for i, date in enumerate(pd.date_range("2013-09-01", periods=60)):
        frame = template.copy()
        frame["flight_date"] = date
        frame = compute_schedule_and_calendar_features(frame)
        frame["label_available_at"] = frame["decision_time"] + pd.Timedelta(days=2)
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    blocks, manifest = development_blocks(all_rows.iloc[:54], all_rows.iloc[54:])
    seen = set()
    for name, frame in blocks.items():
        dates = set(frame.flight_date)
        assert not seen.intersection(dates)
        seen |= dates
        assert (
            frame.label_available_at
            < pd.Timestamp(manifest[name]["labels_available_before"])
        ).all()
