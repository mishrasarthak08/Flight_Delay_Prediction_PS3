"""Verify saved metrics, threshold selection, and inference against predictions."""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_DIR, MODELS_DIR
from src.calibration import select_operational_threshold
from src.inference import FlightDelayPredictor
from src.data import load_raw_dataset


def main():
    records = json.loads((RESULTS_DIR / "metrics/evaluation_metrics.json").read_text())
    names = {
        "Logistic Regression": "logistic_regression",
        "XGBoost": "xgboost",
        "PyTorch Embedding MLP": "pytorch_mlp",
    }
    raw = load_raw_dataset()
    checks = []
    for metric in records:
        if metric["model_name"] not in names:
            continue
        key = names[metric["model_name"]]
        prediction = pd.read_csv(
            RESULTS_DIR / f"predictions/{key}_test_predictions.csv"
        )
        threshold = json.loads((MODELS_DIR / f"{key}_threshold.json").read_text())
        assert prediction.record_id.is_unique
        y, p, pc = prediction.y_true, prediction.prob_raw, prediction.prob_calibrated
        for field, value in [
            ("pr_auc", average_precision_score(y, p)),
            ("roc_auc", roc_auc_score(y, p)),
            ("brier_score_raw", brier_score_loss(y, p)),
            ("brier_score_calibrated", brier_score_loss(y, pc)),
        ]:
            assert np.isclose(metric[field], value, atol=1e-12), (
                key,
                field,
                value,
                metric[field],
            )
        alerts = pc.to_numpy() >= threshold["selected_threshold"]
        np.testing.assert_array_equal(alerts.astype(int), prediction.y_pred_operational)
        assert np.isclose(
            precision_score(y, alerts, zero_division=0),
            metric["precision_at_op_threshold"],
        )
        assert np.isclose(
            recall_score(y, alerts, zero_division=0), metric["recall_at_op_threshold"]
        )
        threshold_data = pd.read_csv(
            RESULTS_DIR / f"predictions/{key}_threshold_validation_predictions.csv"
        )
        recomputed = select_operational_threshold(
            threshold_data.prob_calibrated.to_numpy(),
            threshold_data.y_true.to_numpy(),
            threshold["target_precision"],
            min_alerts=threshold["minimum_alerts"],
        )
        assert np.isclose(
            recomputed["selected_threshold"],
            threshold["selected_threshold"],
            atol=1e-12,
        )
        seen = set(prediction.record_id)
        for block in [
            "calibration_fit",
            "calibration_selection",
            "threshold_validation",
        ]:
            block_data = pd.read_csv(
                RESULTS_DIR / f"predictions/{key}_{block}_predictions.csv"
            )
            assert not seen.intersection(block_data.record_id)
            seen.update(block_data.record_id)
        sample = prediction.sample(12, random_state=42)
        source = raw.set_index("id").loc[sample.record_id].reset_index()
        source["flight_date"] = pd.to_datetime(source[["year", "month", "day"]])
        source = source[
            [
                "flight_date",
                "carrier",
                "origin",
                "dest",
                "sched_dep_time",
                "sched_arr_time",
                "distance",
            ]
        ]
        predictor = FlightDelayPredictor(key)
        output = predictor.predict_dataframe(source)
        np.testing.assert_allclose(
            output.predicted_delay_prob_raw, sample.prob_raw, atol=1e-7
        )
        np.testing.assert_allclose(
            output.predicted_delay_prob_calibrated, sample.prob_calibrated, atol=1e-7
        )
        checks.append(
            {
                "model": key,
                "test_rows": len(prediction),
                "inference_replays": len(sample),
                "status": "passed",
            }
        )
    print(json.dumps({"status": "passed", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
