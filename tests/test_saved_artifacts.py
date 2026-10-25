"""Independent metric checks and inference replay using committed evidence."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss
from src.evaluate import evaluate_predictions
from src.inference import FlightDelayPredictor

ROOT = Path(__file__).resolve().parents[1]


def test_metrics_use_raw_ranking_and_selected_probability_decisions():
    result = evaluate_predictions(
        np.array([0, 1, 1, 0]),
        np.array([0.05, 0.1, 0.2, 0.9]),
        np.array([0.1, 0.3, 0.5, 0.4]),
        operational_threshold=0.2,
    )
    assert result["confusion_matrix_op"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert result["precision_at_op_threshold"] == pytest.approx(2 / 3)
    assert result["recall_at_op_threshold"] == 1
    assert result["pr_auc"] == pytest.approx(7 / 12)
    assert result["average_precision_calibrated"] == pytest.approx(5 / 6)
    assert result["brier_score_calibrated"] == pytest.approx(0.2275)


@pytest.mark.parametrize(
    "key,name",
    [
        ("logistic_regression", "Logistic Regression"),
        ("xgboost", "XGBoost"),
        ("pytorch_mlp", "PyTorch Embedding MLP"),
    ],
)
def test_committed_metrics_and_model_replay(key, name):
    records = json.loads((ROOT / "results/metrics/evaluation_metrics.json").read_text())
    metric = next(r for r in records if r["model_name"] == name)
    predictions = pd.read_csv(ROOT / f"results/predictions/{key}_test_predictions.csv")
    assert predictions.record_id.is_unique
    assert metric["pr_auc"] == pytest.approx(
        average_precision_score(predictions.y_true, predictions.prob_raw)
    )
    assert metric["brier_score_calibrated"] == pytest.approx(
        brier_score_loss(predictions.y_true, predictions.prob_calibrated)
    )
    selected = predictions.prob_calibrated >= metric["operational_threshold"]
    np.testing.assert_array_equal(selected.astype(int), predictions.y_pred_operational)
    demo = pd.read_csv(ROOT / "results/reports/demo_flights.csv").head(5)
    predictor = FlightDelayPredictor(key)
    output = predictor.predict_dataframe(demo.drop(columns=["id"]))
    expected = predictions.set_index("record_id").loc[demo.id]
    np.testing.assert_allclose(
        output.predicted_delay_prob_raw, expected.prob_raw, atol=1e-6
    )
    np.testing.assert_allclose(
        output.predicted_delay_prob_calibrated, expected.prob_calibrated, atol=1e-6
    )
