"""Prepare reproducible historical examples from saved test record identifiers."""

from pathlib import Path
import sys
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_raw_dataset
from src.config import RESULTS_DIR

ROOT = Path(__file__).resolve().parents[1]


def main():
    predictions = pd.read_csv(RESULTS_DIR / "predictions/xgboost_test_predictions.csv")
    sample = predictions.sample(min(100, len(predictions)), random_state=42)
    raw = load_raw_dataset().set_index("id").loc[sample.record_id].reset_index()
    raw["flight_date"] = pd.to_datetime(raw[["year", "month", "day"]])
    columns = [
        "id",
        "flight_date",
        "carrier",
        "origin",
        "dest",
        "sched_dep_time",
        "sched_arr_time",
        "distance",
    ]
    raw[columns].sort_values(["flight_date", "sched_dep_time"]).to_csv(
        RESULTS_DIR / "reports/demo_flights.csv", index=False
    )
    records = json.loads((RESULTS_DIR / "metrics/evaluation_metrics.json").read_text())
    lines = [
        "| Model | Raw AP | Raw ROC-AUC | Selected Brier | Accuracy | Precision | Recall | Calibration |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    infeasible = []
    for metric in records:
        treatment = metric.get("calibration_selection", {}).get(
            "selected_method", "constant"
        )
        confusion = metric["confusion_matrix_op"]
        accuracy = (confusion["tp"] + confusion["tn"]) / metric["num_samples"]
        precision = (
            f"{metric['precision_at_op_threshold']:.2%}"
            if metric["precision_defined"]
            else "—"
        )
        lines.append(
            f"| {metric['model_name']} | {metric['pr_auc']:.4f} | {metric['roc_auc']:.4f} | {metric['brier_score_calibrated']:.4f} | {accuracy:.2%} | {precision} | {metric['recall_at_op_threshold']:.2%} | {treatment} |"
        )
        if (
            metric.get("threshold_selection_metadata", {}).get("target_precision_met")
            is False
        ):
            infeasible.append(metric["model_name"])
    if infeasible:
        lines += [
            "",
            "The project-selected 70% development precision target was infeasible with minimum alert support for: "
            + ", ".join(infeasible)
            + ". Their operating points use the declared development-F1 fallback.",
        ]
    readme = ROOT / "README.md"
    text = readme.read_text()
    begin, end = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"
    before, remainder = text.split(begin, 1)
    _, after = remainder.split(end, 1)
    readme.write_text(before + begin + "\n" + "\n".join(lines) + "\n" + end + after)
    print(f"Prepared {len(raw)} historical examples and refreshed the README table.")


if __name__ == "__main__":
    main()
