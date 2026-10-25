"""Build the three analytical notebooks and optionally execute them in order."""

import argparse
from pathlib import Path
import nbformat as nb
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SETUP = """from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path.cwd()
if not (ROOT / "src").exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.config import RESULTS_DIR, ALL_MODEL_FEATURES, FORBIDDEN_COLUMNS
from src.data import load_raw_dataset, clean_and_prepare_dataset, attach_label_availability
from src.features import compute_schedule_and_calendar_features
summary = json.loads((RESULTS_DIR / "run_summary.json").read_text())
plt.rcParams.update({"figure.figsize": (9, 4), "font.size": 11})
"""


def notebook(title, cells):
    out = nb.v4.new_notebook()
    out.cells = [nb.v4.new_markdown_cell(title), nb.v4.new_code_cell(SETUP)]
    out.cells += [
        nb.v4.new_markdown_cell(text) if kind == "md" else nb.v4.new_code_cell(text)
        for kind, text in cells
    ]
    out.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()
    audit = notebook(
        "# 01. Dataset quality and prediction-time contract",
        [
            (
                "md",
                "We predict arrival delay of at least 15 minutes, two hours before scheduled departure. The supervised population requires an observed arrival outcome. Missing outcomes are not labeled on time.",
            ),
            (
                "code",
                """raw = load_raw_dataset()
clean, profile = clean_and_prepare_dataset(raw)
print("Raw shape:", raw.shape, "Observed arrival outcomes:", len(clean))
print("Duplicate flight keys:", profile["duplicate_flight_keys"])
display(raw.head(3))""",
            ),
            (
                "code",
                """missing = raw.isna().sum().rename("missing_rows").to_frame()
missing["missing_fraction"] = missing["missing_rows"] / len(raw)
display(missing[missing.missing_rows > 0])
display(pd.Series(profile).drop(["mapped_columns", "raw_missing_counts"]).to_frame("value"))""",
            ),
            (
                "md",
                "Schedule fields, calendar, route and distance are potential predictors. Actual departure/arrival fields control retrospective label construction and historical availability audits only. They never enter the model feature matrix.",
            ),
            (
                "code",
                """from src.validation import audit_feature_matrix_for_forbidden_columns
print(audit_feature_matrix_for_forbidden_columns(ALL_MODEL_FEATURES))
display(pd.DataFrame({"allowed_model_features": pd.Series(ALL_MODEL_FEATURES), "forbidden_fields": pd.Series(FORBIDDEN_COLUMNS)}))""",
            ),
            (
                "md",
                "Actual arrival equals inferred scheduled arrival UTC plus observed arrival delay. The audit verifies the resulting destination local clock and plausible ground time. The declared 24-hour reporting lag is an assumption because true publication timestamps are unavailable.",
            ),
            (
                "code",
                """features = compute_schedule_and_calendar_features(clean)
features, availability = attach_label_availability(features)
display(pd.Series(availability).to_frame("value"))
known = features.label_available_at.notna()
assert (features.loc[known, "label_available_at"] > features.loc[known, "decision_time"]).all()
print("Every usable current-flight label is released after that flight's prediction time.")
display(features[["flight_date", "decision_time", "label_available_at"]].head())""",
            ),
            (
                "code",
                """manifest = summary["split_manifest"]
display(pd.DataFrame({k: manifest[k] for k in ["train", "validation", "test"]}).T)
display(pd.Series(manifest["model_fit"]).to_frame("value"))
display(pd.DataFrame(manifest["development_blocks"]).T)""",
            ),
            (
                "md",
                "The dataset covers one year and departures from three NYC airports. Generalization to all US flights or current operations is unverified. Missing outcomes create a conditional target population. The original test period has already been inspected, so the final experiment is a corrected retrospective reevaluation.",
            ),
        ],
    )
    eda = notebook(
        "# 02. Exploratory analysis on available training outcomes",
        [
            (
                "code",
                """raw = load_raw_dataset()
clean, _ = clean_and_prepare_dataset(raw)
features, _ = attach_label_availability(compute_schedule_and_calendar_features(clean))
cutoff = pd.Timestamp(summary["split_manifest"]["model_fit"]["fit_available_at"])
train_end = pd.Timestamp(summary["split_manifest"]["train"]["end_date"])
train = features[(features.flight_date <= train_end) & (features.label_available_at < cutoff)]
print("Available fitting rows:", len(train), "Delay prevalence:", round(train.is_delayed.mean(), 4))
assert len(train) == summary["split_manifest"]["model_fit"]["fitted_rows"]""",
            ),
            (
                "md",
                "The following plots use only labels available at the final fitting cutoff. Group rates are descriptive associations. Sample sizes and shared daily disruptions limit comparisons between carriers or airports.",
            ),
            (
                "code",
                """hour = train.groupby("sched_dep_hour").is_delayed.agg(["mean", "count"])
display(hour)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hour.index, hour["mean"], marker="o")
axes[0].set(xlabel="Scheduled departure hour", ylabel="Arrival delay rate")
axes[1].bar(hour.index, hour["count"])
axes[1].set(xlabel="Scheduled departure hour", ylabel="Training flights")
plt.tight_layout(); plt.show()""",
            ),
            (
                "code",
                """fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for axis, key in zip(axes, ["carrier", "origin"]):
    group = train.groupby(key).is_delayed.agg(["mean", "count"]).sort_values("mean")
    display(group)
    axis.bar(group.index, group["mean"])
    axis.set(xlabel=key, ylabel="Arrival delay rate")
    axis.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.show()""",
            ),
            (
                "code",
                """month = train.groupby("month").is_delayed.agg(["mean", "count"])
dow = train.groupby("day_of_week").is_delayed.agg(["mean", "count"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(month.index, month["mean"], marker="o"); axes[0].set(xlabel="Month", ylabel="Delay rate")
axes[1].bar(dow.index, dow["mean"]); axes[1].set(xlabel="Weekday (Monday=0)", ylabel="Delay rate")
plt.tight_layout(); plt.show()
display(train.groupby("route").is_delayed.agg(["mean", "count"]).sort_values("count", ascending=False).head(15))""",
            ),
            (
                "md",
                "The next table is a retrospective distribution-shift audit across predefined partitions. These test-label summaries are not used to choose model parameters, calibration or thresholds.",
            ),
            (
                "code",
                """display(pd.DataFrame({k: summary["split_manifest"][k] for k in ["train", "validation", "test"]}).T[["start_date", "end_date", "row_count", "positive_prevalence"]])
from src.features import ScheduleContext
roster = raw.copy(); roster["flight_date"] = pd.to_datetime(roster[["year", "month", "day"]])
context = ScheduleContext.from_roster(roster)
scheduled = context.transform(compute_schedule_and_calendar_features(clean))
display(scheduled[["origin_hour_scheduled_departures", "origin_window_scheduled_departures"]].describe())
assert sum(context.counts.values()) == len(raw)
print("All raw scheduled flights contribute to demand counts, including missing outcomes.")""",
            ),
            (
                "md",
                "Feature engineering follows these patterns without asserting causality: cyclic clock/calendar representations, carrier and airport identities, smoothed past outcome rates, recent released outcomes, and scheduled demand. The development ablation in notebook 03 measures their empirical contribution.",
            ),
        ],
    )
    comparison = notebook(
        "# 03. ML comparison, calibration and error analysis",
        [
            (
                "md",
                "Logistic Regression and XGBoost are the required Phase 1 models. The embedding MLP is supplementary. AP is Average Precision, not trapezoidal PR area. Every operating threshold uses the same selected probability scale at selection and inference.",
            ),
            (
                "code",
                """display(pd.read_csv(RESULTS_DIR / "metrics/model_comparison.csv"))
metrics = json.loads((RESULTS_DIR / "metrics/evaluation_metrics.json").read_text())
display(pd.DataFrame([{ "model": m["model_name"], "train_AP": m.get("training_average_precision"), "test_raw_AP": m["pr_auc"], "test_calibrated_AP": m.get("average_precision_calibrated"), "alerts": m["operational_alert_count"], "precision_defined": m["precision_defined"]} for m in metrics]))""",
            ),
            (
                "code",
                """search = json.loads((RESULTS_DIR / "reports/temporal_search.json").read_text())
display(pd.DataFrame(search["candidate_summary"]))
display(pd.DataFrame(search["selected_hyperparameters"]).T)""",
            ),
            (
                "code",
                """from IPython.display import display, Image
for name in ["pr_curves.png", "roc_curves.png", "calibration_curves.png", "confusion_matrices.png"]:
    display(Image(filename=str(RESULTS_DIR / "figures" / name), width=900))""",
            ),
            (
                "md",
                "Calibration is a development-data choice, including the identity option. A lower test Brier score is desirable but is not used to choose the calibrator. Raw and selected reliability curves accompany Brier loss because the loss also reflects discrimination.",
            ),
            (
                "code",
                """calibration = json.loads((RESULTS_DIR / "reports/calibration_selection.json").read_text())
for model, result in calibration.items():
    print(model, "selected:", result["selected_method"])
    display(pd.DataFrame(result["candidates"]))
for metric in metrics:
    if "threshold_selection_metadata" in metric:
        print(metric["model_name"])
        display(pd.Series(metric["threshold_selection_metadata"]).to_frame("value"))""",
            ),
            (
                "code",
                """ablation = pd.read_csv(RESULTS_DIR / "metrics/feature_ablation.csv")
display(ablation)
ax = ablation.pivot(index="features", columns="model", values="average_precision").plot.bar(figsize=(10, 4))
ax.set(ylabel="Development Average Precision", xlabel="Feature/protocol variant")
plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.show()""",
            ),
            (
                "md",
                "The frozen-history reference keeps the same fitted model configuration but holds history at its fitting cutoff. The primary daily replay updates only released historical outcomes. Estimator weights remain fixed. The ablation is evaluated on development data, not used to optimize the test result.",
            ),
            (
                "code",
                """slices = pd.read_csv(RESULTS_DIR / "metrics/error_slices.csv")
display(slices[(slices.model == "xgboost") & (slices.group == "origin")])
display(slices[(slices.model == "xgboost") & (slices.group == "flight_date")].sort_values("brier_score", ascending=False).head(10))
for metric in metrics:
    if "confidence_intervals_95_day_block" in metric:
        print(metric["model_name"], metric["confidence_intervals_95_day_block"])
cohort = pd.read_csv(RESULTS_DIR / "reports/rescheduling_cohort.csv")
display(cohort.describe())""",
            ),
            (
                "md",
                "Confidence intervals resample whole dates to retain within-day dependence. They describe this test period, not future regions or years. Rescheduling differences are model sensitivities only: no real-world delay reduction is established. See docs/METHODS.md for the prediction contract and limitations.",
            ),
        ],
    )
    for name, content in [
        ("01_data_audit.ipynb", audit),
        ("02_exploratory_analysis.ipynb", eda),
        ("03_model_comparison.ipynb", comparison),
    ]:
        if args.execute:
            print(f"Executing {name}...", flush=True)
            NotebookClient(
                content,
                timeout=300,
                kernel_name=args.kernel,
                resources={"metadata": {"path": str(ROOT / "notebooks")}},
            ).execute()
            content.metadata["kernelspec"] = {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        nb.write(content, ROOT / "notebooks" / name)
    print("Notebooks built" + (" and executed." if args.execute else "."))


if __name__ == "__main__":
    main()
