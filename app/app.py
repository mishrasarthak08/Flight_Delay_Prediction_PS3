"""Historical 2013 demonstrator for the Phase 1 ML evaluation."""

from pathlib import Path
import json
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import RESULTS_DIR, MODELS_DIR
from src.inference import FlightDelayPredictor
from src.reschedule import ReschedulingSimulator

st.set_page_config(page_title="Flight Delay Prediction", page_icon="✈️", layout="wide")
st.title("Flight Delay Prediction")
st.caption("Phase 1 machine learning evaluation · Historical NYC departures, 2013")


@st.cache_data
def _cached_json(path, version):
    return json.loads(Path(path).read_text())


@st.cache_resource
def _cached_predictor(name, version):
    return FlightDelayPredictor(name)


def read_json(path):
    return _cached_json(str(path), Path(path).stat().st_mtime_ns)


def load_predictor(name):
    version = (MODELS_DIR / "bundle_metadata.json").stat().st_mtime_ns
    return _cached_predictor(name, version)


def examples():
    path = RESULTS_DIR / "reports/demo_flights.csv"
    if not path.exists():
        st.error(
            "Historical examples are missing. Run python scripts/build_demo.py after the training pipeline."
        )
        st.stop()
    return pd.read_csv(path)


def select_flight(key):
    table = examples()
    options = table.id.tolist()
    by_id = table.set_index("id")

    def label(record_id):
        row = by_id.loc[record_id]
        return f"{row.flight_date} · {row.carrier} {row.origin}–{row.dest} · {int(row.sched_dep_time):04d}"

    record_id = st.selectbox(
        "Historical scheduled flight", options, format_func=label, key=key
    )
    row = by_id.loc[record_id]
    inputs = row[
        [
            "flight_date",
            "carrier",
            "origin",
            "dest",
            "sched_dep_time",
            "sched_arr_time",
            "distance",
        ]
    ].to_dict()
    st.dataframe(pd.DataFrame([inputs]), hide_index=True, width="stretch")
    return inputs


summary_path = RESULTS_DIR / "run_summary.json"
if not summary_path.exists():
    st.error(
        "No experiment artifacts found. Follow the README to download data and run the pipeline."
    )
    st.stop()
summary = read_json(str(summary_path))
metadata = read_json(str(MODELS_DIR / "bundle_metadata.json"))
page = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Predict a historical flight",
        "Model evaluation",
        "Training-data EDA",
        "Rescheduling sensitivity",
        "Method and limitations",
    ],
)
st.sidebar.info(
    "Decision time: two hours before scheduled departure. This bundle demonstrates 2013 flights and does not provide current operational forecasts."
)

if page == "Overview":
    columns = st.columns(3)
    columns[0].metric(
        "Raw scheduled flights", f"{summary['dataset']['initial_raw_rows']:,}"
    )
    columns[1].metric(
        "Observed arrival outcomes", f"{summary['dataset']['operated_flights_count']:,}"
    )
    columns[2].metric("Origins", "EWR, JFK, LGA")
    st.write(
        "We estimate the probability of arrival at least 15 minutes late, conditional on an observed arrival outcome. Logistic Regression and XGBoost are the Phase 1 models. The embedding MLP is supplementary."
    )
    st.write(
        "The experiment tunes candidate models using four expanding date folds inside training, then chooses their feature families on the forward selection block. Separate development periods fit calibration, select calibration, and select the operating threshold. Historical features use labels available before each training prediction and daily snapshots of outcomes released before prediction."
    )
    st.warning(
        "The original test results were inspected during code review. These corrected results are a retrospective reevaluation, not a fresh external test."
    )
    st.dataframe(
        pd.read_csv(RESULTS_DIR / "metrics/model_comparison.csv"),
        hide_index=True,
        width="stretch",
    )

elif page == "Predict a historical flight":
    labels = {
        "Logistic Regression": "logistic_regression",
        "XGBoost": "xgboost",
        "Embedding MLP (supplementary)": "pytorch_mlp",
    }
    labels = {
        name: key for name, key in labels.items() if key in metadata["supported_models"]
    }
    chosen = st.selectbox("Model", list(labels))
    inputs = select_flight("prediction_flight")
    if st.button("Predict delay risk", type="primary"):
        try:
            predictor = load_predictor(labels[chosen])
            result = predictor.predict_single(inputs)
            cols = st.columns(3)
            cols[0].metric(
                "Predicted delay probability",
                f"{result['predicted_delay_prob_calibrated']:.1%}",
            )
            cols[1].metric(
                "Selected threshold", f"{result['operational_threshold']:.3f}"
            )
            cols[2].metric(
                "Decision",
                "Delay alert" if result["predicted_is_delayed"] else "No alert",
            )
            st.write(
                f"Selected probability treatment: {predictor.calibrator.method}. A no-alert decision does not guarantee an on-time arrival."
            )
            if not predictor.threshold_meta["target_precision_met"]:
                st.warning(
                    "The 70% development precision target was infeasible with the required alert support. The saved fallback rule maximizes development F1. Test precision may differ."
                )
        except (ValueError, FileNotFoundError) as error:
            st.error(str(error))

elif page == "Model evaluation":
    st.dataframe(
        pd.read_csv(RESULTS_DIR / "metrics/model_comparison.csv"),
        hide_index=True,
        width="stretch",
    )
    st.write(
        "AP denotes Average Precision on raw model scores. Operational decisions use the selected calibration output and a threshold selected on that same score scale. Lower Brier loss is better. The constant predictor uses the fitted training prevalence."
    )
    for filename in [
        "pr_curves.png",
        "roc_curves.png",
        "calibration_curves.png",
        "confusion_matrices.png",
        "feature_importances.png",
    ]:
        st.image(str(RESULTS_DIR / "figures" / filename), width="stretch")
    st.subheader("Feature ablation on development data")
    st.dataframe(
        pd.read_csv(RESULTS_DIR / "metrics/feature_ablation.csv"),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Calibration selection on its own development block")
    st.json(read_json(str(RESULTS_DIR / "reports/calibration_selection.json")))

elif page == "Training-data EDA":
    st.write(
        "These figures describe the outer training dates. Group differences are associations and do not establish causes of delay."
    )
    for filename in [
        "eda_delay_by_hour.png",
        "eda_delay_by_month.png",
        "eda_delay_by_carrier.png",
        "eda_delay_by_origin.png",
    ]:
        st.image(str(RESULTS_DIR / "figures" / filename), width="stretch")
    st.json(summary["dataset"])

elif page == "Rescheduling sensitivity":
    st.write(
        "This analysis holds the historical information set and background timetable fixed. It assumes the selected historical flight appears in that timetable and moves only that focal flight."
    )
    inputs = select_flight("sensitivity_flight")
    shifts = st.multiselect(
        "Departure shifts in minutes",
        [-60, -45, -30, -15, 0, 15, 30, 45, 60],
        default=[-60, -30, 0, 30, 60],
    )
    if st.button("Compare schedule shifts", type="primary"):
        try:
            predictor = load_predictor("xgboost")
            prepared = predictor.preprocess_raw_input(pd.DataFrame([inputs])).iloc[0]
            sim = ReschedulingSimulator(
                predictor.model,
                predictor.preprocessor,
                predictor.calibrator,
                schedule_context=predictor.schedule_context,
            )
            result = sim.simulate_shifts_for_flight(prepared, sorted(shifts))
            scenarios = pd.DataFrame(result["scenarios"])
            st.dataframe(scenarios, hide_index=True, width="stretch")
            st.bar_chart(
                scenarios.set_index("shift_minutes")["predicted_delay_probability"]
            )
            st.metric(
                "Lowest model risk shift", f"{result['optimal_shift_minutes']:+d} min"
            )
        except (ValueError, FileNotFoundError) as error:
            st.error(str(error))
    st.warning(ReschedulingSimulator.CAUSAL_DISCLAIMER)

else:
    st.markdown("""
**Population and target.** NYC departures in 2013 with a measured arrival delay. Missing arrival outcomes are excluded from supervised evaluation, but retained in timetable-demand counts.

**Information at prediction.** Scheduled carrier, route, clocks, distance and calendar. Historical outcomes enter only after their reconstructed actual arrival plus a declared 24-hour reporting lag. The source lacks true publication timestamps. The static source timetable is assumed available at the decision time.

**Date protocol.** All flights on a date share a partition. Model fitting purges labels unavailable at the fitting cutoff. Development, test and inference replay the same local-midnight history snapshots. Earlier development/test outcomes may enter later snapshots only after their release time. Estimator weights, calibration and thresholds remain fixed during test.

**Calibration and decisions.** Calibration fitting, method selection and threshold selection use separate date blocks. Candidate parameters use training-only temporal folds. The forward pipeline-selection block chooses feature families and calibration. A failed precision target remains explicitly reported.

**Limits.** One year and three origins do not establish performance for current flights or other regions. Weather, aircraft rotations, airport capacity, crews and connections are unobserved. Rescheduling outputs are predictive sensitivity, with no demonstrated causal benefit.
""")
    st.json(summary["split_manifest"])
