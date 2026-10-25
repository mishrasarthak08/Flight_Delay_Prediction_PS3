"""
Streamlit Web Application for Flight Delay Prediction & Rescheduling Simulator.
Provides interactive prediction, exploratory data analytics, model comparison,
and what-if departure rescheduling sensitivity analysis.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is available
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RESULTS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    MODELS_DIR,
)
from src.inference import FlightDelayPredictor
from src.reschedule import ReschedulingSimulator
from src.calibration import ModelCalibrator

# Page setup
st.set_page_config(
    page_title="Flight Delay Prediction | ML & Deep Learning",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-medium {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor(model_name: str = "xgboost"):
    """Loads predictor artifact."""
    try:
        return FlightDelayPredictor(model_name=model_name, models_dir=MODELS_DIR)
    except Exception as e:
        return None


@st.cache_data
def load_run_summary() -> Optional[Dict[str, Any]]:
    path = RESULTS_DIR / "run_summary.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_model_comparison() -> Optional[pd.DataFrame]:
    path = METRICS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# Sidebar Navigation
st.sidebar.title("✈️ Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "Home & Project Overview",
        "Flight Delay Prediction",
        "Offline Rescheduling Simulator",
        "Model Performance & Benchmarks",
        "Exploratory Delay Analytics",
        "Methodology & Limitations",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Operational Contract:**\n\n"
    "Predictions are made **2 hours prior to scheduled departure** using only pre-departure attributes."
)


# ------------------------------------------------------------------------------
# PAGE 1: Home & Project Overview
# ------------------------------------------------------------------------------
if page == "Home & Project Overview":
    st.markdown('<div class="main-header">Flight Delay Prediction via Gradient Boosting & Deep Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Pre-departure probabilistic arrival delay forecasting and what-if schedule sensitivity analysis.</div>', unsafe_allow_html=True)

    summary = load_run_summary()
    if summary:
        ds = summary["dataset"]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records Analyzed", f"{ds['operated_flights_count']:,}")
        with col2:
            st.metric("Delay Prevalence (>=15m)", f"{ds['positive_class_prevalence']*100:.1f}%")
        with col3:
            st.metric("Unique Origin Hubs", f"{ds['unique_origins']}")
        with col4:
            st.metric("Operating Airlines", f"{ds['unique_carriers']}")
    else:
        st.warning("Pipeline artifacts not found. Please run the training pipeline first.")

    st.markdown("---")
    st.markdown("""
    ### 🎯 Research Objectives
    1. **Strict Operational Cutoff**: Generate probabilistic delay alerts at $T - 2$ hours prior to scheduled wheels-off.
    2. **Zero-Leakage Historical Aggregation**: Empirical Bayes past-only smoothing using completed calendar dates exclusively.
    3. **Multi-Architecture Comparison**:
       - **Baseline**: Regularized Logistic Regression with One-Hot & Standardized features.
       - **Gradient Boosting**: Tuned XGBoost with histogram-based partitioning & early stopping.
       - **Deep Learning**: PyTorch Neural Network with learned categorical entity embeddings for carrier, origin, and destination.
    4. **Uncertainty Calibration & Decision Thresholding**: Isotonic probability calibration evaluated on an untouched chronological test partition.
    5. **What-If Schedule Sensitivity**: Offline rescheduling simulator evaluating candidate departure shifts without causal overclaiming.
    """)


# ------------------------------------------------------------------------------
# PAGE 2: Flight Delay Prediction
# ------------------------------------------------------------------------------
elif page == "Flight Delay Prediction":
    st.markdown('<div class="main-header">Real-Time Delay Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter pre-departure flight specifications to generate calibrated delay probabilities.</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Select Model Architecture",
        ["XGBoost", "PyTorch Embedding MLP", "Logistic Regression"]
    )
    model_key = {
        "XGBoost": "xgboost",
        "PyTorch Embedding MLP": "pytorch_mlp",
        "Logistic Regression": "logistic_regression",
    }[model_choice]

    predictor = load_predictor(model_key)

    if predictor is None:
        st.error(f"Trained model artifacts for '{model_choice}' were not found in `models/`. Please execute `python scripts/run_pipeline.py` first.")
    else:
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                carrier = st.selectbox("Airline Carrier", ["UA", "AA", "DL", "B6", "EV", "MQ", "WN", "US", "9E", "VX", "FL", "AS", "F9", "YV", "HA", "OO"])
                origin = st.selectbox("Origin Airport", ["EWR", "JFK", "LGA"])
                dest = st.text_input("Destination Airport (e.g. ORD, LAX, MIA)", value="ORD").upper()

            with c2:
                dep_date = st.date_input("Scheduled Flight Date")
                dep_time_val = st.time_input("Scheduled Departure Time (Local)", value=pd.to_datetime("08:30").time())
                arr_time_val = st.time_input("Scheduled Arrival Time (Local)", value=pd.to_datetime("10:45").time())

            with c3:
                distance = st.number_input("Flight Distance (Miles)", min_value=50, max_value=5000, value=719)
                st.caption(f"Operational Prediction Cutoff: 2 hours before scheduled departure ({dep_time_val}).")

            submit_btn = st.form_submit_button("Compute Delay Probability", use_container_width=True)

        if submit_btn:
            sched_dep_num = dep_time_val.hour * 100 + dep_time_val.minute
            sched_arr_num = arr_time_val.hour * 100 + arr_time_val.minute

            flight_dict = {
                "flight_date": str(dep_date),
                "sched_dep_time": sched_dep_num,
                "sched_arr_time": sched_arr_num,
                "carrier": carrier,
                "origin": origin,
                "dest": dest,
                "distance": distance,
            }

            try:
                res = predictor.predict_single(flight_dict)

                st.markdown("---")
                st.subheader("Prediction Results")
                r1, r2, r3, r4 = st.columns(4)

                prob_cal = res["predicted_delay_prob_calibrated"]
                prob_raw = res["predicted_delay_prob_raw"]
                is_del = res["predicted_is_delayed"]
                tier = res["risk_tier"]

                with r1:
                    st.metric("Calibrated Delay Risk", f"{prob_cal * 100:.1f}%")
                with r2:
                    st.metric("Raw Model Risk", f"{prob_raw * 100:.1f}%")
                with r3:
                    st.metric("Operating Threshold", f"{res['operational_threshold']:.3f}")
                with r4:
                    badge_cls = "badge-low" if tier == "Low Risk" else ("badge-medium" if tier == "Moderate Risk" else "badge-high")
                    st.markdown(f"**Risk Tier**: <span class='{badge_cls}'>{tier}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Action**: {'ALERT: High Delay Probability' if is_del == 1 else 'NORMAL: Scheduled Operation'}")

            except Exception as ex:
                st.error(f"Inference error: {ex}")


# ------------------------------------------------------------------------------
# PAGE 3: Offline Rescheduling Simulator
# ------------------------------------------------------------------------------
elif page == "Offline Rescheduling Simulator":
    st.markdown('<div class="main-header">Offline Departure Rescheduling Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">What-if sensitivity analysis over departure schedule shifts under fixed historical context.</div>', unsafe_allow_html=True)

    st.warning(
        "**Methodological Notice**: This tool computes statistical what-if sensitivity scores based on historical schedule correlations. "
        "It does NOT establish causal delay reductions and does not model gate constraints, slot availability, crew duty legalities, or air traffic ground delay programs."
    )

    predictor = load_predictor("xgboost")
    if predictor is None:
        st.error("Predictor artifacts not found. Please train models first.")
    else:
        with st.form("sim_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                carrier = st.selectbox("Carrier", ["UA", "AA", "DL", "B6", "EV", "MQ", "WN"])
                origin = st.selectbox("Origin", ["EWR", "JFK", "LGA"])
                dest = st.text_input("Destination", "ORD").upper()
            with c2:
                dep_date = st.date_input("Flight Date")
                dep_time = st.time_input("Scheduled Departure", value=pd.to_datetime("17:30").time())
                arr_time = st.time_input("Scheduled Arrival", value=pd.to_datetime("19:45").time())
            with c3:
                distance = st.number_input("Distance (Miles)", value=719)
                shifts = st.multiselect("Candidate Shifts (Minutes)", [-60, -45, -30, -15, 0, 15, 30, 45, 60], default=[-60, -30, 0, 30, 60])

            sim_btn = st.form_submit_button("Simulate Schedule Shifts", use_container_width=True)

        if sim_btn and 0 in shifts:
            sched_dep_num = dep_time.hour * 100 + dep_time.minute
            sched_arr_num = arr_time.hour * 100 + arr_time.minute

            flight_dict = {
                "flight_date": str(dep_date),
                "sched_dep_time": sched_dep_num,
                "sched_arr_time": sched_arr_num,
                "carrier": carrier,
                "origin": origin,
                "dest": dest,
                "distance": distance,
            }

            prepped_df = predictor.preprocess_raw_input(pd.DataFrame([flight_dict]))
            flight_row = prepped_df.iloc[0]

            sim = ReschedulingSimulator(
                model_wrapper=predictor.model,
                preprocessor=predictor.preprocessor,
                calibrator=predictor.calibrator,
                is_pytorch=predictor.is_pytorch,
            )

            res = sim.simulate_shifts_for_flight(flight_row, candidate_shifts=sorted(shifts))
            df_scen = pd.DataFrame(res["scenarios"])

            st.subheader("Schedule Shift Sensitivity Scenarios")
            st.dataframe(df_scen, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Baseline Risk (0 min)", f"{res['baseline_delay_probability']*100:.1f}%")
                st.metric("Optimal Departure Shift", f"{res['optimal_shift_minutes']} mins")
            with col2:
                st.metric("Predicted Optimal Risk", f"{res['optimal_delay_probability']*100:.1f}%")
                st.metric("Absolute Risk Difference", f"{res['best_risk_difference']*100:+.1f}%")

            # Chart
            st.bar_chart(df_scen.set_index("shift_minutes")["predicted_delay_probability"])


# ------------------------------------------------------------------------------
# PAGE 4: Model Performance & Benchmarks
# ------------------------------------------------------------------------------
elif page == "Model Performance & Benchmarks":
    st.markdown('<div class="main-header">Model Performance & Chronological Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation on untouched chronological test partition (November 7 - December 31, 2013).</div>', unsafe_allow_html=True)

    df_comp = load_model_comparison()
    if df_comp is not None:
        st.subheader("Official Model Comparison Table")
        st.dataframe(df_comp, use_container_width=True)
    else:
        st.info("Comparison table not yet generated. Run `python scripts/run_pipeline.py`.")

    st.markdown("---")
    st.subheader("Evaluation Visualizations")

    col1, col2 = st.columns(2)
    roc_img = FIGURES_DIR / "roc_curves.png"
    pr_img = FIGURES_DIR / "pr_curves.png"
    cal_img = FIGURES_DIR / "calibration_curves.png"
    feat_img = FIGURES_DIR / "feature_importances.png"
    cm_img = FIGURES_DIR / "confusion_matrices.png"

    with col1:
        if roc_img.exists():
            st.image(str(roc_img), caption="Receiver Operating Characteristic (ROC) Comparison")
        if cal_img.exists():
            st.image(str(cal_img), caption="Reliability Curves (Isotonic Calibration)")

    with col2:
        if pr_img.exists():
            st.image(str(pr_img), caption="Precision-Recall Curves with Baseline Prevalence")
        if feat_img.exists():
            st.image(str(feat_img), caption="Top 20 Predictive Feature Importances (XGBoost)")

    if cm_img.exists():
        st.image(str(cm_img), caption="Confusion Matrices at Operational Decision Thresholds", use_container_width=True)


# ------------------------------------------------------------------------------
# PAGE 5: Exploratory Delay Analytics
# ------------------------------------------------------------------------------
elif page == "Exploratory Delay Analytics":
    st.markdown('<div class="main-header">Exploratory Delay Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical flight delay dynamics across schedules, hubs, and airline carriers.</div>', unsafe_allow_html=True)

    eda_hour = FIGURES_DIR / "eda_delay_by_hour.png"
    eda_carrier = FIGURES_DIR / "eda_delay_by_carrier.png"
    eda_origin = FIGURES_DIR / "eda_delay_by_origin.png"
    eda_month = FIGURES_DIR / "eda_delay_by_month.png"

    c1, c2 = st.columns(2)
    with c1:
        if eda_hour.exists():
            st.image(str(eda_hour), caption="Arrival Delay Rate by Scheduled Departure Hour")
        if eda_origin.exists():
            st.image(str(eda_origin), caption="Empirical Delay Rate by NYC Hub (EWR, JFK, LGA)")

    with c2:
        if eda_carrier.exists():
            st.image(str(eda_carrier), caption="Delay Rate by Operating Airline Carrier")
        if eda_month.exists():
            st.image(str(eda_month), caption="Seasonal Delay Fluctuation Across Calendar Months")


# ------------------------------------------------------------------------------
# PAGE 6: Methodology & Limitations
# ------------------------------------------------------------------------------
elif page == "Methodology & Limitations":
    st.markdown('<div class="main-header">Methodological Foundations & Limitations</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🛡️ Leakage-Safe Architectural Contract
    1. **Pre-Departure Information Boundary**: All feature transformations rely strictly on attributes established at the 2-hour pre-departure decision point. Post-flight outcomes (`dep_delay`, `arr_time`, `air_time`, delay causes) are strictly forbidden.
    2. **Past-Only Historical Smoothing**: To eliminate future-to-past leakage, all rolling delay statistics for a given calendar date $D$ aggregate data from completed flights on dates $d < D$ only.
    3. **Chronological Splitting**: Chronological date partitions (70% train, 15% validation, 15% test) replicate realistic forward deployment without random shuffle leakage.

    ---

    ### ⚠️ Limitations & Real-World Constraints
    - **No Direct Causal Interpretation**: Shifting departure time in the offline simulator does not guarantee delay reduction in the real world.
    - **Unobserved Confounders**: The dataset lacks real-time weather radar data, FAA Ground Delay Program (GDP) declarations, and terminal airspace congestion metrics.
    - **Operational Dependencies**: Aircraft tail rotations, minimum gate turn times, crew flight-time duty limitations (Part 117), and passenger connection graphs are not captured in the public flight record schema.
    """)
