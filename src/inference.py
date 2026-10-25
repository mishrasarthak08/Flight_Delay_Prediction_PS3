"""
End-to-end inference pipeline for flight delay prediction.
Loads trained models, preprocessing artifacts, historical lookup state,
and applies identical transformations used during training.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from typing import Dict, Any, Optional
from pathlib import Path
import json
import pickle
import pandas as pd
import torch

from src.config import (
    MODELS_DIR,
    CATEGORICAL_FEATURES,
    NUMERIC_SCHEDULE_FEATURES,
    HISTORICAL_FEATURES,
)
from src.features import (
    compute_schedule_and_calendar_features,
    HistoricalFeatureGenerator,
    ScheduleContext,
)
from src.validation import (
    validate_inference_features,
    audit_feature_matrix_for_forbidden_columns,
)
from src.models import (
    FlightEmbeddingMLP,
)
from src.calibration import ModelCalibrator


class FlightDelayPredictor:
    """
    Historical inference engine using the saved evaluation protocol.
    """

    def __init__(
        self,
        model_name: str = "xgboost",
        models_dir: Path = MODELS_DIR,
        device: Optional[torch.device] = None,
    ):
        aliases = {
            "lr": "logistic_regression",
            "mlp": "pytorch_mlp",
            "pytorch": "pytorch_mlp",
        }
        self.model_name = aliases.get(model_name.lower(), model_name.lower())
        self.models_dir = Path(models_dir)
        self.device = device or torch.device("cpu")
        metadata_path = self.models_dir / "bundle_metadata.json"
        self.metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        context_path = self.models_dir / "schedule_context.json"
        self.schedule_context = (
            ScheduleContext.from_dict(json.loads(context_path.read_text()))
            if context_path.exists()
            else None
        )

        # Load historical generator
        hist_path = self.models_dir / "historical_generator.json"
        if not hist_path.exists():
            raise FileNotFoundError(
                f"Historical feature generator state not found at {hist_path}."
            )
        with open(hist_path, "r") as f:
            self.hist_gen = HistoricalFeatureGenerator.from_dict(json.load(f))
        if self.hist_gen.history_policy == "daily_available_outcomes":
            self.hist_gen.source = pd.read_parquet(
                self.models_dir / "historical_events.parquet"
            )
            self.hist_gen.target_col = "is_delayed"

        # Load operational threshold & calibration metadata
        thresh_path = self.models_dir / f"{self.model_name}_threshold.json"
        if thresh_path.exists():
            with open(thresh_path, "r") as f:
                self.threshold_meta = json.load(f)
                self.operational_threshold = float(
                    self.threshold_meta.get("selected_threshold", 0.50)
                )
        else:
            self.operational_threshold = 0.50
            self.threshold_meta = {
                "selected_threshold": 0.50,
                "target_precision_met": False,
            }

        # Load calibrator if available
        cal_path = self.models_dir / f"{self.model_name}_calibrator.pkl"
        if cal_path.exists():
            self.calibrator = ModelCalibrator.load(cal_path)
        else:
            self.calibrator = None

        # Load model and preprocessor
        if self.model_name == "xgboost":
            with open(self.models_dir / "xgboost.pkl", "rb") as f:
                self.model = pickle.load(f)
            from src.preprocessing import FlightTabularPreprocessor

            self.preprocessor = FlightTabularPreprocessor.load(
                self.models_dir / "xgboost_preprocessor.pkl"
            )
            self.is_pytorch = False

        elif self.model_name in ("logistic_regression", "lr"):
            with open(self.models_dir / "logistic_regression.pkl", "rb") as f:
                self.model = pickle.load(f)
            from src.preprocessing import FlightTabularPreprocessor

            path = self.models_dir / "logistic_regression_preprocessor.pkl"
            if not path.exists():
                path = (
                    self.models_dir / "tabular_preprocessor.pkl"
                )  # Synthetic legacy test fixtures.
            self.preprocessor = FlightTabularPreprocessor.load(path)
            self.is_pytorch = False

        elif self.model_name in ("pytorch", "pytorch_mlp", "mlp"):
            from src.preprocessing import PyTorchPreprocessor

            self.preprocessor = PyTorchPreprocessor.load(
                self.models_dir / "pytorch_preprocessor.pkl"
            )
            self.model = FlightEmbeddingMLP.load_checkpoint(
                self.models_dir / "pytorch_mlp.pt", device=self.device
            )
            self.is_pytorch = True
        else:
            raise ValueError(
                f"Unknown model_name: {model_name}. Choose 'xgboost', 'logistic_regression', or 'pytorch_mlp'."
            )

    def preprocess_raw_input(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes raw pre-departure inputs into full model features.
        Audits against forbidden outcome columns.
        """
        # Ensure forbidden fields are strictly absent
        audit_feature_matrix_for_forbidden_columns(input_df.columns, strict=True)

        # 1. Compute schedule & calendar cyclical features
        required = {
            "carrier",
            "origin",
            "dest",
            "distance",
            "sched_dep_time",
            "sched_arr_time",
        }
        if input_df.empty or not required.issubset(input_df.columns):
            raise ValueError(
                f"A nonempty flight table with columns {sorted(required)} is required."
            )
        df_feat = compute_schedule_and_calendar_features(input_df)
        cutoff = self.metadata.get("policy_available_at")
        if cutoff and (df_feat["decision_time"] < pd.Timestamp(cutoff)).any():
            raise ValueError(
                "Prediction precedes the fitted model/calibration/threshold policy cutoff."
            )
        if self.schedule_context is not None:
            dates = df_feat["flight_date"].dt.strftime("%Y-%m-%d")
            if (
                (dates < self.schedule_context.date_min)
                | (dates > self.schedule_context.date_max)
            ).any():
                raise ValueError(
                    "No timetable context for this date. This bundle is a historical 2013 demonstration."
                )
            df_feat = self.schedule_context.transform(df_feat)
        else:
            needed = {
                "origin_hour_scheduled_departures",
                "origin_window_scheduled_departures",
            }
            if not needed.issubset(df_feat.columns):
                raise ValueError(
                    "Timetable context is required; provide scheduled-demand features or a schedule_context artifact."
                )

        # 2. Attach historical features from fitted generator state
        df_full = self.hist_gen.transform_inference(df_feat)

        # 3. Validate feature completeness
        feature_cols = (
            CATEGORICAL_FEATURES + NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES
        )
        validate_inference_features(df_full, expected_features=feature_cols)
        return df_full

    def predict_dataframe(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs batch prediction on dataframe of flights.
        Returns dataframe with raw probability, calibrated probability, and operational prediction.
        """
        df_full = self.preprocess_raw_input(input_df)
        feature_cols = (
            CATEGORICAL_FEATURES + NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES
        )

        if self.is_pytorch:
            c_tens, n_tens, _ = self.preprocessor.transform(df_full[feature_cols])
            probs_raw = self.model.predict_proba(c_tens, n_tens, device=self.device)
        else:
            X_trans = self.preprocessor.transform(df_full[feature_cols])
            probs_raw = self.model.predict_proba(X_trans)

        if self.calibrator is not None:
            probs_cal = self.calibrator.predict_proba(probs_raw)
        else:
            probs_cal = probs_raw

        y_pred = (probs_cal >= self.operational_threshold).astype(int)

        results = df_full.copy()
        results["predicted_delay_prob_raw"] = probs_raw
        results["predicted_delay_prob_calibrated"] = probs_cal
        results["predicted_is_delayed"] = y_pred
        results["operational_threshold_used"] = self.operational_threshold

        def assign_risk_tier(p: float) -> str:
            if p < 0.20:
                return "Low Risk"
            elif p < 0.45:
                return "Moderate Risk"
            else:
                return "High Risk"

        results["risk_tier"] = [assign_risk_tier(p) for p in probs_cal]
        return results

    def predict_single(self, flight_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts delay for a single flight dictionary.
        """
        df = pd.DataFrame([flight_dict])
        pred_df = self.predict_dataframe(df)
        row = pred_df.iloc[0]

        return {
            "model_name": self.model_name,
            "carrier": row.get("carrier", "N/A"),
            "origin": row.get("origin", "N/A"),
            "dest": row.get("dest", "N/A"),
            "flight_date": str(flight_dict.get("flight_date", "N/A")),
            "sched_dep_time": int(flight_dict.get("sched_dep_time", 0)),
            "sched_arr_time": int(flight_dict.get("sched_arr_time", 0)),
            "distance": float(flight_dict.get("distance", 0)),
            "predicted_delay_prob_raw": float(row["predicted_delay_prob_raw"]),
            "predicted_delay_prob_calibrated": float(
                row["predicted_delay_prob_calibrated"]
            ),
            "predicted_is_delayed": int(row["predicted_is_delayed"]),
            "operational_threshold": float(self.operational_threshold),
            "risk_tier": str(row["risk_tier"]),
        }
