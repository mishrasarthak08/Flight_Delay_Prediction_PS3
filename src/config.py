"""
Central configuration for Flight Delay Prediction pipeline.
Defines paths, feature allowlists, forbidden field lists, prediction contracts,
hyperparameters, and chronological split boundaries.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "flights.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
REPORTS_DIR = RESULTS_DIR / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

# Operational Prediction Contract
TARGET_COL: str = "is_delayed"
RAW_TARGET_COL: str = "arr_delay"
DELAY_THRESHOLD_MINUTES: float = 15.0
PREDICTION_HORIZON_HOURS: float = (
    2.0  # Operating timestamp: 2 hours prior to scheduled departure
)

# Centralized Feature Allowlist
CATEGORICAL_FEATURES: List[str] = [
    "carrier",
    "origin",
    "dest",
]

NUMERIC_SCHEDULE_FEATURES: List[str] = [
    "sched_dep_hour",
    "sched_dep_min",
    "sched_dep_min_of_day",
    "sin_sched_dep",
    "cos_sched_dep",
    "sched_arr_hour",
    "sched_arr_min",
    "sched_arr_min_of_day",
    "sin_sched_arr",
    "cos_sched_arr",
    "distance",
    "month",
    "day",
    "day_of_week",
    "is_weekend",
    "sin_day_of_week",
    "cos_day_of_week",
    "sin_month",
    "cos_month",
    "origin_hour_scheduled_departures",
    "origin_window_scheduled_departures",
]

HISTORICAL_FEATURES: List[str] = [
    "carrier_hist_delay_rate",
    "origin_hist_delay_rate",
    "dest_hist_delay_rate",
    "route_hist_delay_rate",
    "hist_carrier_flight_count",
    "hist_origin_flight_count",
    "hist_dest_flight_count",
    "hist_route_flight_count",
    "carrier_recent_delay_rate",
    "origin_recent_delay_rate",
    "route_recent_delay_rate",
]

ALL_MODEL_FEATURES: List[str] = (
    CATEGORICAL_FEATURES + NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES
)

# Explicit Forbidden Fields List (Leakage prevention)
FORBIDDEN_COLUMNS: List[str] = [
    "dep_time",
    "dep_delay",
    "arr_time",
    "arr_delay",
    "air_time",
    "flight",
    "tailnum",
    "id",
    "name",
    "time_hour",
    "actual_dep_time",
    "actual_arr_time",
    "actual_departure",
    "actual_arrival",
    "taxi_out",
    "taxi_in",
    "wheels_off",
    "wheels_on",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
    "cancellation_code",
    "cancelled",
    "diverted",
]

# Schema alias mapping candidates for dataset harmonization
SCHEMA_ALIASES: Dict[str, List[str]] = {
    "arr_delay": ["arr_delay", "ArrDelay", "ARR_DELAY", "ARRIVAL_DELAY"],
    "sched_dep_time": [
        "sched_dep_time",
        "CRSDepTime",
        "SCHEDULED_DEPARTURE",
        "sched_dep",
    ],
    "sched_arr_time": [
        "sched_arr_time",
        "CRSArrTime",
        "SCHEDULED_ARRIVAL",
        "sched_arr",
    ],
    "carrier": ["carrier", "UniqueCarrier", "AIRLINE", "OP_UNIQUE_CARRIER", "airline"],
    "origin": ["origin", "ORIGIN", "ORIGIN_AIRPORT", "origin_airport"],
    "dest": ["dest", "DEST", "DESTINATION_AIRPORT", "dest_airport"],
    "distance": ["distance", "DISTANCE", "Distance"],
    "year": ["year", "YEAR", "Year"],
    "month": ["month", "MONTH", "Month"],
    "day": ["day", "DAY", "Day", "DAY_OF_MONTH", "day_of_month"],
}


@dataclass
class SplittingConfig:
    """Chronological split parameters."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


@dataclass
class SmoothingConfig:
    """Empirical Bayes smoothing pseudo-counts for historical delay rates."""

    carrier_m: float = 20.0
    origin_m: float = 20.0
    dest_m: float = 20.0
    route_m: float = 10.0
    default_prior_delay_rate: float = 0.25
    recent_window_days: int = 7


@dataclass
class ModelHyperparams:
    """Hyperparameters for all model families."""

    # Logistic Regression
    lr_c: float = 0.1
    lr_max_iter: int = 1000
    lr_solver: str = "lbfgs"
    lr_class_weight: str = "balanced"

    # XGBoost
    xgb_n_estimators: int = 150
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.08
    xgb_subsample: float = 1.0
    xgb_colsample_bytree: float = 1.0
    xgb_tree_method: str = "hist"
    xgb_n_jobs: int = 2
    xgb_early_stopping_rounds: int = 15
    xgb_eval_metric: str = "logloss"
    xgb_reg_lambda: float = 5.0
    xgb_min_child_weight: float = 20.0

    # PyTorch Embedding MLP
    mlp_carrier_emb_dim: int = 16
    mlp_origin_emb_dim: int = 16
    mlp_dest_emb_dim: int = 32
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    mlp_dropout: float = 0.2
    mlp_learning_rate: float = 0.001
    mlp_weight_decay: float = 1e-4
    mlp_batch_size: int = 512
    mlp_epochs: int = 20
    mlp_early_stopping_patience: int = 5


@dataclass
class EvaluationConfig:
    """Evaluation, operational thresholding, and calibration parameters."""

    target_precision: float = 0.70
    default_threshold: float = 0.50
    calibration_method: str = "isotonic"  # 'isotonic' or 'sigmoid' (Platt scaling)


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""

    raw_data_path: Path = RAW_DATA_PATH
    processed_data_dir: Path = PROCESSED_DATA_DIR
    models_dir: Path = MODELS_DIR
    results_dir: Path = RESULTS_DIR
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    hyperparams: ModelHyperparams = field(default_factory=ModelHyperparams)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        d = asdict(self)
        # Convert Path objects to strings for serialization
        d["raw_data_path"] = str(self.raw_data_path)
        d["processed_data_dir"] = str(self.processed_data_dir)
        d["models_dir"] = str(self.models_dir)
        d["results_dir"] = str(self.results_dir)
        return d
