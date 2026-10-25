"""
Probability calibration and operational threshold selection module.
Calibrators and operating thresholds are fit EXCLUSIVELY on validation predictions,
preserving separation of policy selection from test labels.
"""

from typing import Dict, Any
import pickle
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve


class ModelCalibrator:
    """
    Fits probability calibrator (Isotonic Regression or Platt scaling / Sigmoid)
    exclusively on validation set predictions.
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = None
        self.is_fitted = False

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "ModelCalibrator":
        """Fits calibrator strictly on validation probabilities and labels."""
        val_probs = np.clip(val_probs, 1e-6, 1.0 - 1e-6)
        if self.method == "none":
            self.is_fitted = True
            return self
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0
            )
            self.calibrator.fit(val_probs, y_val)
        elif self.method == "sigmoid":
            # Platt scaling: LogisticRegression on log-odds
            log_odds = np.log(val_probs / (1.0 - val_probs)).reshape(-1, 1)
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(log_odds, y_val)
        else:
            raise ValueError(
                f"Unknown calibration method: {self.method}. Choose 'isotonic' or 'sigmoid'."
            )

        self.is_fitted = True
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """Transforms uncalibrated probabilities into calibrated probabilities."""
        if not self.is_fitted or self.method == "none":
            return probs
        probs_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic":
            cal_probs = self.calibrator.predict(probs_clipped)
        else:
            log_odds = np.log(probs_clipped / (1.0 - probs_clipped)).reshape(-1, 1)
            cal_probs = self.calibrator.predict_proba(log_odds)[:, 1]
        return np.clip(cal_probs, 0.0, 1.0)

    def save(self, filepath: Path) -> None:
        """Saves fitted calibrator."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "ModelCalibrator":
        with open(filepath, "rb") as f:
            return pickle.load(f)


def select_operational_threshold(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    target_precision: float = 0.70,
    default_threshold: float = 0.50,
    min_alerts: int = 100,
) -> Dict[str, Any]:
    """
    Selects the decision threshold on validation data that maximizes recall
    subject to achieving at least `target_precision`.
    Require minimum alert support. If target precision is infeasible, maximize
    F1 on the threshold-development block and explicitly report infeasibility.
    """
    val_probs, y_val = np.asarray(val_probs), np.asarray(y_val)
    if len(val_probs) == 0 or len(np.unique(y_val)) < 2:
        raise ValueError(
            "Threshold selection needs a nonempty, two-class development sample."
        )
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    # Note: precision_recall_curve returns thresholds of length n-1 (last precision=1, recall=0 has no threshold)

    alert_counts = len(val_probs) - np.searchsorted(
        np.sort(val_probs), thresholds, side="left"
    )
    supported = alert_counts >= min(min_alerts, len(val_probs))
    valid_indices = np.where((precisions[:-1] >= target_precision) & supported)[0]

    if len(valid_indices) > 0:
        # Choose the threshold that maximizes recall among those meeting target precision
        best_idx = valid_indices[np.argmax(recalls[valid_indices])]
        selected_threshold = float(thresholds[best_idx])
        achieved_precision = float(precisions[best_idx])
        achieved_recall = float(recalls[best_idx])
        target_met = True
    else:
        # Explicit supported F1 fallback; do not claim the precision constraint is met.
        f1s = (
            2
            * precisions[:-1]
            * recalls[:-1]
            / (precisions[:-1] + recalls[:-1] + 1e-12)
        )
        best_idx = int(np.argmax(np.where(supported, f1s, -1)))
        selected_threshold = float(thresholds[best_idx])
        achieved_precision = float(precisions[best_idx])
        achieved_recall = float(recalls[best_idx])
        target_met = False

    # Also compute F1 at selected threshold
    f1 = (
        2
        * (achieved_precision * achieved_recall)
        / (achieved_precision + achieved_recall + 1e-9)
    )

    return {
        "target_precision": target_precision,
        "selected_threshold": selected_threshold,
        "validation_precision_at_threshold": achieved_precision,
        "validation_recall_at_threshold": achieved_recall,
        "validation_f1_at_threshold": float(f1),
        "target_precision_met": target_met,
        "selection_dataset": "threshold_validation",
        "probability_scale": "selected_calibrator_output",
        "selection_rule": "max_recall_at_target_precision_else_max_f1_with_minimum_alert_support",
        "minimum_alerts": min_alerts,
        "validation_alert_count": int(alert_counts[best_idx]),
        "default_threshold": default_threshold,
    }
