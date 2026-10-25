"""
Probability calibration and operational threshold selection module.
Calibrators and operating thresholds are fit EXCLUSIVELY on validation predictions,
preserving complete separation from untouched test-set labels.
"""

from typing import Dict, Any, Tuple, Optional
import pickle
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, brier_score_loss


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
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(val_probs, y_val)
        elif self.method == "sigmoid":
            # Platt scaling: LogisticRegression on log-odds
            log_odds = np.log(val_probs / (1.0 - val_probs)).reshape(-1, 1)
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(log_odds, y_val)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}. Choose 'isotonic' or 'sigmoid'.")

        self.is_fitted = True
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """Transforms uncalibrated probabilities into calibrated probabilities."""
        if not self.is_fitted:
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
    default_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Selects the decision threshold on validation data that maximizes recall
    subject to achieving at least `target_precision`.
    If no threshold achieves target precision, selects the threshold achieving
    the highest precision on the validation set and documents this constraint.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    # Note: precision_recall_curve returns thresholds of length n-1 (last precision=1, recall=0 has no threshold)

    valid_indices = np.where(precisions[:-1] >= target_precision)[0]

    if len(valid_indices) > 0:
        # Choose the threshold that maximizes recall among those meeting target precision
        best_idx = valid_indices[np.argmax(recalls[valid_indices])]
        selected_threshold = float(thresholds[best_idx])
        achieved_precision = float(precisions[best_idx])
        achieved_recall = float(recalls[best_idx])
        target_met = True
    else:
        # Target precision cannot be met on validation set. Select threshold yielding highest precision.
        best_idx = int(np.argmax(precisions[:-1]))
        selected_threshold = float(thresholds[best_idx])
        achieved_precision = float(precisions[best_idx])
        achieved_recall = float(recalls[best_idx])
        target_met = False

    # Also compute F1 at selected threshold
    f1 = 2 * (achieved_precision * achieved_recall) / (achieved_precision + achieved_recall + 1e-9)

    return {
        "target_precision": target_precision,
        "selected_threshold": selected_threshold,
        "validation_precision_at_threshold": achieved_precision,
        "validation_recall_at_threshold": achieved_recall,
        "validation_f1_at_threshold": float(f1),
        "target_precision_met": target_met,
        "selection_dataset": "validation",
        "default_threshold": default_threshold,
    }
