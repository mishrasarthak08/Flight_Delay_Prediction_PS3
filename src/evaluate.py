"""
Model evaluation module.
Computes PR-AUC, ROC-AUC, Brier score, Recall at operational precision threshold,
confusion matrices, and reliability metrics on untouched chronological test splits.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from src.config import METRICS_DIR, PREDICTIONS_DIR


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_calibrated: Optional[np.ndarray] = None,
    operational_threshold: float = 0.50,
    model_name: str = "model",
    split_name: str = "test",
) -> Dict[str, Any]:
    """
    Computes rigorous evaluation metrics for a model's probability predictions.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob_raw = np.asarray(y_prob_raw).astype(np.float64)
    y_prob = np.asarray(y_prob_calibrated if y_prob_calibrated is not None else y_prob_raw).astype(np.float64)

    # 1. Ranking & Probability metrics
    pr_auc = float(average_precision_score(y_true, y_prob_raw))
    roc_auc = float(roc_auc_score(y_true, y_prob_raw))
    brier_raw = float(brier_score_loss(y_true, y_prob_raw))
    brier_calibrated = float(brier_score_loss(y_true, y_prob)) if y_prob_calibrated is not None else brier_raw

    # 2. Operating threshold metrics (operational_threshold)
    y_pred_op = (y_prob >= operational_threshold).astype(int)
    prec_op = float(precision_score(y_true, y_pred_op, zero_division=0))
    rec_op = float(recall_score(y_true, y_pred_op, zero_division=0))
    f1_op = float(f1_score(y_true, y_pred_op, zero_division=0))
    cm_op = confusion_matrix(y_true, y_pred_op)
    tn_op, fp_op, fn_op, tp_op = [int(v) for v in cm_op.ravel()]

    # 3. Default threshold metrics (0.50)
    y_pred_def = (y_prob >= 0.50).astype(int)
    prec_def = float(precision_score(y_true, y_pred_def, zero_division=0))
    rec_def = float(recall_score(y_true, y_pred_def, zero_division=0))
    f1_def = float(f1_score(y_true, y_pred_def, zero_division=0))
    cm_def = confusion_matrix(y_true, y_pred_def)
    tn_def, fp_def, fn_def, tp_def = [int(v) for v in cm_def.ravel()]

    # 4. Calibration reliability curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    # 5. Baseline reference (Prevalence)
    prevalence = float(np.mean(y_true))

    metrics = {
        "model_name": model_name,
        "split": split_name,
        "num_samples": len(y_true),
        "positive_count": int(np.sum(y_true)),
        "positive_prevalence": prevalence,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score_raw": brier_raw,
        "brier_score_calibrated": brier_calibrated,
        "operational_threshold": float(operational_threshold),
        "precision_at_op_threshold": prec_op,
        "recall_at_op_threshold": rec_op,
        "f1_at_op_threshold": f1_op,
        "confusion_matrix_op": {"tn": tn_op, "fp": fp_op, "fn": fn_op, "tp": tp_op},
        "default_threshold": 0.50,
        "precision_at_default": prec_def,
        "recall_at_default": rec_def,
        "f1_at_default": f1_def,
        "confusion_matrix_default": {"tn": tn_def, "fp": fp_def, "fn": fn_def, "tp": tp_def},
        "calibration_curve": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        },
    }
    return metrics


def save_test_predictions(
    record_ids: pd.Series,
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_calibrated: np.ndarray,
    y_pred_op: np.ndarray,
    output_path: Path,
) -> None:
    """
    Saves clean, lightweight prediction artifact for reproducibility and auditing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_pred = pd.DataFrame({
        "record_id": record_ids.values,
        "y_true": y_true,
        "prob_raw": np.round(y_prob_raw, 5),
        "prob_calibrated": np.round(y_prob_calibrated, 5),
        "y_pred_operational": y_pred_op,
    })
    df_pred.to_csv(output_path, index=False)


def create_model_comparison_table(all_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Consolidates evaluation metrics across models into a clean comparative summary table.
    """
    rows = []
    for m in all_metrics:
        rows.append({
            "Model": m["model_name"],
            "PR-AUC (Avg Prec)": round(m["pr_auc"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "Brier Score (Raw)": round(m["brier_score_raw"], 4),
            "Brier Score (Calibrated)": round(m["brier_score_calibrated"], 4),
            "Operational Threshold": round(m["operational_threshold"], 3),
            "Precision @ Op Threshold": round(m["precision_at_op_threshold"], 4),
            "Recall @ Op Threshold": round(m["recall_at_op_threshold"], 4),
            "F1 @ Op Threshold": round(m["f1_at_op_threshold"], 4),
            "F1 @ 0.50 Threshold": round(m["f1_at_default"], 4),
            "Test Prevalence": round(m["positive_prevalence"], 4),
        })
    return pd.DataFrame(rows)
