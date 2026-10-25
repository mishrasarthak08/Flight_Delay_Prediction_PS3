"""
Publication-grade visualization module.
Generates ROC curves, PR curves, calibration diagrams, confusion matrices,
feature importances, and exploratory data analysis plots.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from src.config import FIGURES_DIR

# Modern, clean aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_roc_curves(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_test: np.ndarray,
    output_path: Path = FIGURES_DIR / "roc_curves.png",
) -> None:
    """
    Plots comparative ROC curves for all models.
    models_predictions: {model_name: (probs_raw, probs_calibrated)}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    for (name, (p_raw, _)), color in zip(models_predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, p_raw)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, p_raw)
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (ROC-AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pr_curves(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_test: np.ndarray,
    output_path: Path = FIGURES_DIR / "pr_curves.png",
) -> None:
    """
    Plots comparative Precision-Recall curves.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    prevalence = float(np.mean(y_test))

    for (name, (p_raw, _)), color in zip(models_predictions.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, p_raw)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_test, p_raw)
        ax.plot(rec, prec, color=color, lw=2.2, label=f"{name} (PR-AUC = {ap:.3f})")

    ax.axhline(prevalence, color="gray", linestyle="--", lw=1.5, label=f"No-Skill Baseline ({prevalence:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (PR) Curves on Chronological Test Set")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_calibration_curves(
    all_metrics: List[Dict[str, Any]],
    output_path: Path = FIGURES_DIR / "calibration_curves.png",
) -> None:
    """
    Plots reliability curves (calibration diagrams) for evaluated models.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    for m, color in zip(all_metrics, colors):
        name = m["model_name"]
        cal_data = m["calibration_curve"]
        prob_pred = cal_data["prob_pred"]
        prob_true = cal_data["prob_true"]
        brier = m["brier_score_calibrated"]
        ax.plot(prob_pred, prob_true, marker="o", color=color, lw=2, label=f"{name} (Brier = {brier:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly Calibrated")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Empirical)")
    ax.set_title("Reliability Diagram (Probability Calibration)")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrices(
    all_metrics: List[Dict[str, Any]],
    output_path: Path = FIGURES_DIR / "confusion_matrices.png",
) -> None:
    """
    Plots side-by-side confusion matrices at operational decision thresholds.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_models = len(all_metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, m in zip(axes, all_metrics):
        cm_dict = m["confusion_matrix_op"]
        cm = np.array([
            [cm_dict["tn"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tp"]]
        ])
        thresh = m["operational_threshold"]
        name = m["model_name"]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["On-Time", "Delayed"],
            yticklabels=["On-Time", "Delayed"],
        )
        ax.set_title(f"{name}\n(Threshold = {thresh:.3f})")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importances(
    importances: Dict[str, float],
    top_n: int = 20,
    output_path: Path = FIGURES_DIR / "feature_importances.png",
) -> None:
    """
    Plots top-N feature importances for tree-based models.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [x[0] for x in reversed(sorted_items)]
    scores = [x[1] for x in reversed(sorted_items)]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(features, scores, color="#2b5c8f", edgecolor="none", height=0.7)
    ax.set_xlabel("Relative Feature Importance (Gain / Weight)")
    ax.set_title(f"Top {top_n} Predictive Features (XGBoost)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_delay_eda(
    df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> None:
    """
    Generates core exploratory delay analytics figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    from src.config import TARGET_COL

    # 1. Delay rate by departure hour
    if "sched_dep_hour" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        hourly = df.groupby("sched_dep_hour")[TARGET_COL].agg(["mean", "count"]).reset_index()
        ax.plot(hourly["sched_dep_hour"], hourly["mean"] * 100.0, marker="o", color="#d95f02", lw=2.2)
        ax.set_xlabel("Scheduled Departure Hour (24h)")
        ax.set_ylabel("Delay Rate (%)")
        ax.set_title("Flight Arrival Delay Rate by Scheduled Hour of Day")
        ax.set_xticks(range(5, 24))
        plt.tight_layout()
        plt.savefig(output_dir / "eda_delay_by_hour.png", dpi=300)
        plt.close()

    # 2. Delay rate by carrier
    if "carrier" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        carrier_stats = df.groupby("carrier")[TARGET_COL].agg(["mean", "count"]).reset_index()
        carrier_stats = carrier_stats[carrier_stats["count"] >= 50].sort_values("mean", ascending=False)
        sns.barplot(data=carrier_stats, x="carrier", y="mean", color="#2b5c8f", ax=ax)
        ax.set_xlabel("Operating Airline Carrier")
        ax.set_ylabel("Delay Rate (Fraction >= 15 min)")
        ax.set_title("Empirical Arrival Delay Rate by Airline Carrier")
        plt.tight_layout()
        plt.savefig(output_dir / "eda_delay_by_carrier.png", dpi=300)
        plt.close()

    # 3. Delay rate by origin airport
    if "origin" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        origin_stats = df.groupby("origin")[TARGET_COL].mean().reset_index()
        sns.barplot(data=origin_stats, x="origin", y=TARGET_COL, palette="Blues_r", ax=ax)
        ax.set_xlabel("Origin Airport")
        ax.set_ylabel("Delay Rate")
        ax.set_title("Arrival Delay Rate by Origin Airport (NYC Hubs)")
        plt.tight_layout()
        plt.savefig(output_dir / "eda_delay_by_origin.png", dpi=300)
        plt.close()

    # 4. Seasonal delay pattern by month
    if "month" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        monthly = df.groupby("month")[TARGET_COL].mean().reset_index()
        ax.plot(monthly["month"], monthly[TARGET_COL] * 100.0, marker="s", color="#7570b3", lw=2.2)
        ax.set_xlabel("Month of Year (2013)")
        ax.set_ylabel("Delay Rate (%)")
        ax.set_title("Seasonal Flight Delay Variation Across Months")
        ax.set_xticks(range(1, 13))
        plt.tight_layout()
        plt.savefig(output_dir / "eda_delay_by_month.png", dpi=300)
        plt.close()
