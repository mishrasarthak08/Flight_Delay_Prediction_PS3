"""
End-to-End Flight Delay Prediction Pipeline.
Executes data ingestion, schema discovery, chronological splitting,
leakage-safe historical feature engineering, model training (LR, XGB, PyTorch MLP),
validation-based calibration & thresholding, test evaluation, rescheduling simulation,
and reporting.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Dict, Any, List, Tuple
from pathlib import Path
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import torch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    RAW_DATA_PATH,
    RESULTS_DIR,
    MODELS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    TARGET_COL,
    CATEGORICAL_FEATURES,
    NUMERIC_SCHEDULE_FEATURES,
    HISTORICAL_FEATURES,
    ALL_MODEL_FEATURES,
    SplittingConfig,
    SmoothingConfig,
    ModelHyperparams,
    EvaluationConfig,
    PipelineConfig,
)
from src.data import load_raw_dataset, clean_and_prepare_dataset
from src.splitting import chronological_date_split, save_split_manifest
from src.features import compute_schedule_and_calendar_features, HistoricalFeatureGenerator
from src.validation import audit_feature_matrix_for_forbidden_columns, verify_no_future_leakage_in_splits
from src.train import train_logistic_regression, train_xgboost, train_pytorch_mlp
from src.calibration import ModelCalibrator, select_operational_threshold
from src.evaluate import evaluate_predictions, save_test_predictions, create_model_comparison_table
from src.reschedule import ReschedulingSimulator
from src.plotting import (
    plot_roc_curves,
    plot_pr_curves,
    plot_calibration_curves,
    plot_confusion_matrices,
    plot_feature_importances,
    plot_delay_eda,
)


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def run_pipeline(
    data_path: Path = RAW_DATA_PATH,
    results_dir: Path = RESULTS_DIR,
    models_dir: Path = MODELS_DIR,
    random_seed: int = 42,
    target_precision: float = 0.70,
    mlp_epochs: int = 15,
) -> Dict[str, Any]:
    print("=" * 80)
    print("STARTING FLIGHT DELAY PREDICTION PIPELINE")
    print(f"Data Path: {data_path}")
    print(f"Results Directory: {results_dir}")
    print(f"Models Directory: {models_dir}")
    print(f"Random Seed: {random_seed}")
    print("=" * 80)

    # 1. Ingestion & Schema Discovery
    print("\n[Stage 1/8] Ingesting and Auditing Dataset...")
    df_raw = load_raw_dataset(data_path)
    df_clean, audit_stats = clean_and_prepare_dataset(df_raw)
    reports_dir = results_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "dataset_profile.json", "w") as f:
        json.dump(audit_stats, f, indent=2)
    print(f"Dataset profile saved. Operated flights: {len(df_clean):,}, Delay rate: {audit_stats['positive_class_prevalence']:.4f}")

    # 2. Schedule and Calendar Feature Engineering
    print("\n[Stage 2/8] Computing Pre-Departure Schedule and Cyclical Features...")
    df_features = compute_schedule_and_calendar_features(df_clean)

    # 3. Chronological Date-Based Splitting
    print("\n[Stage 3/8] Chronological Date Splitting (70% Train, 15% Val, 15% Test)...")
    split_cfg = SplittingConfig(random_seed=random_seed)
    train_df, val_df, test_df, split_manifest = chronological_date_split(df_features, config=split_cfg)
    save_split_manifest(split_manifest, reports_dir / "split_manifest.json")
    print(f"Train: {split_manifest['train']['start_date']} to {split_manifest['train']['end_date']} ({split_manifest['train']['row_count']:,} flights, prev={split_manifest['train']['positive_prevalence']:.4f})")
    print(f"Val:   {split_manifest['validation']['start_date']} to {split_manifest['validation']['end_date']} ({split_manifest['validation']['row_count']:,} flights, prev={split_manifest['validation']['positive_prevalence']:.4f})")
    print(f"Test:  {split_manifest['test']['start_date']} to {split_manifest['test']['end_date']} ({split_manifest['test']['row_count']:,} flights, prev={split_manifest['test']['positive_prevalence']:.4f})")

    # Verify split boundaries
    verify_no_future_leakage_in_splits(train_df["flight_date"], val_df["flight_date"], test_df["flight_date"])

    # 4. Leakage-Safe Past-Only Empirical Bayes Historical Rates
    print("\n[Stage 4/8] Generating Leakage-Safe Historical Delay Rates (Past-Only Dates)...")
    # Fit historical generator across chronological dates of training data
    smoothing_cfg = SmoothingConfig(default_prior_delay_rate=split_manifest["train"]["positive_prevalence"])
    hist_gen = HistoricalFeatureGenerator(config=smoothing_cfg)

    # In sequential deployment simulation:
    # Train dates get historical stats strictly from earlier train dates d < D
    train_feat = hist_gen.fit_transform_chronological(train_df, date_col="flight_date", target_col=TARGET_COL)

    # Save historical generator state up through end of training
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "historical_generator.json", "w") as f:
        json.dump(hist_gen.to_dict(), f, indent=2)

    # Sequential forward simulation for Val and Test:
    # Each date uses historical statistics accumulated strictly up to the day before that date!
    # Concatenate val and test sequentially to continue the forward simulation without self-date leakage
    val_feat = hist_gen.fit_transform_chronological(val_df, date_col="flight_date", target_col=TARGET_COL)
    test_feat = hist_gen.fit_transform_chronological(test_df, date_col="flight_date", target_col=TARGET_COL)

    # 5. Feature Audit (Leakage Verification)
    print("\n[Stage 5/8] Auditing Feature Matrices for Forbidden Post-Flight Columns...")
    all_features = CATEGORICAL_FEATURES + NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES
    audit_feature_matrix_for_forbidden_columns(train_feat[all_features].columns, strict=True)
    audit_feature_matrix_for_forbidden_columns(val_feat[all_features].columns, strict=True)
    audit_feature_matrix_for_forbidden_columns(test_feat[all_features].columns, strict=True)
    print("Zero leakage verified: No forbidden outcome columns present in model feature sets.")

    # 6. Model Training
    print("\n[Stage 6/8] Training Models on Training Partition...")
    hparams = ModelHyperparams(mlp_epochs=mlp_epochs)

    # Model A: Logistic Regression
    print("  -> Training Model A: Logistic Regression Pipeline...")
    lr_model, tabular_prep, lr_train_probs, lr_val_probs = train_logistic_regression(
        train_feat, val_feat, hyperparams=hparams, random_seed=random_seed, output_dir=models_dir
    )

    # Model B: XGBoost
    print("  -> Training Model B: XGBoost Classifier...")
    xgb_model, xgb_train_probs, xgb_val_probs = train_xgboost(
        train_feat, val_feat, tabular_prep, hyperparams=hparams, random_seed=random_seed, output_dir=models_dir
    )

    # Model C: PyTorch Embedding MLP
    print("  -> Training Model C: PyTorch Categorical Embedding MLP...")
    device = torch.device("cpu")
    print(f"     (Using compute device: {device})")
    mlp_model, pytorch_prep, mlp_train_probs, mlp_val_probs = train_pytorch_mlp(
        train_feat, val_feat, hyperparams=hparams, random_seed=random_seed, device=device, output_dir=models_dir
    )

    # 7. Validation Calibration & Threshold Selection
    print("\n[Stage 7/8] Fitting Calibration and Selecting Operational Decision Thresholds on Validation Data...")
    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    models = {
        "Logistic Regression": {
            "model": lr_model,
            "val_probs": lr_val_probs,
            "test_feat": tabular_prep.transform(test_feat[all_features]),
            "is_pytorch": False,
            "save_name": "logistic_regression",
        },
        "XGBoost": {
            "model": xgb_model,
            "val_probs": xgb_val_probs,
            "test_feat": tabular_prep.transform(test_feat[all_features]),
            "is_pytorch": False,
            "save_name": "xgboost",
        },
        "PyTorch Embedding MLP": {
            "model": mlp_model,
            "val_probs": mlp_val_probs,
            "test_feat": test_feat[all_features],
            "is_pytorch": True,
            "save_name": "pytorch_mlp",
        },
    }

    all_metrics: List[Dict[str, Any]] = []
    test_prob_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    predictions_dir = results_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    record_ids = test_df["id"] if "id" in test_df else pd.Series(range(len(test_df)))

    for name, info in models.items():
        val_p = info["val_probs"]
        sname = info["save_name"]

        # Select operating threshold on validation predictions
        thresh_info = select_operational_threshold(val_p, y_val, target_precision=target_precision)
        op_thresh = thresh_info["selected_threshold"]

        # Save threshold info
        with open(models_dir / f"{sname}_threshold.json", "w") as f:
            json.dump(thresh_info, f, indent=2)

        # Fit probability calibrator on validation predictions
        calibrator = ModelCalibrator(method="isotonic")
        calibrator.fit(val_p, y_val)
        calibrator.save(models_dir / f"{sname}_calibrator.pkl")

        # Untouched Chronological Test Set Predictions
        if info["is_pytorch"]:
            c_test, n_test, _ = pytorch_prep.transform(info["test_feat"])
            test_probs_raw = mlp_model.predict_proba(c_test, n_test, device=torch.device("cpu"))
        else:
            test_probs_raw = info["model"].predict_proba(info["test_feat"])

        test_probs_calibrated = calibrator.predict_proba(test_probs_raw)
        test_prob_dict[name] = (test_probs_raw, test_probs_calibrated)

        # Evaluate on test set
        metrics = evaluate_predictions(
            y_true=y_test,
            y_prob_raw=test_probs_raw,
            y_prob_calibrated=test_probs_calibrated,
            operational_threshold=op_thresh,
            model_name=name,
            split_name="test",
        )
        metrics["threshold_selection_metadata"] = thresh_info
        all_metrics.append(metrics)

        # Save predictions artifact
        y_pred_op = (test_probs_calibrated >= op_thresh).astype(int)
        save_test_predictions(
            record_ids=record_ids,
            y_true=y_test,
            y_prob_raw=test_probs_raw,
            y_prob_calibrated=test_probs_calibrated,
            y_pred_op=y_pred_op,
            output_path=predictions_dir / f"{sname}_test_predictions.csv",
        )

        print(f"  [{name}] PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | Brier: {metrics['brier_score_calibrated']:.4f} | Op Threshold: {op_thresh:.3f} | Recall@OpThresh: {metrics['recall_at_op_threshold']:.4f}")

    # Save model comparison table
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison_df = create_model_comparison_table(all_metrics)
    comparison_df.to_csv(metrics_dir / "model_comparison.csv", index=False)
    with open(metrics_dir / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nModel comparison table saved to {metrics_dir / 'model_comparison.csv'}")

    # 8. Offline Rescheduling Simulator Scenario
    print("\n[Stage 8/8] Running Offline Rescheduling Sensitivity Analysis & Generating Figures...")
    best_calibrator = ModelCalibrator.load(models_dir / "xgboost_calibrator.pkl")
    rescheduler = ReschedulingSimulator(
        model_wrapper=xgb_model,
        preprocessor=tabular_prep,
        calibrator=best_calibrator,
        is_pytorch=False,
    )

    # Run simulator on representative test flight
    sample_flight = test_feat.iloc[100]
    reschedule_out = rescheduler.simulate_shifts_for_flight(sample_flight)
    with open(reports_dir / "rescheduling_scenario_sample.json", "w") as f:
        json.dump(reschedule_out, f, indent=2, default=json_serial)

    # Publication Figures
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_curves(test_prob_dict, y_test, figures_dir / "roc_curves.png")
    plot_pr_curves(test_prob_dict, y_test, figures_dir / "pr_curves.png")
    plot_calibration_curves(all_metrics, figures_dir / "calibration_curves.png")
    plot_confusion_matrices(all_metrics, figures_dir / "confusion_matrices.png")

    xgb_importances = xgb_model.get_feature_importances(tabular_prep.feature_names_)
    plot_feature_importances(xgb_importances, top_n=20, output_path=figures_dir / "feature_importances.png")
    plot_delay_eda(df_features, output_dir=figures_dir)
    print(f"All evaluation figures saved to {figures_dir}")

    # Complete Run Summary
    run_summary = {
        "status": "SUCCESS",
        "dataset": audit_stats,
        "split_manifest": split_manifest,
        "metrics_summary": [
            {
                "model": m["model_name"],
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "brier_score_calibrated": m["brier_score_calibrated"],
                "operational_threshold": m["operational_threshold"],
                "recall_at_op_threshold": m["recall_at_op_threshold"],
                "precision_at_op_threshold": m["precision_at_op_threshold"],
                "target_precision_met": m["threshold_selection_metadata"]["target_precision_met"],
            }
            for m in all_metrics
        ],
        "rescheduling_sample": reschedule_out,
        "hyperparams": hparams.__dict__,
        "device_used": str(device),
        "random_seed": random_seed,
    }
    with open(results_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2, default=json_serial)
    print(f"Pipeline complete! Summary saved to {results_dir / 'run_summary.json'}")

    return run_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run flight delay prediction pipeline")
    parser.add_argument("--data-path", type=str, default=str(RAW_DATA_PATH), help="Path to raw flights.csv")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR), help="Path to results directory")
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR), help="Path to models directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target-precision", type=float, default=0.70, help="Operational target precision")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for PyTorch MLP")
    args = parser.parse_args()

    run_pipeline(
        data_path=Path(args.data_path),
        results_dir=Path(args.results_dir),
        models_dir=Path(args.models_dir),
        random_seed=args.seed,
        target_precision=args.target_precision,
        mlp_epochs=args.epochs,
    )
