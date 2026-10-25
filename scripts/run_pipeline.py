"""Reproduce the Phase 1 ML experiment, with disjoint temporal decisions."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss
from src.config import (
    RAW_DATA_PATH,
    RESULTS_DIR,
    MODELS_DIR,
    ALL_MODEL_FEATURES,
    ModelHyperparams,
)
from src.data import (
    load_raw_dataset,
    clean_and_prepare_dataset,
    attach_label_availability,
)
from src.features import compute_schedule_and_calendar_features, ScheduleContext
from src.splitting import chronological_date_split
from src.experiments import (
    prepare_partitions,
    temporal_search,
    development_blocks,
    choose_calibrator,
    ablation_study,
    day_block_intervals,
    forward_feature_selection,
)
from src.train import train_logistic_regression, train_xgboost, train_pytorch_mlp
from src.preprocessing import FlightTabularPreprocessor
from src.calibration import select_operational_threshold
from src.evaluate import (
    evaluate_predictions,
    save_test_predictions,
    create_model_comparison_table,
)
from src.validation import audit_feature_matrix_for_forbidden_columns
from src.plotting import (
    plot_roc_curves,
    plot_pr_curves,
    plot_calibration_curves,
    plot_confusion_matrices,
    plot_feature_importances,
    plot_delay_eda,
)
from src.reschedule import ReschedulingSimulator


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
        + "\n"
    )


def run_pipeline(
    data_path=RAW_DATA_PATH,
    results_dir=RESULTS_DIR,
    models_dir=MODELS_DIR,
    random_seed=42,
    target_precision=0.70,
    mlp_epochs=15,
    include_mlp=True,
    bootstrap_repeats=200,
):
    started = time.time()
    results_dir, models_dir, data_path = (
        Path(results_dir),
        Path(models_dir),
        Path(data_path),
    )
    reports, metrics_dir, figures, predictions = [
        results_dir / x for x in ["reports", "metrics", "figures", "predictions"]
    ]
    for directory in [reports, metrics_dir, figures, predictions, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    print("Loading original dataset and auditing outcome availability...", flush=True)
    raw = load_raw_dataset(data_path)
    clean, audit = clean_and_prepare_dataset(raw)
    roster = raw.copy()
    roster["flight_date"] = pd.to_datetime(roster[["year", "month", "day"]])
    context = ScheduleContext.from_roster(roster)
    features = context.transform(compute_schedule_and_calendar_features(clean))
    features, availability = attach_label_availability(features)
    audit["historical_label_availability"] = availability
    audit["dataset_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    audit["source_url"] = "https://www.kaggle.com/datasets/matinsajadi/flights"
    audit["schedule_roster_rows"] = len(roster)
    save_json(reports / "dataset_profile.json", audit)
    train, validation, test, manifest = chronological_date_split(features)
    print(
        "Selecting features and hyperparameters on four training-only date folds...",
        flush=True,
    )
    hyperparams, search = temporal_search(train, random_seed)
    save_json(reports / "temporal_search.json", search)
    fitted, _, history, purge = prepare_partitions(train, validation, rolling=True)
    # This is a historical event database, not a fitted predictor. Future events
    # remain invisible to all earlier queries through strict availability filters.
    history.fit(features)
    history.freeze(pd.Timestamp(purge["fit_available_at"]))
    fitted = history.transform_expanding(fitted)
    validation_feat = history.transform_inference(validation)
    test_feat = history.transform_inference(test)
    manifest["model_fit"] = purge
    manifest["history_policy"] = (
        "Daily local-midnight snapshots of outcomes available after actual arrival plus 24h. Identical historical replay for training, development, test and inference. Estimator weights stay fixed."
    )
    blocks, block_manifest = development_blocks(validation_feat, test_feat)
    print(
        "Selecting the final feature family on the forward pipeline-selection block...",
        flush=True,
    )
    hyperparams, numeric_sets, feature_sets, forward_records = (
        forward_feature_selection(
            fitted, blocks["calibration_selection"], search, random_seed
        )
    )
    search["training_cv_feature_winners"] = search["selected_feature_sets"]
    search["selected_feature_sets"], search["selected_numeric_features"] = (
        feature_sets,
        numeric_sets,
    )
    search["selected_hyperparameters"] = {k: asdict(v) for k, v in hyperparams.items()}
    search["forward_pipeline_selection"] = forward_records
    save_json(reports / "temporal_search.json", search)
    manifest["development_blocks"] = block_manifest
    save_json(reports / "split_manifest.json", manifest)
    save_json(models_dir / "historical_generator.json", history.to_dict())
    event_columns = [
        "label_available_at",
        "is_delayed",
        "carrier",
        "origin",
        "dest",
        "route",
    ]
    history.source[event_columns].sort_values("label_available_at").to_parquet(
        models_dir / "historical_events.parquet", index=False
    )
    save_json(models_dir / "schedule_context.json", context.to_dict())
    for frame in [fitted, validation_feat, test_feat]:
        audit_feature_matrix_for_forbidden_columns(frame[ALL_MODEL_FEATURES].columns)
    cal_fit = blocks["calibration_fit"]
    print(
        "Fitting final models with training labels available at the model cutoff...",
        flush=True,
    )
    lr, prep, lr_train, _ = train_logistic_regression(
        fitted,
        cal_fit,
        hyperparams["logistic_regression"],
        random_seed,
        models_dir,
        numeric_features=search["selected_numeric_features"]["logistic_regression"],
    )
    xgb_prep = FlightTabularPreprocessor(
        numeric_features=search["selected_numeric_features"]["xgboost"]
    ).fit(fitted[ALL_MODEL_FEATURES])
    xgb_prep.save(models_dir / "xgboost_preprocessor.pkl")
    xgb, xgb_train, _ = train_xgboost(
        fitted, cal_fit, xgb_prep, hyperparams["xgboost"], random_seed, models_dir
    )
    model_infos = [
        ("logistic_regression", "Logistic Regression", lr, prep, lr_train, False),
        ("xgboost", "XGBoost", xgb, xgb_prep, xgb_train, False),
    ]
    if include_mlp:
        # Supplementary model only. Fixed epochs, no calibration-label checkpoint selection.
        hp = ModelHyperparams(mlp_epochs=mlp_epochs)
        mlp, mp, mt, _ = train_pytorch_mlp(
            fitted,
            cal_fit,
            hp,
            random_seed,
            torch.device("cpu"),
            models_dir,
            select_checkpoint=False,
            numeric_features=search["selected_numeric_features"]["xgboost"],
        )
        model_infos.append(("pytorch_mlp", "PyTorch Embedding MLP", mlp, mp, mt, True))
    elif any(models_dir.glob("pytorch_mlp*")):
        raise ValueError(
            "Existing MLP artifacts would become stale. Include MLP or use a fresh models directory."
        )
    all_metrics, probability_sets, prediction_frames, calibration_records = (
        [],
        {},
        {},
        {},
    )
    policy_cutoff = test_feat.decision_time.min().isoformat()
    bundle = {
        "schema_version": 2,
        "policy_available_at": policy_cutoff,
        "history_cutoff": history.state_cutoff,
        "history_policy": history.history_policy,
        "schedule_date_min": context.date_min,
        "schedule_date_max": context.date_max,
        "supported_models": [x[0] for x in model_infos],
        "feature_names": ALL_MODEL_FEATURES,
        "selected_numeric_features": search["selected_numeric_features"],
        "selected_feature_sets": search["selected_feature_sets"],
        "prediction_horizon_hours": 2,
        "historical_reporting_lag_hours": 24,
        "supplementary_models": ["pytorch_mlp"] if include_mlp else [],
    }
    save_json(models_dir / "bundle_metadata.json", bundle)

    def probabilities(model, preprocessor, frame, pytorch):
        if pytorch:
            c, n, _ = preprocessor.transform(frame[ALL_MODEL_FEATURES])
            return model.predict_proba(c, n)
        return model.predict_proba(preprocessor.transform(frame[ALL_MODEL_FEATURES]))

    for key, name, model, preprocessor, train_probs, is_pytorch in model_infos:
        pfit = probabilities(model, preprocessor, cal_fit, is_pytorch)
        selection = blocks["calibration_selection"]
        pselect = probabilities(model, preprocessor, selection, is_pytorch)
        calibrator, cal_info = choose_calibrator(
            pfit, cal_fit.is_delayed, pselect, selection.is_delayed
        )
        calibrator.save(models_dir / f"{key}_calibrator.pkl")
        calibration_records[key] = cal_info
        threshold_frame = blocks["threshold_validation"]
        threshold_probs = calibrator.predict_proba(
            probabilities(model, preprocessor, threshold_frame, is_pytorch)
        )
        threshold_info = select_operational_threshold(
            threshold_probs, threshold_frame.is_delayed.to_numpy(), target_precision
        )
        threshold_info.update(
            {
                "calibration_method": calibrator.method,
                "policy_available_at": policy_cutoff,
            }
        )
        save_json(models_dir / f"{key}_threshold.json", threshold_info)
        for block_name, block in blocks.items():
            raw_probs = probabilities(model, preprocessor, block, is_pytorch)
            pd.DataFrame(
                {
                    "record_id": block["id"],
                    "flight_date": block.flight_date,
                    "y_true": block.is_delayed,
                    "prob_raw": raw_probs,
                    "prob_calibrated": calibrator.predict_proba(raw_probs),
                }
            ).to_csv(predictions / f"{key}_{block_name}_predictions.csv", index=False)
        raw_probs = probabilities(model, preprocessor, test_feat, is_pytorch)
        cal_probs = calibrator.predict_proba(raw_probs)
        result = evaluate_predictions(
            test_feat.is_delayed,
            raw_probs,
            cal_probs,
            threshold_info["selected_threshold"],
            name,
        )
        result["threshold_selection_metadata"], result["calibration_selection"] = (
            threshold_info,
            cal_info,
        )
        result["training_average_precision"] = float(
            average_precision_score(fitted.is_delayed, train_probs)
        )
        result["confidence_intervals_95_day_block"] = day_block_intervals(
            test_feat.is_delayed,
            raw_probs,
            test_feat.flight_date,
            random_seed,
            bootstrap_repeats,
            brier_probabilities=cal_probs,
        )
        result["interval_score_scale"] = (
            "raw_ranking_metrics_and_selected_probability_brier"
        )
        result["supplementary"] = is_pytorch
        all_metrics.append(result)
        probability_sets[name] = (raw_probs, cal_probs)
        filename = predictions / f"{key}_test_predictions.csv"
        save_test_predictions(
            test_feat["id"],
            test_feat.is_delayed.to_numpy(),
            raw_probs,
            cal_probs,
            (cal_probs >= threshold_info["selected_threshold"]).astype(int),
            filename,
        )
        pf = pd.read_csv(filename)
        for col in ["flight_date", "carrier", "origin", "dest", "sched_dep_hour"]:
            pf[col] = test_feat[col].to_numpy()
        pf.to_csv(filename, index=False)
        prediction_frames[key] = pf
        print(
            f'{name}: AP={result["pr_auc"]:.4f}, ROC-AUC={result["roc_auc"]:.4f}, Brier={result["brier_score_calibrated"]:.4f}, precision={result["precision_at_op_threshold"]:.3f}, recall={result["recall_at_op_threshold"]:.3f}, calibration={calibrator.method}',
            flush=True,
        )
    prior = float(fitted.is_delayed.mean())
    baseline_probs = np.full(len(test_feat), prior)
    baseline = evaluate_predictions(
        test_feat.is_delayed,
        baseline_probs,
        baseline_probs,
        0.5,
        "Constant training prior",
    )
    baseline["constant_probability"] = prior
    all_metrics.append(baseline)
    save_json(metrics_dir / "evaluation_metrics.json", all_metrics)
    create_model_comparison_table(all_metrics).to_csv(
        metrics_dir / "model_comparison.csv", index=False
    )
    save_json(reports / "calibration_selection.json", calibration_records)
    print(
        "Running fixed-configuration feature ablations on development data...",
        flush=True,
    )
    pd.DataFrame(
        ablation_study(
            fitted, blocks["calibration_selection"], hyperparams, random_seed
        )
    ).to_csv(metrics_dir / "feature_ablation.csv", index=False)
    slices = []
    for key, pf in prediction_frames.items():
        for column in ["origin", "carrier", "sched_dep_hour", "flight_date"]:
            for value, group in pf.groupby(column):
                y = group.y_true.to_numpy()
                slices.append(
                    {
                        "model": key,
                        "group": column,
                        "value": str(value),
                        "n": len(group),
                        "prevalence": float(y.mean()),
                        "average_precision": (
                            float(average_precision_score(y, group.prob_calibrated))
                            if len(np.unique(y)) > 1
                            else None
                        ),
                        "brier_score": float(
                            brier_score_loss(y, group.prob_calibrated)
                        ),
                        "alert_count": int(group.y_pred_operational.sum()),
                    }
                )
    pd.DataFrame(slices).to_csv(metrics_dir / "error_slices.csv", index=False)
    plot_roc_curves(
        probability_sets, test_feat.is_delayed.to_numpy(), figures / "roc_curves.png"
    )
    plot_pr_curves(
        probability_sets, test_feat.is_delayed.to_numpy(), figures / "pr_curves.png"
    )
    plot_calibration_curves(all_metrics[:-1], figures / "calibration_curves.png")
    plot_confusion_matrices(all_metrics, figures / "confusion_matrices.png")
    plot_feature_importances(
        xgb.get_feature_importances(xgb_prep.feature_names_),
        output_path=figures / "feature_importances.png",
    )
    plot_delay_eda(fitted, figures)
    print("Running non-causal rescheduling sensitivity cohort...", flush=True)
    from src.calibration import ModelCalibrator

    simulator = ReschedulingSimulator(
        xgb,
        xgb_prep,
        ModelCalibrator.load(models_dir / "xgboost_calibrator.pkl"),
        schedule_context=context,
    )
    cohort = []
    for _, row in test_feat.sample(
        n=min(100, len(test_feat)), random_state=random_seed
    ).iterrows():
        output = simulator.simulate_shifts_for_flight(row)
        cohort.append(
            {
                "record_id": int(row["id"]),
                "flight_date": str(row.flight_date.date()),
                "baseline_probability": output["baseline_delay_probability"],
                "shift_minutes": output["optimal_shift_minutes"],
                "selected_probability": output["optimal_delay_probability"],
                "risk_difference": output["best_risk_difference"],
            }
        )
    pd.DataFrame(cohort).to_csv(reports / "rescheduling_cohort.csv", index=False)
    sample = simulator.simulate_shifts_for_flight(test_feat.iloc[100])
    save_json(reports / "rescheduling_scenario_sample.json", sample)
    source_files = sorted(
        (Path(__file__).resolve().parents[1] / "src").glob("*.py")
    ) + [Path(__file__)]
    source_hash = hashlib.sha256(
        b"".join(p.read_bytes() for p in source_files)
    ).hexdigest()
    versions = {
        p: importlib.metadata.version(p)
        for p in [
            "numpy",
            "pandas",
            "scikit-learn",
            "xgboost",
            "torch",
            "matplotlib",
            "seaborn",
        ]
    }
    summary = {
        "status": "SUCCESS",
        "protocol_version": 2,
        "dataset": audit,
        "split_manifest": manifest,
        "metrics_summary": [
            {
                "model": m["model_name"],
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "brier_score_calibrated": m["brier_score_calibrated"],
                "precision_at_op_threshold": m["precision_at_op_threshold"],
                "recall_at_op_threshold": m["recall_at_op_threshold"],
            }
            for m in all_metrics
        ],
        "selected_hyperparameters": {k: asdict(v) for k, v in hyperparams.items()},
        "selected_feature_sets": search["selected_feature_sets"],
        "rescheduling_sample": sample,
        "rescheduling_cohort_size": len(cohort),
        "random_seed": random_seed,
        "bootstrap": {
            "method": "whole-date resampling",
            "repeats": bootstrap_repeats,
            "seed": random_seed,
        },
        "environment": versions,
        "python": sys.version,
        "source_sha256": source_hash,
        "base_git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_changes": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "elapsed_seconds": round(time.time() - started, 2),
        "test_reuse_disclosure": "The original test results were inspected during review. This is a corrected retrospective reevaluation, not a newly untouched external test.",
        "phase_scope": "Logistic Regression and XGBoost are the Phase 1 ML models. MLP is supplementary.",
    }
    save_json(results_dir / "run_summary.json", summary)
    legacy_preprocessor = models_dir / "tabular_preprocessor.pkl"
    if legacy_preprocessor.exists():
        legacy_preprocessor.unlink()
    print(f'Completed in {summary["elapsed_seconds"]:.1f}s', flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-precision", type=float, default=0.70)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    args = parser.parse_args()
    run_pipeline(
        args.data_path,
        args.results_dir,
        args.models_dir,
        args.seed,
        args.target_precision,
        args.epochs,
        not args.skip_mlp,
        args.bootstrap_repeats,
    )
