"""Date-based model selection, disjoint calibration, and uncertainty analysis."""

from dataclasses import replace, asdict
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from src.config import (
    ModelHyperparams,
    ALL_MODEL_FEATURES,
    NUMERIC_SCHEDULE_FEATURES,
    HISTORICAL_FEATURES,
)
from src.features import HistoricalFeatureGenerator
from src.preprocessing import FlightTabularPreprocessor
from src.models import LogisticRegressionBaseline, XGBoostModel
from src.calibration import ModelCalibrator


def prepare_partitions(train, later, rolling=False):
    """Purge labels unavailable at fitting time; use frozen or daily replay history."""
    cutoff = pd.to_datetime(later["decision_time"], utc=True).min()
    usable = train[train["label_available_at"] < cutoff].copy()
    if usable.empty or usable["is_delayed"].nunique() < 2:
        raise ValueError(
            "No usable two-class fitting population after availability purge."
        )
    generator = HistoricalFeatureGenerator()
    if rolling:
        generator.enable_rolling().fit(pd.concat([train, later]))
        fit_features = generator.transform_expanding(usable)
    else:
        fit_features = generator.fit_transform_chronological(usable)
    generator.freeze(cutoff)
    return (
        fit_features,
        generator.transform_inference(later),
        generator,
        {
            "fit_available_at": cutoff.isoformat(),
            "candidate_training_rows": len(train),
            "fitted_rows": len(usable),
            "purged_unavailable_or_invalid_labels": len(train) - len(usable),
        },
    )


def temporal_search(train, seed=42):
    """Joint feature/parameter selection on four expanding, 30-date holdouts."""
    dates = np.sort(train["flight_date"].unique())
    if len(dates) < 100:
        raise ValueError("Temporal search requires at least 100 training dates.")
    numeric_sets = {
        "schedule_demand": list(NUMERIC_SCHEDULE_FEATURES),
        "cyclic_schedule_demand": [
            c
            for c in NUMERIC_SCHEDULE_FEATURES
            if c not in ["month", "day", "day_of_week"]
        ],
        "history_rates_demand": NUMERIC_SCHEDULE_FEATURES
        + [c for c in HISTORICAL_FEATURES if "rate" in c],
        "all_features": NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES,
    }
    candidates = []
    for c in [0.03, 0.3]:
        for weight in [None, "balanced"]:
            candidates.append(
                ("logistic_regression", {"lr_c": c, "lr_class_weight": weight})
            )
    for depth in [3, 5]:
        for weight in [20.0, 100.0]:
            candidates.append(
                (
                    "xgboost",
                    {
                        "xgb_max_depth": depth,
                        "xgb_min_child_weight": weight,
                        "xgb_n_estimators": 300,
                        "xgb_learning_rate": 0.06,
                        "xgb_reg_lambda": 5.0,
                    },
                )
            )
    records = []
    for fold, fraction in enumerate([0.40, 0.60, 0.80, 1.0], 1):
        end = int(len(dates) * fraction)
        start = end - 30
        fitting = train[train.flight_date < dates[start]]
        held = train[train.flight_date.isin(dates[start:end])]
        fitting, held, _, availability = prepare_partitions(fitting, held, rolling=True)
        yt, yv = fitting.is_delayed.to_numpy(), held.is_delayed.to_numpy()
        for variant, numeric in numeric_sets.items():
            prep = FlightTabularPreprocessor(numeric_features=numeric)
            xt = prep.fit_transform(fitting[ALL_MODEL_FEATURES])
            xv = prep.transform(held[ALL_MODEL_FEATURES])
            for family, parameters in candidates:
                hp = replace(ModelHyperparams(), **parameters)
                model = (
                    LogisticRegressionBaseline(hp, seed)
                    if family == "logistic_regression"
                    else XGBoostModel(hp, seed)
                )
                if family == "xgboost":
                    model.fit(xt, yt, xv, yv)
                    rounds = model.model.best_iteration + 1
                else:
                    model.fit(xt, yt)
                    rounds = None
                prediction = model.predict_proba(xv)
                records.append(
                    {
                        "model": family,
                        "feature_set": variant,
                        "parameters": parameters,
                        "fold": fold,
                        "train_end": str(fitting.flight_date.max().date()),
                        "validation_start": str(held.flight_date.min().date()),
                        "validation_end": str(held.flight_date.max().date()),
                        **availability,
                        "average_precision": float(
                            average_precision_score(yv, prediction)
                        ),
                        "roc_auc": float(roc_auc_score(yv, prediction)),
                        "brier_score": float(brier_score_loss(yv, prediction)),
                        "best_rounds": rounds,
                    }
                )
                print(
                    f"  fold={fold} {family} {variant} {parameters} AP={records[-1]['average_precision']:.4f}",
                    flush=True,
                )
    selected, summary, selected_features, selected_sets = {}, [], {}, {}
    for family in ["logistic_regression", "xgboost"]:
        for variant in numeric_sets:
            for _, parameters in [c for c in candidates if c[0] == family]:
                rows = [
                    r
                    for r in records
                    if r["model"] == family
                    and r["parameters"] == parameters
                    and r["feature_set"] == variant
                ]
                summary.append(
                    {
                        "model": family,
                        "feature_set": variant,
                        "parameters": parameters,
                        "mean_average_precision": float(
                            np.mean([r["average_precision"] for r in rows])
                        ),
                        "std_average_precision": float(
                            np.std([r["average_precision"] for r in rows])
                        ),
                        "mean_brier_score": float(
                            np.mean([r["brier_score"] for r in rows])
                        ),
                        "median_best_rounds": (
                            int(np.median([r["best_rounds"] for r in rows]))
                            if family == "xgboost"
                            else None
                        ),
                    }
                )
        best = max(
            [s for s in summary if s["model"] == family],
            key=lambda s: (s["mean_average_precision"], -s["mean_brier_score"]),
        )
        chosen = dict(best["parameters"])
        if family == "xgboost":
            chosen["xgb_n_estimators"] = best["median_best_rounds"]
            chosen["xgb_early_stopping_rounds"] = None
        selected[family] = replace(ModelHyperparams(), **chosen)
        selected_features[family] = numeric_sets[best["feature_set"]]
        selected_sets[family] = best["feature_set"]
    return selected, {
        "selection_metric": "mean_average_precision_across_four_training_date_folds",
        "fold_records": records,
        "candidate_summary": summary,
        "selected_hyperparameters": {k: asdict(v) for k, v in selected.items()},
        "selected_numeric_features": selected_features,
        "selected_feature_sets": selected_sets,
    }


def forward_feature_selection(train, selection, search, seed=42):
    """Select among CV-tuned feature families on the forward selection block.

    This block selects complete prediction pipelines and calibration; it never
    fits an estimator's weights. Threshold validation and test remain separate.
    """
    schedule = list(NUMERIC_SCHEDULE_FEATURES)
    variants = {
        "schedule_demand": schedule,
        "cyclic_schedule_demand": [
            c for c in schedule if c not in ["month", "day", "day_of_week"]
        ],
        "history_rates_demand": schedule
        + [c for c in HISTORICAL_FEATURES if "rate" in c],
        "all_features": schedule + HISTORICAL_FEATURES,
    }
    records, chosen, numeric_sets, chosen_names = [], {}, {}, {}
    for family in ["logistic_regression", "xgboost"]:
        for name, numeric in variants.items():
            candidates = [
                r
                for r in search["candidate_summary"]
                if r["model"] == family and r["feature_set"] == name
            ]
            best = max(
                candidates,
                key=lambda r: (r["mean_average_precision"], -r["mean_brier_score"]),
            )
            hp = replace(ModelHyperparams(), **best["parameters"])
            if family == "xgboost":
                hp = replace(
                    hp,
                    xgb_n_estimators=best["median_best_rounds"],
                    xgb_early_stopping_rounds=None,
                )
            prep = FlightTabularPreprocessor(numeric_features=numeric)
            xt, xv = prep.fit_transform(train), prep.transform(selection)
            model = (
                LogisticRegressionBaseline(hp, seed)
                if family == "logistic_regression"
                else XGBoostModel(hp, seed)
            )
            model.fit(xt, train.is_delayed.to_numpy())
            p = model.predict_proba(xv)
            records.append(
                {
                    "model": family,
                    "feature_set": name,
                    "numeric_features": numeric,
                    "hyperparameters": asdict(hp),
                    "selection_dataset": "calibration_selection",
                    "average_precision": float(
                        average_precision_score(selection.is_delayed, p)
                    ),
                    "brier_score": float(brier_score_loss(selection.is_delayed, p)),
                }
            )
        winner = max(
            [r for r in records if r["model"] == family],
            key=lambda r: (r["average_precision"], -r["brier_score"]),
        )
        chosen[family] = ModelHyperparams(**winner["hyperparameters"])
        numeric_sets[family], chosen_names[family] = (
            winner["numeric_features"],
            winner["feature_set"],
        )
    return chosen, numeric_sets, chosen_names, records


def development_blocks(validation, test):
    dates = np.sort(validation.flight_date.unique())
    partitions = np.array_split(dates, 3)
    blocks, manifest = {}, {}
    names = ["calibration_fit", "calibration_selection", "threshold_validation"]
    for i, (name, group_dates) in enumerate(zip(names, partitions)):
        frame = validation[validation.flight_date.isin(group_dates)].copy()
        next_dates = partitions[i + 1] if i < 2 else np.sort(test.flight_date.unique())
        next_frame = (
            validation[validation.flight_date.isin(next_dates)] if i < 2 else test
        )
        available_by = next_frame.decision_time.min()
        valid = frame[frame.label_available_at < available_by].copy()
        blocks[name] = valid
        manifest[name] = {
            "start_date": str(frame.flight_date.min().date()),
            "end_date": str(frame.flight_date.max().date()),
            "candidate_rows": len(frame),
            "usable_rows": len(valid),
            "purged_rows": len(frame) - len(valid),
            "labels_available_before": available_by.isoformat(),
            "prevalence": float(valid.is_delayed.mean()),
        }
    return blocks, manifest


def choose_calibrator(fit_probs, fit_y, select_probs, select_y):
    scores, fitted = [], {}
    for method in ["none", "sigmoid", "isotonic"]:
        cal = ModelCalibrator(method).fit(fit_probs, fit_y)
        prediction = cal.predict_proba(select_probs)
        fitted[method] = cal
        scores.append(
            {
                "method": method,
                "selection_brier_score": float(brier_score_loss(select_y, prediction)),
            }
        )
    best = min(scores, key=lambda r: r["selection_brier_score"])["method"]
    return fitted[best], {
        "selected_method": best,
        "selection_dataset": "calibration_selection",
        "candidates": scores,
    }


def ablation_study(train, development, hyperparams, seed=42):
    schedule = [x for x in NUMERIC_SCHEDULE_FEATURES if not x.startswith("origin_")]
    variants = {
        "schedule_calendar": schedule,
        "plus_history": schedule + HISTORICAL_FEATURES,
        "plus_scheduled_demand": NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES,
        "frozen_history_reference": NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES,
    }
    records = []
    for variant, numeric in variants.items():
        prep = FlightTabularPreprocessor(numeric_features=numeric)
        if variant == "frozen_history_reference":
            frozen = HistoricalFeatureGenerator().fit(train)
            frozen.freeze(development.decision_time.min())
            evaluated = frozen.transform_inference(development)
        else:
            evaluated = development
        xt, xv = prep.fit_transform(train), prep.transform(evaluated)
        for family in ["logistic_regression", "xgboost"]:
            hp = hyperparams[family]
            model = (
                LogisticRegressionBaseline(hp, seed)
                if family == "logistic_regression"
                else XGBoostModel(hp, seed)
            )
            model.fit(xt, train.is_delayed.to_numpy())
            p = model.predict_proba(xv)
            records.append(
                {
                    "model": family,
                    "features": variant,
                    "numeric_feature_count": len(numeric),
                    "evaluation_partition": "calibration_selection",
                    "average_precision": float(
                        average_precision_score(development.is_delayed, p)
                    ),
                    "roc_auc": float(roc_auc_score(development.is_delayed, p)),
                    "brier_score": float(brier_score_loss(development.is_delayed, p)),
                }
            )
    return records


def day_block_intervals(
    y, probabilities, dates, seed=42, repeats=200, brier_probabilities=None
):
    """Resample whole flight dates, retaining within-date dependence."""
    rng, y, probabilities = (
        np.random.default_rng(seed),
        np.asarray(y),
        np.asarray(probabilities),
    )
    groups = list(
        pd.Series(np.arange(len(y)))
        .groupby(pd.to_datetime(dates).to_numpy())
        .apply(list)
    )
    brier_probabilities = (
        probabilities
        if brier_probabilities is None
        else np.asarray(brier_probabilities)
    )
    estimates = []
    for _ in range(repeats):
        idx = np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        )
        estimates.append(
            [
                average_precision_score(y[idx], probabilities[idx]),
                roc_auc_score(y[idx], probabilities[idx]),
                brier_score_loss(y[idx], brier_probabilities[idx]),
            ]
        )
    bounds = np.quantile(estimates, [0.025, 0.975], axis=0)
    return {
        name: {"lower": float(bounds[0, i]), "upper": float(bounds[1, i])}
        for i, name in enumerate(["average_precision", "roc_auc", "brier_score"])
    }
