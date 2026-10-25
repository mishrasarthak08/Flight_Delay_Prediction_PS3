"""Small real estimator fits verify training/checkpoint behavior in CI."""

from dataclasses import replace
import numpy as np
import pandas as pd
from src.config import ModelHyperparams, ALL_MODEL_FEATURES
from src.data import generate_synthetic_dataset, clean_and_prepare_dataset
from src.features import compute_schedule_and_calendar_features, ScheduleContext
from src.experiments import prepare_partitions
from src.train import train_logistic_regression, train_xgboost, train_pytorch_mlp
from src.models import FlightEmbeddingMLP


def test_training_functions_fit_and_reload_real_models(tmp_path):
    raw = generate_synthetic_dataset(500, 42)
    clean, _ = clean_and_prepare_dataset(raw)
    roster = raw.copy()
    roster["flight_date"] = pd.to_datetime(roster[["year", "month", "day"]])
    frame = ScheduleContext.from_roster(roster).transform(
        compute_schedule_and_calendar_features(clean)
    )
    frame["label_available_at"] = frame.decision_time + pd.Timedelta(days=2)
    dates = np.sort(frame.flight_date.unique())
    train, valid = (
        frame[frame.flight_date < dates[45]],
        frame[frame.flight_date >= dates[45]],
    )
    fitted, held, _, _ = prepare_partitions(train, valid)
    hp = replace(
        ModelHyperparams(),
        xgb_n_estimators=5,
        xgb_early_stopping_rounds=None,
        mlp_epochs=2,
        mlp_batch_size=len(fitted),
        mlp_hidden_dims=[16, 8],
        mlp_dropout=0,
    )
    lr, prep, lp, lv = train_logistic_regression(fitted, held, hp, output_dir=tmp_path)
    xgb, xp, xv = train_xgboost(fitted, held, prep, hp, output_dir=tmp_path)
    mlp, mp, mt, mv = train_pytorch_mlp(
        fitted, held, hp, output_dir=tmp_path, select_checkpoint=False
    )
    for values in [lp, lv, xp, xv, mt, mv]:
        assert np.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()
    loaded = FlightEmbeddingMLP.load_checkpoint(tmp_path / "pytorch_mlp.pt")
    cats, nums, _ = mp.transform(held[ALL_MODEL_FEATURES])
    np.testing.assert_allclose(loaded.predict_proba(cats, nums), mv, atol=1e-7)
    assert (tmp_path / "logistic_regression_preprocessor.pkl").exists()
