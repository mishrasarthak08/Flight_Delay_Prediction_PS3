"""
Training routines for Logistic Regression, XGBoost, and PyTorch Embedding MLP.
Enforces validation monitoring, early stopping, and checkpointing.
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from src.config import (
    ModelHyperparams,
    PipelineConfig,
    MODELS_DIR,
    TARGET_COL,
    CATEGORICAL_FEATURES,
    NUMERIC_SCHEDULE_FEATURES,
    HISTORICAL_FEATURES,
)
from src.preprocessing import (
    FlightTabularPreprocessor,
    PyTorchPreprocessor,
    FlightDataset,
)
from src.models import (
    LogisticRegressionBaseline,
    XGBoostModel,
    FlightEmbeddingMLP,
)


def train_logistic_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hyperparams: ModelHyperparams = ModelHyperparams(),
    random_seed: int = 42,
    output_dir: Path = MODELS_DIR,
) -> Tuple[LogisticRegressionBaseline, FlightTabularPreprocessor, np.ndarray, np.ndarray]:
    """
    Trains Logistic Regression baseline using scikit-learn preprocessing pipeline.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = FlightTabularPreprocessor(sparse_output=False)

    feature_cols = CATEGORICAL_FEATURES + NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES
    X_train = np.ascontiguousarray(preprocessor.fit_transform(train_df[feature_cols]), dtype=np.float32)
    y_train = np.ascontiguousarray(train_df[TARGET_COL].values, dtype=np.int32)

    X_val = np.ascontiguousarray(preprocessor.transform(val_df[feature_cols]), dtype=np.float32)
    y_val = np.ascontiguousarray(val_df[TARGET_COL].values, dtype=np.int32)

    model = LogisticRegressionBaseline(hyperparams=hyperparams, random_seed=random_seed)
    model.fit(X_train, y_train)

    # Save artifacts
    preprocessor.save(output_dir / "tabular_preprocessor.pkl")
    import pickle
    with open(output_dir / "logistic_regression.pkl", "wb") as f:
        pickle.dump(model, f)

    train_probs = model.predict_proba(X_train)
    val_probs = model.predict_proba(X_val)
    return model, preprocessor, train_probs, val_probs


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessor: FlightTabularPreprocessor,
    hyperparams: ModelHyperparams = ModelHyperparams(),
    random_seed: int = 42,
    output_dir: Path = MODELS_DIR,
) -> Tuple[XGBoostModel, np.ndarray, np.ndarray]:
    """
    Trains XGBoost model with early stopping on validation split.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = CATEGORICAL_FEATURES + NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES

    X_train = np.ascontiguousarray(preprocessor.transform(train_df[feature_cols]), dtype=np.float32)
    y_train = np.ascontiguousarray(train_df[TARGET_COL].values, dtype=np.float32)

    X_val = np.ascontiguousarray(preprocessor.transform(val_df[feature_cols]), dtype=np.float32)
    y_val = np.ascontiguousarray(val_df[TARGET_COL].values, dtype=np.float32)

    print(f"     Fitting XGBoost hist on {X_train.shape}...")
    model = XGBoostModel(hyperparams=hyperparams, random_seed=random_seed)
    model.fit(X_train, y_train, X_val, y_val)
    print(f"     XGBoost fitted successfully! Best iteration: {model.model.best_iteration}")

    # Save artifact
    import pickle
    with open(output_dir / "xgboost.pkl", "wb") as f:
        pickle.dump(model, f)

    train_probs = model.predict_proba(X_train)
    val_probs = model.predict_proba(X_val)
    return model, train_probs, val_probs


def train_pytorch_mlp(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hyperparams: ModelHyperparams = ModelHyperparams(),
    random_seed: int = 42,
    device: Optional[torch.device] = None,
    output_dir: Path = MODELS_DIR,
) -> Tuple[FlightEmbeddingMLP, PyTorchPreprocessor, np.ndarray, np.ndarray]:
    """
    Trains PyTorch Embedding MLP with early stopping on validation loss.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if device is None:
        device = torch.device("cpu")

    # Fit PyTorch Preprocessor strictly on training data
    preprocessor = PyTorchPreprocessor()
    feature_cols = CATEGORICAL_FEATURES + NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES
    preprocessor.fit(train_df[feature_cols])

    # Convert to Tensors
    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values

    c_train, n_train, yt_train = preprocessor.transform(train_df[feature_cols], y_train)
    c_val, n_val, yt_val = preprocessor.transform(val_df[feature_cols], y_val)

    train_dataset = FlightDataset(c_train, n_train, yt_train)
    val_dataset = FlightDataset(c_val, n_val, yt_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams.mlp_batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams.mlp_batch_size,
        shuffle=False,
    )

    # Model definition
    num_carriers = preprocessor.vocab_sizes["carrier"]
    num_origins = preprocessor.vocab_sizes["origin"]
    num_dests = preprocessor.vocab_sizes["dest"]
    num_numeric = len(preprocessor.numeric_features)

    model = FlightEmbeddingMLP(
        num_carriers=num_carriers,
        num_origins=num_origins,
        num_dests=num_dests,
        num_numeric_features=num_numeric,
        carrier_emb_dim=hyperparams.mlp_carrier_emb_dim,
        origin_emb_dim=hyperparams.mlp_origin_emb_dim,
        dest_emb_dim=hyperparams.mlp_dest_emb_dim,
        hidden_dims=hyperparams.mlp_hidden_dims,
        dropout=hyperparams.mlp_dropout,
    )
    model.to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams.mlp_learning_rate,
        weight_decay=hyperparams.mlp_weight_decay,
    )

    best_val_loss = float("inf")
    best_state_dict = None
    patience = hyperparams.mlp_early_stopping_patience
    patience_counter = 0

    for epoch in range(1, hyperparams.mlp_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_cat, batch_num, batch_y in train_loader:
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_cat, batch_num)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        train_loss /= len(train_dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_preds_list = []
        with torch.no_grad():
            for batch_cat, batch_num, batch_y in val_loader:
                batch_cat = batch_cat.to(device)
                batch_num = batch_num.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_cat, batch_num)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_preds_list.append(probs)

        val_loss /= len(val_dataset)
        val_probs_arr = np.concatenate(val_preds_list)
        val_prauc = average_precision_score(y_val, val_probs_arr)
        print(f"     [Epoch {epoch}/{hyperparams.mlp_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PR-AUC: {val_prauc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Save artifacts
    preprocessor.save(output_dir / "pytorch_preprocessor.pkl")
    model.save_checkpoint(
        output_dir / "pytorch_mlp.pt",
        config_dict={
            "best_val_loss": float(best_val_loss),
            "epochs_run": epoch,
            "random_seed": random_seed,
        },
    )

    # Compute final predictions on CPU
    model.to(torch.device("cpu"))
    train_probs = model.predict_proba(c_train, n_train)
    val_probs = model.predict_proba(c_val, n_val)
    return model, preprocessor, train_probs, val_probs
