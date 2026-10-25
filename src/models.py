"""
Model architectures for Flight Delay Prediction.
Includes:
- Model A: Scikit-learn Logistic Regression baseline
- Model B: XGBoost Gradient Boosted Decision Trees
- Model C: PyTorch Embedding MLP with learned entity embeddings for carrier, origin, and destination
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from src.config import ModelHyperparams


class LogisticRegressionBaseline:
    """Logistic Regression baseline classifier wrapper."""

    def __init__(
        self, hyperparams: ModelHyperparams = ModelHyperparams(), random_seed: int = 42
    ):
        self.model = LogisticRegression(
            C=hyperparams.lr_c,
            max_iter=hyperparams.lr_max_iter,
            solver=hyperparams.lr_solver,
            class_weight=hyperparams.lr_class_weight,
            random_state=random_seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class XGBoostModel:
    """XGBoost Gradient Boosted Classifier wrapper."""

    def __init__(
        self, hyperparams: ModelHyperparams = ModelHyperparams(), random_seed: int = 42
    ):
        self.hyperparams = hyperparams
        self.random_seed = random_seed
        self.model = xgb.XGBClassifier(
            n_estimators=hyperparams.xgb_n_estimators,
            max_depth=hyperparams.xgb_max_depth,
            learning_rate=hyperparams.xgb_learning_rate,
            subsample=hyperparams.xgb_subsample,
            colsample_bytree=hyperparams.xgb_colsample_bytree,
            eval_metric=hyperparams.xgb_eval_metric,
            tree_method=hyperparams.xgb_tree_method,
            random_state=random_seed,
            n_jobs=hyperparams.xgb_n_jobs,
            early_stopping_rounds=hyperparams.xgb_early_stopping_rounds,
            reg_lambda=hyperparams.xgb_reg_lambda,
            min_child_weight=hyperparams.xgb_min_child_weight,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostModel":
        X_train_c = np.ascontiguousarray(X_train, dtype=np.float32)
        y_train_c = np.ascontiguousarray(y_train, dtype=np.float32)

        if X_val is not None and y_val is not None:
            X_val_c = np.ascontiguousarray(X_val, dtype=np.float32)
            y_val_c = np.ascontiguousarray(y_val, dtype=np.float32)
            eval_set = [(X_val_c, y_val_c)]
        else:
            eval_set = None

        self.model.fit(
            X_train_c,
            y_train_c,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.ascontiguousarray(X, dtype=np.float32))[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        importances = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(feature_names, importances)}


class FlightEmbeddingMLP(nn.Module):
    """
    Compact PyTorch Neural Network using learned categorical embeddings
    for airline carrier, origin airport, and destination airport,
    concatenated with standardized numeric temporal and operational features.
    """

    def __init__(
        self,
        num_carriers: int,
        num_origins: int,
        num_dests: int,
        num_numeric_features: int,
        carrier_emb_dim: int = 16,
        origin_emb_dim: int = 16,
        dest_emb_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_carriers = num_carriers
        self.num_origins = num_origins
        self.num_dests = num_dests
        self.num_numeric_features = num_numeric_features
        self.carrier_emb_dim = carrier_emb_dim
        self.origin_emb_dim = origin_emb_dim
        self.dest_emb_dim = dest_emb_dim
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.dropout_rate = dropout

        # Categorical Embeddings
        self.carrier_embedding = nn.Embedding(
            num_carriers, carrier_emb_dim, padding_idx=0
        )
        self.origin_embedding = nn.Embedding(num_origins, origin_emb_dim, padding_idx=0)
        self.dest_embedding = nn.Embedding(num_dests, dest_emb_dim, padding_idx=0)

        # Total input dimension into MLP
        total_in_dim = (
            carrier_emb_dim + origin_emb_dim + dest_emb_dim + num_numeric_features
        )

        # Dense MLP Backbone
        layers: List[nn.Module] = []
        curr_dim = total_in_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        # Binary classification output logit
        layers.append(nn.Linear(curr_dim, 1))
        self.backbone = nn.Sequential(*layers)

    def forward(
        self, cat_tensor: torch.Tensor, num_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        cat_tensor: (batch_size, 3) where columns are [carrier_idx, origin_idx, dest_idx]
        num_tensor: (batch_size, num_numeric_features)
        Returns: logits (batch_size, 1)
        """
        c_emb = self.carrier_embedding(cat_tensor[:, 0])
        o_emb = self.origin_embedding(cat_tensor[:, 1])
        d_emb = self.dest_embedding(cat_tensor[:, 2])

        # Concatenate embeddings and numeric features
        x = torch.cat([c_emb, o_emb, d_emb, num_tensor], dim=1)
        logits = self.backbone(x)
        return logits

    def predict_proba(
        self,
        cat_tensor: torch.Tensor,
        num_tensor: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> np.ndarray:
        """Computes sigmoid probabilities in evaluation mode."""
        self.eval()
        with torch.no_grad():
            cat_dev = cat_tensor.to(device)
            num_dev = num_tensor.to(device)
            logits = self(cat_dev, num_dev)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def save_checkpoint(
        self, filepath: Path, config_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """Saves model weights and architecture configuration."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "architecture": {
                "num_carriers": self.num_carriers,
                "num_origins": self.num_origins,
                "num_dests": self.num_dests,
                "num_numeric_features": self.num_numeric_features,
                "carrier_emb_dim": self.carrier_emb_dim,
                "origin_emb_dim": self.origin_emb_dim,
                "dest_emb_dim": self.dest_emb_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout_rate,
            },
            "config": config_dict or {},
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load_checkpoint(
        cls, filepath: Path, device: torch.device = torch.device("cpu")
    ) -> "FlightEmbeddingMLP":
        """Loads model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        arch = checkpoint["architecture"]
        model = cls(**arch)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
