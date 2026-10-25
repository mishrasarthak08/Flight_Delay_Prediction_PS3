"""
Preprocessing pipelines for classical tabular models and deep learning PyTorch architectures.
Ensures zero data leakage: all scalers, encoders, and vocabularies are fit exclusively
on the training partition.
"""

from typing import Dict, Any, List, Tuple, Optional
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_SCHEDULE_FEATURES,
    HISTORICAL_FEATURES,
    ALL_MODEL_FEATURES,
    MODELS_DIR,
)


class FlightTabularPreprocessor:
    """
    Scikit-learn ColumnTransformer for Logistic Regression and XGBoost.
    Handles numeric imputation/scaling and categorical one-hot encoding.
    """

    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        sparse_output: bool = False,
    ):
        self.numeric_features = numeric_features or (NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES)
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES
        self.sparse_output = sparse_output
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame) -> "FlightTabularPreprocessor":
        """Fits imputer, scaler, and one-hot encoder strictly on training data."""
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=self.sparse_output)),
        ])

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", num_transformer, self.numeric_features),
                ("cat", cat_transformer, self.categorical_features),
            ],
            remainder="drop",
        )

        self.pipeline.fit(X)

        # Extract generated feature names
        num_cols = self.numeric_features
        cat_encoder = self.pipeline.named_transformers_["cat"].named_steps["ohe"]
        cat_cols = list(cat_encoder.get_feature_names_out(self.categorical_features))
        self.feature_names_ = list(num_cols) + cat_cols
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms feature matrix using fitted transformers."""
        if self.pipeline is None:
            raise RuntimeError("Preprocessor must be fit before transform.")
        return self.pipeline.transform(X).astype(np.float32)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, filepath: Path) -> None:
        """Saves fitted preprocessor to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "FlightTabularPreprocessor":
        """Loads fitted preprocessor from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class PyTorchPreprocessor:
    """
    Preprocessor for the PyTorch Embedding MLP.
    Constructs category vocabularies with <UNK> token and fits numeric standard scaling.
    """

    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        numeric_features: Optional[List[str]] = None,
    ):
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES
        self.numeric_features = numeric_features or (NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES)
        self.vocabularies: Dict[str, Dict[str, int]] = {}
        self.vocab_sizes: Dict[str, int] = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> "PyTorchPreprocessor":
        """Fits category index mappings and numeric standardizer exclusively on training data."""
        # 1. Fit categorical vocabularies
        for col in self.categorical_features:
            unique_vals = sorted(X[col].dropna().astype(str).unique().tolist())
            # Reserve 0 for <UNK>
            vocab = {"<UNK>": 0}
            for idx, val in enumerate(unique_vals, start=1):
                vocab[val] = idx
            self.vocabularies[col] = vocab
            self.vocab_sizes[col] = len(vocab)

        # 2. Fit numeric imputer and scaler
        num_data = X[self.numeric_features].values
        num_imputed = self.imputer.fit_transform(num_data)
        self.scaler.fit(num_imputed)

        self.is_fitted = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Transforms dataframe into tensors:
        - cat_tensor: (N, num_cats) of type int64
        - num_tensor: (N, num_numerics) of type float32
        - y_tensor: (N, 1) of type float32 (or None)
        """
        if not self.is_fitted:
            raise RuntimeError("PyTorchPreprocessor must be fit before calling transform.")

        # Categorical index mapping
        cat_cols = []
        for col in self.categorical_features:
            vocab = self.vocabularies[col]
            series_str = X[col].astype(str)
            mapped_indices = series_str.map(lambda val: vocab.get(val, 0)).values
            cat_cols.append(mapped_indices)
        cat_array = np.column_stack(cat_cols).astype(np.int64)

        # Numeric scaling
        num_data = X[self.numeric_features].values
        num_imputed = self.imputer.transform(num_data)
        num_scaled = self.scaler.transform(num_imputed).astype(np.float32)

        cat_tensor = torch.tensor(cat_array, dtype=torch.long)
        num_tensor = torch.tensor(num_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None

        return cat_tensor, num_tensor, y_tensor

    def save(self, filepath: Path) -> None:
        """Saves fitted PyTorch preprocessor."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "PyTorchPreprocessor":
        with open(filepath, "rb") as f:
            return pickle.load(f)


class FlightDataset(Dataset):
    """PyTorch Dataset yielding (cat_tensor, num_tensor, y_tensor)."""

    def __init__(
        self,
        cat_tensor: torch.Tensor,
        num_tensor: torch.Tensor,
        y_tensor: Optional[torch.Tensor] = None
    ):
        self.cat_tensor = cat_tensor
        self.num_tensor = num_tensor
        self.y_tensor = y_tensor

    def __len__(self) -> int:
        return len(self.cat_tensor)

    def __getitem__(self, idx: int):
        if self.y_tensor is not None:
            return self.cat_tensor[idx], self.num_tensor[idx], self.y_tensor[idx]
        return self.cat_tensor[idx], self.num_tensor[idx]
