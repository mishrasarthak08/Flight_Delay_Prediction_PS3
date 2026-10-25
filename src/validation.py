"""
Validation module enforcing data contracts, absence of target leakage,
and verification of allowed vs forbidden features.
"""

from typing import List, Set, Dict, Any, Union
import pandas as pd

from src.config import (
    FORBIDDEN_COLUMNS,
    TARGET_COL,
    RAW_TARGET_COL,
    ALL_MODEL_FEATURES,
)


class DataLeakageError(ValueError):
    """Raised when forbidden columns or future target outcomes leak into the feature set."""

    pass


class SchemaValidationError(ValueError):
    """Raised when required features are missing or corrupt in the feature matrix."""

    pass


def audit_feature_matrix_for_forbidden_columns(
    feature_columns: Union[List[str], pd.Index, Set[str]], strict: bool = True
) -> Dict[str, Any]:
    """
    Scans a list of feature columns against the explicit forbidden fields list.
    Fails fast with DataLeakageError if any forbidden column is present.
    """
    feature_set = {col.lower() for col in feature_columns}
    forbidden_set = {col.lower() for col in FORBIDDEN_COLUMNS}
    target_names = {TARGET_COL.lower(), RAW_TARGET_COL.lower()}

    # Check for forbidden columns
    leaked_forbidden = list(feature_set.intersection(forbidden_set))
    leaked_targets = list(feature_set.intersection(target_names))

    violations = leaked_forbidden + leaked_targets

    audit_result = {
        "is_clean": len(violations) == 0,
        "total_features": len(feature_columns),
        "violations_found": violations,
        "checked_against_forbidden_count": len(FORBIDDEN_COLUMNS),
    }

    if strict and not audit_result["is_clean"]:
        raise DataLeakageError(
            f"DATA LEAKAGE DETECTED! Feature matrix contains prohibited columns: {violations}. "
            f"These columns reveal actual departure/arrival outcomes or post-flight information."
        )

    return audit_result


def validate_inference_features(
    df: pd.DataFrame, expected_features: List[str] = ALL_MODEL_FEATURES
) -> None:
    """
    Validates that an inference dataframe contains all necessary model features
    and no forbidden post-flight outcome columns.
    """
    audit_feature_matrix_for_forbidden_columns(df.columns, strict=True)

    missing_features = [feat for feat in expected_features if feat not in df.columns]
    if missing_features:
        raise SchemaValidationError(
            f"Inference dataframe is missing required features: {missing_features}"
        )


def verify_no_future_leakage_in_splits(
    train_dates: pd.Series, val_dates: pd.Series, test_dates: pd.Series
) -> None:
    """
    Verifies that train, validation, and test date sets are strictly disjoint and monotonically ordered.
    """
    train_set = set(pd.to_datetime(train_dates).dt.date.unique())
    val_set = set(pd.to_datetime(val_dates).dt.date.unique())
    test_set = set(pd.to_datetime(test_dates).dt.date.unique())

    train_val_overlap = train_set.intersection(val_set)
    val_test_overlap = val_set.intersection(test_set)
    train_test_overlap = train_set.intersection(test_set)

    if train_val_overlap or val_test_overlap or train_test_overlap:
        raise DataLeakageError(
            f"Temporal split date overlap detected! "
            f"Train-Val overlap: {train_val_overlap}, "
            f"Val-Test overlap: {val_test_overlap}, "
            f"Train-Test overlap: {train_test_overlap}"
        )

    max_train = max(train_set)
    min_val = min(val_set)
    max_val = max(val_set)
    min_test = min(test_set)

    if max_train >= min_val:
        raise DataLeakageError(
            f"Chronological ordering violation: Max train date ({max_train}) >= Min val date ({min_val})"
        )
    if max_val >= min_test:
        raise DataLeakageError(
            f"Chronological ordering violation: Max val date ({max_val}) >= Min test date ({min_test})"
        )
