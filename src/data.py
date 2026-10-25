"""
Data loading, schema discovery, data quality cleaning, and dataset profiling.
Enforces the operational prediction contract and separation between targets,
features, and excluded post-flight outcomes.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    TARGET_COL,
    RAW_TARGET_COL,
    DELAY_THRESHOLD_MINUTES,
    SCHEMA_ALIASES,
)


def discover_and_map_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Discovers dataset schema using alias maps and renames columns to standard canonical names.
    Raises ValueError if any critical field cannot be discovered.
    """
    col_map: Dict[str, str] = {}
    actual_cols = {col.lower(): col for col in df.columns}

    # Map essential fields
    for canonical_name, aliases in SCHEMA_ALIASES.items():
        found = False
        for alias in aliases:
            if alias.lower() in actual_cols:
                col_map[actual_cols[alias.lower()]] = canonical_name
                found = True
                break
        if not found:
            raise ValueError(
                f"Required canonical field '{canonical_name}' could not be resolved from columns: {list(df.columns)}"
            )

    df_mapped = df.rename(columns=col_map)
    return df_mapped, col_map


def load_raw_dataset(
    data_path: Optional[Path] = None, nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Loads raw flights dataset from CSV.
    """
    path = data_path or RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please place 'flights.csv' at this path or configure a download."
        )

    df = pd.read_csv(path, nrows=nrows)
    return df


def clean_and_prepare_dataset(
    df: pd.DataFrame, delay_threshold: float = DELAY_THRESHOLD_MINUTES
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw dataset, maps schema, constructs binary target, filters non-operated flights,
    and returns cleaned DataFrame along with comprehensive audit statistics.
    """
    initial_row_count = len(df)
    initial_col_count = len(df.columns)
    raw_missing_counts = {c: int(df[c].isna().sum()) for c in df.columns}

    # 1. Schema discovery and normalization
    df_clean, mapping = discover_and_map_schema(df)
    key = ["year", "month", "day", "carrier", "origin", "dest", "sched_dep_time"]
    if "flight" in df_clean:
        key.append("flight")
    duplicate_count = int(df_clean.duplicated(key).sum())
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate flight keys; resolve these before modeling."
        )

    # 2. Date construction
    df_clean["flight_date"] = pd.to_datetime(
        df_clean[["year", "month", "day"]], errors="coerce"
    )
    invalid_dates = df_clean["flight_date"].isna().sum()
    if invalid_dates > 0:
        df_clean = df_clean.dropna(subset=["flight_date"])

    # 3. Analyze target and missingness (Operational Population definition)
    # Target: arrival delay >= 15 minutes among operated flights with observed outcome
    raw_target_nulls = int(df_clean[RAW_TARGET_COL].isna().sum())

    # Exclude flights with missing arrival delay (cancellations and diversions)
    df_operated = df_clean[df_clean[RAW_TARGET_COL].notna()].copy()
    excluded_count = raw_target_nulls

    # 4. Binary target construction
    df_operated[TARGET_COL] = (df_operated[RAW_TARGET_COL] >= delay_threshold).astype(
        int
    )

    # 5. Clean string categorical identifiers
    for cat_col in ["carrier", "origin", "dest"]:
        if cat_col in df_operated.columns:
            if df_operated[cat_col].isna().any():
                raise ValueError(f"Missing categorical identifiers in {cat_col}.")
            df_operated[cat_col] = (
                df_operated[cat_col].astype(str).str.strip().str.upper()
            )

    # Route identifier
    df_operated["route"] = df_operated["origin"] + "_" + df_operated["dest"]

    # Distance check
    df_operated["distance"] = pd.to_numeric(df_operated["distance"], errors="coerce")
    df_operated = df_operated[df_operated["distance"] > 0]
    from src.timeutils import hhmm_minutes

    hhmm_minutes(df_operated["sched_dep_time"])
    hhmm_minutes(df_operated["sched_arr_time"])

    # Audit statistics
    target_series = df_operated[TARGET_COL]
    audit_stats = {
        "initial_raw_rows": initial_row_count,
        "initial_raw_cols": initial_col_count,
        "raw_missing_counts": raw_missing_counts,
        "duplicate_flight_keys": duplicate_count,
        "invalid_dates_excluded": int(invalid_dates),
        "invalid_distance_excluded": int(
            len(df_clean[df_clean[RAW_TARGET_COL].notna()]) - len(df_operated)
        ),
        "mapped_columns": mapping,
        "excluded_cancellations_or_missing_target": excluded_count,
        "operated_flights_count": len(df_operated),
        "target_column": TARGET_COL,
        "raw_target_column": RAW_TARGET_COL,
        "delay_threshold_minutes": delay_threshold,
        "delayed_flights_count": int(target_series.sum()),
        "on_time_flights_count": int((target_series == 0).sum()),
        "positive_class_prevalence": float(target_series.mean()),
        "unique_flight_dates": int(df_operated["flight_date"].nunique()),
        "date_min": df_operated["flight_date"].min().strftime("%Y-%m-%d"),
        "date_max": df_operated["flight_date"].max().strftime("%Y-%m-%d"),
        "unique_carriers": int(df_operated["carrier"].nunique()),
        "carrier_list": sorted(df_operated["carrier"].unique().tolist()),
        "unique_origins": int(df_operated["origin"].nunique()),
        "origin_list": sorted(df_operated["origin"].unique().tolist()),
        "unique_destinations": int(df_operated["dest"].nunique()),
        "unique_routes": int(df_operated["route"].nunique()),
    }

    return df_operated, audit_stats


def attach_label_availability(df, reporting_lag_hours=24):
    """Audit arrival timestamps and add a conservative, declared reporting lag.

    These outcome-derived timestamps control which HISTORICAL labels can be
    read. They never enter the predictive feature matrix. The source lacks
    actual publication timestamps, so the 24-hour lag is an explicit assumption.
    """
    from src.timeutils import (
        arrival_timestamps,
        airport_timezones,
        departure_timestamps,
    )

    result = df.copy()
    dep = departure_timestamps(result)
    arrival = arrival_timestamps(result, dep) + pd.to_timedelta(
        result[RAW_TARGET_COL], unit="m"
    )
    clocks_match = pd.Series(False, index=result.index)
    for dest, idx in result.groupby("dest").groups.items():
        local = arrival.loc[idx].dt.tz_convert(airport_timezones()[dest])
        derived = local.dt.hour * 100 + local.dt.minute
        recorded = pd.to_numeric(result.loc[idx, "arr_time"], errors="coerce") % 2400
        clocks_match.loc[idx] = (derived == recorded).to_numpy()
    actual_dep = dep + pd.to_timedelta(result["dep_delay"], unit="m")
    elapsed = (arrival - actual_dep).dt.total_seconds() / 60
    ground_minutes = elapsed - result["air_time"]
    plausible = ground_minutes.between(0, 720) & clocks_match
    result["label_available_at"] = (
        arrival + pd.Timedelta(hours=reporting_lag_hours)
    ).where(plausible)
    audit = {
        "reporting_lag_hours": reporting_lag_hours,
        "arrival_clock_mismatches": int((~clocks_match).sum()),
        "history_labels_excluded_by_timestamp_audit": int((~plausible).sum()),
        "history_labels_usable": int(plausible.sum()),
        "scheduled_arrival_date_assumption": "First destination local schedule clock after departure, trips under 24h.",
        "reporting_assumption": "Arrival outcome becomes available 24h after reconstructed actual arrival; true feed timestamps unavailable.",
    }
    return result, audit


def generate_synthetic_dataset(
    n_samples: int = 1000, random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic synthetic flight dataset fixture for unit testing and offline checks.
    """
    np.random.seed(random_seed)
    carriers = ["UA", "AA", "DL", "B6", "WN", "EV", "MQ"]
    origins = ["EWR", "JFK", "LGA"]
    dests = ["ORD", "ATL", "LAX", "MIA", "BOS", "DFW", "SFO", "DEN"]

    dates = pd.date_range("2013-01-01", periods=60, freq="D")
    sample_dates = pd.to_datetime(np.random.choice(dates, size=n_samples))

    sched_dep_hours = np.random.randint(5, 23, size=n_samples)
    sched_dep_mins = np.random.choice([0, 15, 30, 45, 10, 20, 50], size=n_samples)
    sched_dep_times = sched_dep_hours * 100 + sched_dep_mins

    durations = np.random.randint(60, 360, size=n_samples)
    sched_arr_minutes = (sched_dep_hours * 60 + sched_dep_mins + durations) % 1440
    sched_arr_times = (sched_arr_minutes // 60) * 100 + (sched_arr_minutes % 60)

    # Delays (gamma / exponential distribution with on-time mass)
    delays = np.random.exponential(scale=18.0, size=n_samples) - 10.0
    # Simulate some cancellations
    is_cancelled = np.random.binomial(1, 0.025, size=n_samples).astype(bool)
    delays[is_cancelled] = np.nan

    df = pd.DataFrame(
        {
            "id": np.arange(n_samples),
            "year": sample_dates.year,
            "month": sample_dates.month,
            "day": sample_dates.day,
            "dep_time": np.where(is_cancelled, np.nan, sched_dep_times + 5),
            "sched_dep_time": sched_dep_times,
            "dep_delay": np.where(
                is_cancelled, np.nan, np.random.normal(5, 15, n_samples)
            ),
            "arr_time": np.where(is_cancelled, np.nan, sched_arr_times + 10),
            "sched_arr_time": sched_arr_times,
            "arr_delay": delays,
            "carrier": np.random.choice(carriers, size=n_samples),
            "flight": np.random.randint(100, 9999, size=n_samples),
            "tailnum": [
                "N" + str(np.random.randint(10000, 99999)) for _ in range(n_samples)
            ],
            "origin": np.random.choice(origins, size=n_samples),
            "dest": np.random.choice(dests, size=n_samples),
            "air_time": np.where(is_cancelled, np.nan, durations - 20),
            "distance": np.random.randint(200, 2600, size=n_samples),
            "hour": sched_dep_hours,
            "minute": sched_dep_mins,
            "time_hour": sample_dates.strftime("%Y-%m-%d %H:00:00"),
            "name": "Sample Airline Inc.",
        }
    )
    return df
