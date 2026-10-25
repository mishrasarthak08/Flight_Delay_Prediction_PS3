"""
Feature engineering module.
Extracts cyclical schedule representations, calendar signals, and calculates
leakage-safe empirical Bayes smoothed historical delay rates using ONLY strictly earlier dates.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from src.config import (
    SmoothingConfig,
    NUMERIC_SCHEDULE_FEATURES,
    HISTORICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)


def compute_schedule_and_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes scheduled departure and arrival time representations, cyclical transformations,
    and calendar indicators using exclusively pre-departure scheduled information.
    """
    df = df.copy()

    # 1. Scheduled departure decomposition
    # sched_dep_time is in HHMM format (e.g. 515 for 05:15, 1730 for 17:30)
    sched_dep = df["sched_dep_time"].astype(int)
    dep_hour = (sched_dep // 100) % 24
    dep_min = sched_dep % 100
    dep_min_of_day = dep_hour * 60 + dep_min

    df["sched_dep_hour"] = dep_hour
    df["sched_dep_min"] = dep_min
    df["sched_dep_min_of_day"] = dep_min_of_day
    # Cyclical encoding of departure time (24 * 60 = 1440 minutes in a day)
    df["sin_sched_dep"] = np.sin(2.0 * np.pi * dep_min_of_day / 1440.0)
    df["cos_sched_dep"] = np.cos(2.0 * np.pi * dep_min_of_day / 1440.0)

    # 2. Scheduled arrival decomposition
    sched_arr = df["sched_arr_time"].astype(int)
    arr_hour = (sched_arr // 100) % 24
    arr_min = sched_arr % 100
    arr_min_of_day = arr_hour * 60 + arr_min

    df["sched_arr_hour"] = arr_hour
    df["sched_arr_min"] = arr_min
    df["sched_arr_min_of_day"] = arr_min_of_day
    # Cyclical encoding of arrival time
    df["sin_sched_arr"] = np.sin(2.0 * np.pi * arr_min_of_day / 1440.0)
    df["cos_sched_arr"] = np.cos(2.0 * np.pi * arr_min_of_day / 1440.0)

    # 3. Calendar patterns from flight_date
    if "flight_date" in df.columns:
        dt = pd.to_datetime(df["flight_date"])
        df["month"] = dt.dt.month
        df["day"] = dt.dt.day
        df["day_of_week"] = dt.dt.day_of_week
        df["is_weekend"] = dt.dt.day_of_week.isin([5, 6]).astype(int)
    else:
        # Fallback to year/month/day columns if present
        dt = pd.to_datetime(df[["year", "month", "day"]])
        df["month"] = dt.dt.month
        df["day"] = dt.dt.day
        df["day_of_week"] = dt.dt.day_of_week
        df["is_weekend"] = dt.dt.day_of_week.isin([5, 6]).astype(int)

    # Cyclical day of week (7 days) and month (12 months)
    df["sin_day_of_week"] = np.sin(2.0 * np.pi * df["day_of_week"] / 7.0)
    df["cos_day_of_week"] = np.cos(2.0 * np.pi * df["day_of_week"] / 7.0)
    df["sin_month"] = np.sin(2.0 * np.pi * (df["month"] - 1) / 12.0)
    df["cos_month"] = np.cos(2.0 * np.pi * (df["month"] - 1) / 12.0)

    # Route string
    if "route" not in df.columns and "origin" in df.columns and "dest" in df.columns:
        df["route"] = df["origin"] + "_" + df["dest"]

    return df


class HistoricalFeatureGenerator:
    """
    Computes leakage-safe empirical Bayes smoothed historical delay rates.
    GUARANTEE: For any flight on calendar date D, historical aggregates are computed
    STRICTLY from flights on dates d < D. Flights on date D or later never enter calculations.
    """

    def __init__(self, config: SmoothingConfig = SmoothingConfig()):
        self.config = config
        self.carrier_m = config.carrier_m
        self.origin_m = config.origin_m
        self.dest_m = config.dest_m
        self.route_m = config.route_m
        self.prior = config.default_prior_delay_rate

        # Historical lookup state up to the end of training period (for inference & rescheduling)
        self.latest_carrier_stats: Dict[str, Tuple[int, int]] = {}  # {carrier: (delayed_count, total_count)}
        self.latest_origin_stats: Dict[str, Tuple[int, int]] = {}
        self.latest_dest_stats: Dict[str, Tuple[int, int]] = {}
        self.latest_route_stats: Dict[str, Tuple[int, int]] = {}
        self.is_fitted = False

    def fit_transform_chronological(
        self,
        df: pd.DataFrame,
        date_col: str = "flight_date",
        target_col: str = TARGET_COL,
    ) -> pd.DataFrame:
        """
        Computes historical delay features sequentially across chronological dates.
        For each date D:
          - Looks up cumulative stats from all prior dates d < D.
          - Computes smoothed rate: (prior_delays + m * prior) / (prior_flights + m)
          - Assigns features to flights of date D.
          - Updates cumulative stats by adding outcomes from date D for subsequent dates.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' required to fit historical delay stats.")

        df = df.copy()
        df["_date"] = pd.to_datetime(df[date_col]).dt.date
        unique_dates = sorted(df["_date"].unique())

        # Cumulative trackers across history: {entity: [delayed_count, total_count]}
        carrier_cum: Dict[str, List[int]] = {}
        origin_cum: Dict[str, List[int]] = {}
        dest_cum: Dict[str, List[int]] = {}
        route_cum: Dict[str, List[int]] = {}

        # Arrays to populate
        n_rows = len(df)
        carrier_rate = np.zeros(n_rows, dtype=np.float32)
        origin_rate = np.zeros(n_rows, dtype=np.float32)
        dest_rate = np.zeros(n_rows, dtype=np.float32)
        route_rate = np.zeros(n_rows, dtype=np.float32)

        carrier_count = np.zeros(n_rows, dtype=np.float32)
        origin_count = np.zeros(n_rows, dtype=np.float32)
        dest_count = np.zeros(n_rows, dtype=np.float32)
        route_count = np.zeros(n_rows, dtype=np.float32)

        # Map date to row indices
        date_groups = df.groupby("_date").indices

        # Prior delay rate fallback
        prior = self.prior

        for d in unique_dates:
            idxs = date_groups[d]
            carriers = df["carrier"].iloc[idxs].values
            origins = df["origin"].iloc[idxs].values
            dests = df["dest"].iloc[idxs].values
            routes = df["route"].iloc[idxs].values
            targets = df[target_col].iloc[idxs].values

            # 1. READ strictly prior cumulative counts for current date flights
            for i, idx in enumerate(idxs):
                c = carriers[i]
                o = origins[i]
                dst = dests[i]
                r = routes[i]

                # Carrier
                c_del, c_tot = carrier_cum.get(c, [0, 0])
                carrier_rate[idx] = (c_del + self.carrier_m * prior) / (c_tot + self.carrier_m)
                carrier_count[idx] = c_tot

                # Origin
                o_del, o_tot = origin_cum.get(o, [0, 0])
                origin_rate[idx] = (o_del + self.origin_m * prior) / (o_tot + self.origin_m)
                origin_count[idx] = o_tot

                # Dest
                dst_del, dst_tot = dest_cum.get(dst, [0, 0])
                dest_rate[idx] = (dst_del + self.dest_m * prior) / (dst_tot + self.dest_m)
                dest_count[idx] = dst_tot

                # Route
                r_del, r_tot = route_cum.get(r, [0, 0])
                route_rate[idx] = (r_del + self.route_m * prior) / (r_tot + self.route_m)
                route_count[idx] = r_tot

            # 2. UPDATE cumulative counts AFTER assigning features for date D
            # This ensures date D outcomes are only available for future dates d' > D
            for i in range(len(idxs)):
                c = carriers[i]
                o = origins[i]
                dst = dests[i]
                r = routes[i]
                y = int(targets[i])

                # Update carrier
                c_entry = carrier_cum.setdefault(c, [0, 0])
                c_entry[0] += y
                c_entry[1] += 1

                # Update origin
                o_entry = origin_cum.setdefault(o, [0, 0])
                o_entry[0] += y
                o_entry[1] += 1

                # Update dest
                dst_entry = dest_cum.setdefault(dst, [0, 0])
                dst_entry[0] += y
                dst_entry[1] += 1

                # Update route
                r_entry = route_cum.setdefault(r, [0, 0])
                r_entry[0] += y
                r_entry[1] += 1

        # Store latest stats for future inference / simulation
        self.latest_carrier_stats = {k: (v[0], v[1]) for k, v in carrier_cum.items()}
        self.latest_origin_stats = {k: (v[0], v[1]) for k, v in origin_cum.items()}
        self.latest_dest_stats = {k: (v[0], v[1]) for k, v in dest_cum.items()}
        self.latest_route_stats = {k: (v[0], v[1]) for k, v in route_cum.items()}
        self.is_fitted = True

        df["carrier_hist_delay_rate"] = carrier_rate
        df["origin_hist_delay_rate"] = origin_rate
        df["dest_hist_delay_rate"] = dest_rate
        df["route_hist_delay_rate"] = route_rate

        df["hist_carrier_flight_count"] = carrier_count
        df["hist_origin_flight_count"] = origin_count
        df["hist_dest_flight_count"] = dest_count
        df["hist_route_flight_count"] = route_count

        df = df.drop(columns=["_date"])
        return df

    def transform_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attaches historical delay features to an unlabelled inference or rescheduling DataFrame
        using the fitted historical state. Unseen entities gracefully revert to the prior delay rate.
        """
        if not self.is_fitted:
            raise RuntimeError("HistoricalFeatureGenerator must be fitted before calling transform_inference.")

        df = df.copy()
        n = len(df)
        prior = self.prior

        carrier_rate = np.zeros(n, dtype=np.float32)
        origin_rate = np.zeros(n, dtype=np.float32)
        dest_rate = np.zeros(n, dtype=np.float32)
        route_rate = np.zeros(n, dtype=np.float32)

        carrier_count = np.zeros(n, dtype=np.float32)
        origin_count = np.zeros(n, dtype=np.float32)
        dest_count = np.zeros(n, dtype=np.float32)
        route_count = np.zeros(n, dtype=np.float32)

        for i in range(n):
            c = df["carrier"].iloc[i]
            o = df["origin"].iloc[i]
            dst = df["dest"].iloc[i]
            r = df["route"].iloc[i] if "route" in df.columns else f"{o}_{dst}"

            c_del, c_tot = self.latest_carrier_stats.get(c, (0, 0))
            carrier_rate[i] = (c_del + self.carrier_m * prior) / (c_tot + self.carrier_m)
            carrier_count[i] = c_tot

            o_del, o_tot = self.latest_origin_stats.get(o, (0, 0))
            origin_rate[i] = (o_del + self.origin_m * prior) / (o_tot + self.origin_m)
            origin_count[i] = o_tot

            dst_del, dst_tot = self.latest_dest_stats.get(dst, (0, 0))
            dest_rate[i] = (dst_del + self.dest_m * prior) / (dst_tot + self.dest_m)
            dest_count[i] = dst_tot

            r_del, r_tot = self.latest_route_stats.get(r, (0, 0))
            route_rate[i] = (r_del + self.route_m * prior) / (r_tot + self.route_m)
            route_count[i] = r_tot

        df["carrier_hist_delay_rate"] = carrier_rate
        df["origin_hist_delay_rate"] = origin_rate
        df["dest_hist_delay_rate"] = dest_rate
        df["route_hist_delay_rate"] = route_rate

        df["hist_carrier_flight_count"] = carrier_count
        df["hist_origin_flight_count"] = origin_count
        df["hist_dest_flight_count"] = dest_count
        df["hist_route_flight_count"] = route_count

        return df

    def to_dict(self) -> Dict[str, Any]:
        """Serializes fitted historical stats for checkpointing."""
        return {
            "is_fitted": self.is_fitted,
            "carrier_m": self.carrier_m,
            "origin_m": self.origin_m,
            "dest_m": self.dest_m,
            "route_m": self.route_m,
            "prior": self.prior,
            "latest_carrier_stats": self.latest_carrier_stats,
            "latest_origin_stats": self.latest_origin_stats,
            "latest_dest_stats": self.latest_dest_stats,
            "latest_route_stats": self.latest_route_stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoricalFeatureGenerator":
        """Reconstructs generator from dictionary."""
        config = SmoothingConfig(
            carrier_m=data["carrier_m"],
            origin_m=data["origin_m"],
            dest_m=data["dest_m"],
            route_m=data["route_m"],
            default_prior_delay_rate=data["prior"],
        )
        gen = cls(config)
        gen.is_fitted = data["is_fitted"]
        gen.latest_carrier_stats = {k: tuple(v) for k, v in data["latest_carrier_stats"].items()}
        gen.latest_origin_stats = {k: tuple(v) for k, v in data["latest_origin_stats"].items()}
        gen.latest_dest_stats = {k: tuple(v) for k, v in data["latest_dest_stats"].items()}
        gen.latest_route_stats = {k: tuple(v) for k, v in data["latest_route_stats"].items()}
        return gen
