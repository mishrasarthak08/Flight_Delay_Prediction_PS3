"""Schedule demand and historical outcomes available before each decision.

Daily replay reads only outcomes released before local midnight. Estimator
weights stay fixed. A frozen training-only reference is also supported.
"""

from dataclasses import asdict
import numpy as np
import pandas as pd
from src.config import SmoothingConfig, TARGET_COL
from src.timeutils import hhmm_minutes, departure_timestamps


def compute_schedule_and_calendar_features(df):
    df = df.copy()
    if "flight_date" not in df:
        df["flight_date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["flight_date"] = pd.to_datetime(df["flight_date"], errors="raise").dt.normalize()
    for col in ["carrier", "origin", "dest"]:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{col} must contain nonempty identifiers.")
        df[col] = df[col].astype(str).str.strip().str.upper()
    distance = pd.to_numeric(df["distance"], errors="coerce")
    if (~np.isfinite(distance) | (distance <= 0)).any():
        raise ValueError("Distance must be a finite positive number of miles.")
    df["distance"] = distance
    for part in ["dep", "arr"]:
        minutes = hhmm_minutes(df[f"sched_{part}_time"]) % 1440
        df[f"sched_{part}_hour"], df[f"sched_{part}_min"] = minutes // 60, minutes % 60
        df[f"sched_{part}_min_of_day"] = minutes
        df[f"sin_sched_{part}"] = np.sin(2 * np.pi * minutes / 1440)
        df[f"cos_sched_{part}"] = np.cos(2 * np.pi * minutes / 1440)
    date = df["flight_date"]
    df["month"], df["day"], df["day_of_week"] = (
        date.dt.month,
        date.dt.day,
        date.dt.day_of_week,
    )
    df["is_weekend"] = date.dt.day_of_week.isin([5, 6]).astype(int)
    for name, values, period in [
        ("day_of_week", df["day_of_week"], 7),
        ("month", df["month"] - 1, 12),
    ]:
        df[f"sin_{name}"] = np.sin(2 * np.pi * values / period)
        df[f"cos_{name}"] = np.cos(2 * np.pi * values / period)
    df["route"] = df["origin"] + "_" + df["dest"]
    df["scheduled_departure_utc"] = departure_timestamps(df)
    df["decision_time"] = df["scheduled_departure_utc"] - pd.Timedelta(hours=2)
    return df


class ScheduleContext:
    """Static timetable counts, including flights with missing arrival outcomes.

    Assumption: this source timetable is known at T-2h. It is not a versioned
    historical schedule feed. Counts measure scheduled demand, not realized load.
    """

    def __init__(self, counts=None, date_min=None, date_max=None):
        self.counts = counts or {}
        self.date_min, self.date_max = date_min, date_max

    @staticmethod
    def key(origin, timestamp):
        return f"{origin}|{pd.Timestamp(timestamp).strftime('%Y-%m-%dT%H')}"

    @classmethod
    def from_roster(cls, roster):
        dep = (
            departure_timestamps(roster)
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.floor("h")
        )
        table = pd.DataFrame(
            {"origin": roster["origin"].astype(str).str.upper(), "hour": dep}
        )
        counts = {
            cls.key(o, t): int(n)
            for (o, t), n in table.groupby(["origin", "hour"]).size().items()
        }
        return cls(counts, str(dep.min().date()), str(dep.max().date()))

    def counts_for(self, origin, timestamp):
        timestamp = pd.Timestamp(timestamp).floor("h")
        hour = self.counts.get(self.key(origin, timestamp), 0)
        window = sum(
            self.counts.get(self.key(origin, timestamp + pd.Timedelta(hours=h)), 0)
            for h in [-1, 0, 1]
        )
        return hour, window

    def transform(self, df):
        result = df.copy()
        local = (
            result["scheduled_departure_utc"]
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )
        rows = [self.counts_for(o, t) for o, t in zip(result["origin"], local)]
        result["origin_hour_scheduled_departures"] = [r[0] for r in rows]
        result["origin_window_scheduled_departures"] = [r[1] for r in rows]
        return result

    def to_dict(self):
        return {
            "counts": self.counts,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "assumption": "Static source timetable known at T-2h; counts include missing outcomes.",
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["counts"], data["date_min"], data["date_max"])


class HistoricalFeatureGenerator:
    """Timestamp-filtered history with a fixed Beta(1,3) cold start.

    Global smoothing uses only previously available labels. Persisted frozen
    snapshots reject prediction times earlier than their state cutoff.
    """

    groups = ["carrier", "origin", "dest", "route"]

    def __init__(self, config=None):
        self.config = config or SmoothingConfig()
        self.prior = self.config.default_prior_delay_rate
        for group in self.groups:
            setattr(self, f"{group}_m", getattr(self.config, f"{group}_m"))
            setattr(self, f"latest_{group}_stats", {})
        self.is_fitted, self.state_cutoff, self.recent_stats = False, None, {}
        self.history_policy = "frozen"

    def enable_rolling(self):
        """Replay daily snapshots using only outcomes released before local midnight."""
        self.history_policy = "daily_available_outcomes"
        return self

    def fit(self, df, date_col="flight_date", target_col=TARGET_COL):
        if target_col not in df:
            raise ValueError(f"Target column {target_col!r} is required.")
        self.source = df.copy()
        if "label_available_at" not in self.source:
            # Synthetic fixtures. The real pipeline supplies audited arrival times.
            self.source["label_available_at"] = pd.to_datetime(
                self.source[date_col], utc=True
            ) + pd.Timedelta(days=1)
        self.source["label_available_at"] = pd.to_datetime(
            self.source["label_available_at"], utc=True
        )
        self.source = self.source[self.source["label_available_at"].notna()].copy()
        self.target_col, self.is_fitted = target_col, True
        self.freeze(
            self.source["label_available_at"].max() + pd.Timedelta(nanoseconds=1)
        )
        return self

    def _queries(self, df):
        col = "decision_time" if "decision_time" in df else "flight_date"
        timestamps = pd.to_datetime(df[col], utc=True)
        if self.history_policy == "daily_available_outcomes":
            timestamps = (
                timestamps.dt.tz_convert("America/New_York")
                .dt.normalize()
                .dt.tz_convert("UTC")
            )
        return timestamps.astype("int64").to_numpy()

    @staticmethod
    def _lookup(events, queries, target):
        events = events.sort_values("label_available_at", kind="stable")
        times = events["label_available_at"].astype("int64").to_numpy()
        cumulative = np.r_[0.0, events[target].to_numpy(dtype=float).cumsum()]
        indices = np.searchsorted(times, queries, side="left")
        return cumulative[indices], indices.astype(float)

    def transform_expanding(self, df):
        if not self.is_fitted or not hasattr(self, "source"):
            raise RuntimeError("Fit historical events before expanding transformation.")
        result, queries = df.copy(), self._queries(df)
        gd, gn = self._lookup(self.source, queries, self.target_col)
        prior = (gd + 4 * self.prior) / (gn + 4)
        lookback = int(pd.Timedelta(days=self.config.recent_window_days).value)
        for group in self.groups:
            m = getattr(self, f"{group}_m")
            delays, n = np.zeros(len(result)), np.zeros(len(result))
            recent_delays, recent_n = np.zeros(len(result)), np.zeros(len(result))
            events_by_key = dict(tuple(self.source.groupby(group)))
            for key, pos in result.groupby(group).indices.items():
                events = events_by_key.get(key)
                if events is None:
                    continue
                delays[pos], n[pos] = self._lookup(
                    events, queries[pos], self.target_col
                )
                old_d, old_n = self._lookup(
                    events, queries[pos] - lookback, self.target_col
                )
                recent_delays[pos], recent_n[pos] = delays[pos] - old_d, n[pos] - old_n
            result[f"{group}_hist_delay_rate"] = (delays + m * prior) / (n + m)
            result[f"hist_{group}_flight_count"] = n
            if group in ["carrier", "origin", "route"]:
                result[f"{group}_recent_delay_rate"] = (recent_delays + m * prior) / (
                    recent_n + m
                )
        return result

    def fit_transform_chronological(
        self, df, date_col="flight_date", target_col=TARGET_COL
    ):
        return self.fit(df, date_col, target_col).transform_expanding(df)

    def freeze(self, cutoff):
        cutoff = pd.Timestamp(cutoff)
        cutoff = (
            cutoff.tz_localize("UTC")
            if cutoff.tzinfo is None
            else cutoff.tz_convert("UTC")
        )
        observed = self.source[self.source["label_available_at"] < cutoff]
        self.frozen_prior = float(
            (observed[self.target_col].sum() + 4 * self.prior) / (len(observed) + 4)
        )
        self.state_cutoff = cutoff.isoformat()
        recent = observed[
            observed["label_available_at"]
            >= cutoff - pd.Timedelta(days=self.config.recent_window_days)
        ]
        for group in self.groups:
            stats = observed.groupby(group)[self.target_col].agg(["sum", "count"])
            setattr(
                self,
                f"latest_{group}_stats",
                {str(k): (int(r["sum"]), int(r["count"])) for k, r in stats.iterrows()},
            )
            stats = recent.groupby(group)[self.target_col].agg(["sum", "count"])
            self.recent_stats[group] = {
                str(k): (int(r["sum"]), int(r["count"])) for k, r in stats.iterrows()
            }
        return self

    def transform_inference(self, df):
        if not self.is_fitted:
            raise RuntimeError(
                "HistoricalFeatureGenerator must be fitted before inference."
            )
        if self.history_policy == "daily_available_outcomes":
            if not hasattr(self, "source"):
                raise RuntimeError(
                    "Daily replay requires the saved historical_events.parquet store."
                )
            return self.transform_expanding(df)
        if (
            self.state_cutoff is not None
            and (self._queries(df) < pd.Timestamp(self.state_cutoff).value).any()
        ):
            raise ValueError(
                "Prediction precedes the saved historical-state cutoff. Use a historical snapshot."
            )
        result, prior = df.copy(), getattr(self, "frozen_prior", self.prior)
        for group in self.groups:
            m, stats = getattr(self, f"{group}_m"), getattr(
                self, f"latest_{group}_stats"
            )
            values = [stats.get(str(key), (0, 0)) for key in result[group]]
            result[f"{group}_hist_delay_rate"] = [
                (d + m * prior) / (n + m) for d, n in values
            ]
            result[f"hist_{group}_flight_count"] = [n for _, n in values]
            if group in ["carrier", "origin", "route"]:
                recent = self.recent_stats.get(group, {})
                values = [recent.get(str(key), (0, 0)) for key in result[group]]
                result[f"{group}_recent_delay_rate"] = [
                    (d + m * prior) / (n + m) for d, n in values
                ]
        return result

    def to_dict(self):
        return {
            "schema_version": 2,
            "is_fitted": self.is_fitted,
            "history_policy": self.history_policy,
            "config": asdict(self.config),
            "state_cutoff": self.state_cutoff,
            "frozen_prior": getattr(self, "frozen_prior", self.prior),
            "recent_stats": self.recent_stats,
            **{
                f"latest_{g}_stats": getattr(self, f"latest_{g}_stats")
                for g in self.groups
            },
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("schema_version") != 2:
            raise ValueError(
                "Historical artifact predates the corrected protocol. Rerun the pipeline."
            )
        gen = cls(SmoothingConfig(**data["config"]))
        gen.is_fitted, gen.state_cutoff = data["is_fitted"], data["state_cutoff"]
        gen.history_policy = data.get("history_policy", "frozen")
        gen.frozen_prior, gen.recent_stats = data["frozen_prior"], data["recent_stats"]
        for group in gen.groups:
            setattr(gen, f"latest_{group}_stats", data[f"latest_{group}_stats"])
        return gen
