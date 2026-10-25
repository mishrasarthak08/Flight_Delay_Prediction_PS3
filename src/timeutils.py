"""Timezone-aware schedule timestamps for the 2013 NYC departure population."""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def airport_timezones():
    path = Path(__file__).resolve().parents[1] / "data/metadata/airport_timezones.json"
    return json.loads(path.read_text())["timezones"]


def hhmm_minutes(values):
    """Reject corrupt clocks. The dataset's 2400 convention denotes midnight."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & (numeric == np.floor(numeric))
    valid &= (numeric.between(0, 2359) & ((numeric % 100) < 60)) | (numeric == 2400)
    if not valid.all():
        raise ValueError(
            "Schedule times must be valid integer HHMM clocks (0000-2359 or 2400)."
        )
    return (numeric // 100 * 60 + numeric % 100).astype(int)


def departure_timestamps(df):
    dates = pd.to_datetime(df["flight_date"], errors="raise").dt.normalize()
    local = dates + pd.to_timedelta(hhmm_minutes(df["sched_dep_time"]), unit="m")
    return local.dt.tz_localize(
        "America/New_York", ambiguous="raise", nonexistent="raise"
    ).dt.tz_convert("UTC")


def arrival_timestamps(df, departure_utc=None):
    """First scheduled destination clock after departure, for trips under 24h."""
    dep = departure_timestamps(df) if departure_utc is None else departure_utc
    zones = airport_timezones()
    result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    minutes = hhmm_minutes(df["sched_arr_time"])
    for dest, indices in df.groupby("dest").groups.items():
        if dest not in zones:
            raise ValueError(f"No timezone metadata for destination {dest!r}.")
        zone = zones[dest]
        base = dep.loc[indices].dt.tz_convert(zone).dt.tz_localize(None).dt.normalize()
        local = base + pd.to_timedelta(minutes.loc[indices], unit="m")
        candidate = (
            local.dt.tz_localize(zone, ambiguous=False, nonexistent="shift_forward")
            .dt.tz_convert("UTC")
            .copy()
        )
        earlier = candidate < dep.loc[indices]
        if earlier.any():
            candidate.loc[earlier] = (
                (local.loc[earlier] + pd.Timedelta(days=1))
                .dt.tz_localize(zone, ambiguous=False, nonexistent="shift_forward")
                .dt.tz_convert("UTC")
            )
        result.loc[indices] = candidate
    return result
