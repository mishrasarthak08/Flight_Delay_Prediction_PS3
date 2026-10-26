"""Offline schedule sensitivity with fixed information and explicit assumptions."""

import numpy as np
import pandas as pd
from src.config import ALL_MODEL_FEATURES, HISTORICAL_FEATURES
from src.features import compute_schedule_and_calendar_features
from src.timeutils import departure_timestamps, arrival_timestamps, airport_timezones


class ReschedulingSimulator:
    ALLOWED_SHIFTS = [-60, -30, 0, 30, 60]
    CAUSAL_DISCLAIMER = (
        "DISCLAIMER: These are offline model sensitivity predictions, not causal delay reductions. "
        "Aircraft rotations, airport slots, gates, crews, connections and reactions by other flights "
        "are not modeled. The original historical information set remains fixed. "
        "Timetable counts assume the source schedule is known at the decision time."
    )

    def __init__(
        self,
        model_wrapper,
        preprocessor,
        calibrator=None,
        is_pytorch=False,
        schedule_context=None,
    ):
        self.model, self.preprocessor, self.calibrator = (
            model_wrapper,
            preprocessor,
            calibrator,
        )
        self.is_pytorch, self.schedule_context = is_pytorch, schedule_context

    def simulate_shifts_for_flight(self, flight_row, candidate_shifts=None):
        shifts = (
            self.ALLOWED_SHIFTS if candidate_shifts is None else list(candidate_shifts)
        )
        if not shifts or 0 not in shifts:
            raise ValueError("Candidate shifts must include the zero-shift baseline.")
        if len(set(shifts)) != len(shifts) or any(
            not isinstance(s, (int, np.integer))
            or isinstance(s, bool)
            or abs(s) > 60
            or s % 15
            for s in shifts
        ):
            raise ValueError(
                "Use distinct integer shifts in 15-minute increments between -60 and +60."
            )
        original = pd.DataFrame([flight_row]).reset_index(drop=True)
        dep = departure_timestamps(original).iloc[0]
        arr = arrival_timestamps(original, pd.Series([dep])).iloc[0]
        zones = airport_timezones()
        original_local = dep.tz_convert("America/New_York").tz_localize(None)
        candidates = []
        for shift in shifts:
            row = flight_row.copy()
            new_dep_utc, new_arr_utc = dep + pd.Timedelta(
                minutes=shift
            ), arr + pd.Timedelta(minutes=shift)
            local_dep = new_dep_utc.tz_convert("America/New_York")
            local_arr = new_arr_utc.tz_convert(zones[str(row["dest"])])
            row["flight_date"] = pd.Timestamp(local_dep.date())
            row["sched_dep_time"] = local_dep.hour * 100 + local_dep.minute
            row["sched_arr_time"] = local_arr.hour * 100 + local_arr.minute
            engineered = compute_schedule_and_calendar_features(
                pd.DataFrame([row])
            ).iloc[0]
            # Forecast information remains anchored to the ORIGINAL decision time.
            for col in HISTORICAL_FEATURES:
                engineered[col] = flight_row[col]
            engineered["decision_time"] = flight_row.get(
                "decision_time", dep - pd.Timedelta(hours=2)
            )
            engineered["shift_minutes"] = shift
            if self.schedule_context is not None:
                new_local = local_dep.tz_localize(None)
                hour, window = self.schedule_context.counts_for(
                    row["origin"], new_local
                )
                # Remove the focal scheduled flight from its original bucket, then add it to its candidate bucket.
                old_hour, new_hour = original_local.floor("h"), new_local.floor("h")
                present = (
                    self.schedule_context.counts.get(
                        self.schedule_context.key(row["origin"], old_hour), 0
                    )
                    > 0
                )
                hour += 1 - int(present and old_hour == new_hour)
                window += 1 - int(
                    present and abs((old_hour - new_hour).total_seconds()) <= 3600
                )
                engineered["origin_hour_scheduled_departures"] = hour
                engineered["origin_window_scheduled_departures"] = window
            else:
                for col in [
                    "origin_hour_scheduled_departures",
                    "origin_window_scheduled_departures",
                ]:
                    if col not in engineered:
                        raise ValueError(
                            "Schedule context or explicit scheduled-demand features are required."
                        )
            engineered["scheduled_duration_minutes"] = (
                new_arr_utc - new_dep_utc
            ).total_seconds() / 60
            candidates.append(engineered)
        frame = pd.DataFrame(candidates)
        if self.is_pytorch:
            c, n, _ = self.preprocessor.transform(frame[ALL_MODEL_FEATURES])
            raw = self.model.predict_proba(c, n)
        else:
            raw = self.model.predict_proba(
                self.preprocessor.transform(frame[ALL_MODEL_FEATURES])
            )
        probabilities = (
            self.calibrator.predict_proba(raw) if self.calibrator is not None else raw
        )
        baseline = float(probabilities[shifts.index(0)])
        results = []
        for (_, row), probability in zip(frame.iterrows(), probabilities):
            probability = float(probability)
            risk_difference = probability - baseline
            results.append(
                {
                    "shift_minutes": int(row.shift_minutes),
                    "candidate_departure": row.scheduled_departure_utc.isoformat(),
                    "scheduled_duration_minutes": float(row.scheduled_duration_minutes),
                    "predicted_delay_probability": probability,
                    "absolute_risk_difference": risk_difference,
                    "relative_risk_change_pct": (
                        risk_difference / baseline * 100 if baseline else None
                    ),
                }
            )
        ranked = sorted(
            results,
            key=lambda r: (r["predicted_delay_probability"], abs(r["shift_minutes"])),
        )
        best = ranked[0]
        return {
            "flight_info": {
                "carrier": str(flight_row.carrier),
                "origin": str(flight_row.origin),
                "dest": str(flight_row.dest),
                "original_sched_dep": f"{original_local.hour:02d}:{original_local.minute:02d}",
                "distance": float(flight_row.distance),
            },
            "baseline_delay_probability": baseline,
            "optimal_shift_minutes": best["shift_minutes"],
            "optimal_delay_probability": best["predicted_delay_probability"],
            "best_risk_difference": best["absolute_risk_difference"],
            "scenarios": results,
            "ranked_candidates": ranked,
            "disclaimer": self.CAUSAL_DISCLAIMER,
            "congestion_assumption": "One focal flight shifts against a fixed background timetable.",
        }
