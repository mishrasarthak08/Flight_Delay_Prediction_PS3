"""
Offline Departure Rescheduling Simulator.
Provides what-if schedule shift analysis under explicit operational assumptions.
NOTE: This is a predictive sensitivity tool, NOT a causal optimization engine.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_SCHEDULE_FEATURES,
    HISTORICAL_FEATURES,
)


class ReschedulingSimulator:
    """
    Simulates departure time shifts (e.g. -60, -30, 0, +30, +60 mins) to assess
    changes in predicted delay probability under pre-departure schedule features.
    """

    ALLOWED_SHIFTS: List[int] = [-60, -30, 0, 30, 60]

    CAUSAL_DISCLAIMER: str = (
        "DISCLAIMER: This offline rescheduling simulator calculates what-if model risk predictions "
        "based on historically observed schedule patterns. It does NOT prove causal delay reduction. "
        "Real-world flight rescheduling must account for aircraft tail rotations, airport slot constraints, "
        "gate availability, FAA ground delay programs, crew duty limitations, and passenger connections, "
        "none of which are modeled in this statistical analysis."
    )

    def __init__(
        self,
        model_wrapper: Any,
        preprocessor: Any,
        calibrator: Optional[Any] = None,
        is_pytorch: bool = False,
    ):
        self.model = model_wrapper
        self.preprocessor = preprocessor
        self.calibrator = calibrator
        self.is_pytorch = is_pytorch

    def simulate_shifts_for_flight(
        self,
        flight_row: pd.Series,
        candidate_shifts: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates candidate departure shifts for a single flight record.
        Preserves scheduled duration, route, carrier, distance, and historical features.
        """
        shifts = candidate_shifts or self.ALLOWED_SHIFTS
        candidates: List[pd.Series] = []

        base_dep_min_of_day = int(flight_row["sched_dep_min_of_day"])
        base_arr_min_of_day = int(flight_row["sched_arr_min_of_day"])

        for shift in shifts:
            row = flight_row.copy()
            new_dep_min = (base_dep_min_of_day + shift) % 1440
            new_arr_min = (base_arr_min_of_day + shift) % 1440

            row["shift_minutes"] = shift
            row["sched_dep_min_of_day"] = new_dep_min
            row["sched_dep_hour"] = new_dep_min // 60
            row["sched_dep_min"] = new_dep_min % 60
            row["sin_sched_dep"] = np.sin(2.0 * np.pi * new_dep_min / 1440.0)
            row["cos_sched_dep"] = np.cos(2.0 * np.pi * new_dep_min / 1440.0)

            row["sched_arr_min_of_day"] = new_arr_min
            row["sched_arr_hour"] = new_arr_min // 60
            row["sched_arr_min"] = new_arr_min % 60
            row["sin_sched_arr"] = np.sin(2.0 * np.pi * new_arr_min / 1440.0)
            row["cos_sched_arr"] = np.cos(2.0 * np.pi * new_arr_min / 1440.0)

            candidates.append(row)

        df_candidates = pd.DataFrame(candidates)
        feature_cols = CATEGORICAL_FEATURES + NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES

        # Predict probabilities
        if self.is_pytorch:
            c_tens, n_tens, _ = self.preprocessor.transform(df_candidates[feature_cols])
            probs_raw = self.model.predict_proba(c_tens, n_tens)
        else:
            X_trans = self.preprocessor.transform(df_candidates[feature_cols])
            probs_raw = self.model.predict_proba(X_trans)

        if self.calibrator is not None:
            probs = self.calibrator.predict_proba(probs_raw)
        else:
            probs = probs_raw

        # Baseline probability (shift == 0)
        baseline_idx = shifts.index(0) if 0 in shifts else 0
        baseline_prob = float(probs[baseline_idx])

        results = []
        for shift, prob in zip(shifts, probs):
            prob = float(prob)
            risk_diff = prob - baseline_prob
            pct_change = (risk_diff / baseline_prob * 100.0) if baseline_prob > 0 else 0.0

            results.append({
                "shift_minutes": int(shift),
                "predicted_delay_probability": float(round(prob, 4)),
                "absolute_risk_difference": float(round(risk_diff, 4)),
                "relative_risk_change_pct": float(round(pct_change, 2)),
            })

        # Rank candidates: minimize predicted probability; tie-break by smallest absolute shift
        ranked = sorted(results, key=lambda x: (x["predicted_delay_probability"], abs(x["shift_minutes"])))
        optimal_candidate = ranked[0]

        return {
            "flight_info": {
                "carrier": str(flight_row.get("carrier", "N/A")),
                "origin": str(flight_row.get("origin", "N/A")),
                "dest": str(flight_row.get("dest", "N/A")),
                "original_sched_dep": f"{int(flight_row.get('sched_dep_hour', 0)):02d}:{int(flight_row.get('sched_dep_min', 0)):02d}",
                "distance": int(flight_row.get("distance", 0)),
            },
            "baseline_delay_probability": float(round(baseline_prob, 4)),
            "optimal_shift_minutes": int(optimal_candidate["shift_minutes"]),
            "optimal_delay_probability": float(optimal_candidate["predicted_delay_probability"]),
            "best_risk_difference": float(optimal_candidate["absolute_risk_difference"]),
            "scenarios": results,
            "ranked_candidates": ranked,
            "disclaimer": self.CAUSAL_DISCLAIMER,
        }
