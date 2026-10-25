# Flight Delay Prediction - Dataset Audit Report

## 1. Executive Summary
- **Source File**: `flights.csv`
- **Total Raw Records**: 336,776
- **Total Raw Columns**: 21
- **Operated Flights with Observed Outcome**: 327,346 (97.20%)
- **Excluded Flights (Cancellations & Diversions)**: 9,430 (2.80%)

## 2. Operational Prediction Contract
- **Declared Prediction Timestamp**: 2 hours before scheduled departure (`CRSDepTime` / `sched_dep_time`).
- **Target Variable**: `is_delayed` (Binary: 1 if `arr_delay` >= 15.0 mins, else 0).
- **Delayed Flights (Target = 1)**: 80,100
- **On-Time Flights (Target = 0)**: 247,246
- **Positive Class Prevalence**: **24.47%**

## 3. Date & Geographic Coverage
- **Temporal Range**: 2013-01-01 to 2013-12-31 (365 unique calendar dates)
- **Origin Airports (3)**: EWR, JFK, LGA
- **Unique Destination Airports**: 104
- **Unique Airline Carriers (16)**: 9E, AA, AS, B6, DL, EV, F9, FL, HA, MQ, OO, UA, US, VX, WN, YV
- **Unique Operational Routes**: 223

## 4. Missing Value Breakdown (Raw Data)
| Column Name | Missing Count | Missing Percentage |
| :--- | :--- | :--- |
| `id` | 0 | 0.00% |
| `year` | 0 | 0.00% |
| `month` | 0 | 0.00% |
| `day` | 0 | 0.00% |
| `dep_time` | 8,255 | 2.45% |
| `sched_dep_time` | 0 | 0.00% |
| `dep_delay` | 8,255 | 2.45% |
| `arr_time` | 8,713 | 2.59% |
| `sched_arr_time` | 0 | 0.00% |
| `arr_delay` | 9,430 | 2.80% |
| `carrier` | 0 | 0.00% |
| `flight` | 0 | 0.00% |
| `tailnum` | 2,512 | 0.75% |
| `origin` | 0 | 0.00% |
| `dest` | 0 | 0.00% |
| `air_time` | 9,430 | 2.80% |
| `distance` | 0 | 0.00% |
| `hour` | 0 | 0.00% |
| `minute` | 0 | 0.00% |
| `time_hour` | 0 | 0.00% |
| `name` | 0 | 0.00% |

## 5. Leakage Prevention and Column Classification
- **Allowed Pre-Departure Features**: Scheduled departure/arrival hour and minute, cyclical transformations, distance, route, airline carrier, origin, destination, calendar date indicators, and past-only smoothed historical delay rates.
- **Prohibited Outcome Fields**: `dep_time`, `dep_delay`, `arr_time`, `arr_delay`, `air_time`, `taxi_out`, `taxi_in`, `wheels_off`, `wheels_on`, delay causal breakdowns. All post-departure and current-flight outcome fields are strictly excluded from model feature matrices.
