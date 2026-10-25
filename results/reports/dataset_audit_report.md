# Dataset audit

Raw scheduled flights: **336,776**. Observed arrival outcomes: **327,346**.
Missing arrival outcomes excluded from supervision: **9,430**. Their individual exclusion reasons are not fully verified.
Target: arrival delay at least 15 minutes. Positive prevalence: **24.47%**.
Duplicate flight keys: **0**. Raw destinations: **105**. Supervised destinations: **104**.

## Missingness

| Column | Missing rows | Fraction |
|---|---:|---:|
| id | 0 | 0.00% |
| year | 0 | 0.00% |
| month | 0 | 0.00% |
| day | 0 | 0.00% |
| dep_time | 8,255 | 2.45% |
| sched_dep_time | 0 | 0.00% |
| dep_delay | 8,255 | 2.45% |
| arr_time | 8,713 | 2.59% |
| sched_arr_time | 0 | 0.00% |
| arr_delay | 9,430 | 2.80% |
| carrier | 0 | 0.00% |
| flight | 0 | 0.00% |
| tailnum | 2,512 | 0.75% |
| origin | 0 | 0.00% |
| dest | 0 | 0.00% |
| air_time | 9,430 | 2.80% |
| distance | 0 | 0.00% |
| hour | 0 | 0.00% |
| minute | 0 | 0.00% |
| time_hour | 0 | 0.00% |
| name | 0 | 0.00% |

## Historical label availability

|                                            | value                                                                                                       |
|:-------------------------------------------|:------------------------------------------------------------------------------------------------------------|
| reporting_lag_hours                        | 24                                                                                                          |
| arrival_clock_mismatches                   | 1                                                                                                           |
| history_labels_excluded_by_timestamp_audit | 1                                                                                                           |
| history_labels_usable                      | 327345                                                                                                      |
| scheduled_arrival_date_assumption          | First destination local schedule clock after departure, trips under 24h.                                    |
| reporting_assumption                       | Arrival outcome becomes available 24h after reconstructed actual arrival; true feed timestamps unavailable. |

Current-flight outcomes are never predictors. These timestamps govern release of historical labels only.
The source lacks actual publication timestamps. Arrival plus 24 hours is a declared assumption.

## Provenance

CSV SHA-256: `8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa`

Source: https://www.kaggle.com/datasets/matinsajadi/flights
Original schema: https://nycflights13.tidyverse.org/reference/flights.html

This raw audit does not replace results/reports/dataset_profile.json, which belongs to the trained run.
