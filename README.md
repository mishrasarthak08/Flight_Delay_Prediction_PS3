# Flight Delay Prediction and Rescheduling

**Team T001 · Applied Machine Learning capstone · Phase 1**

Phase 1 machine learning evaluation of 2013 NYC departures. The required comparison is **Logistic Regression versus XGBoost**. A categorical embedding MLP is included as a supplementary experiment.

The task predicts arrival delay of at least 15 minutes **two hours before scheduled departure**, conditional on an observed arrival outcome. The app is a historical demonstrator. It has not been validated for current flight operations.

## Setup and reproduction

Use Python 3.12. The pinned SciPy version requires Python 3.12, so earlier versions are intentionally unsupported. Run commands from this repository's root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[dev,notebooks]'
```

On macOS, install the native OpenMP runtime before using XGBoost:

```bash
brew install libomp
```

To present the saved model without retraining or downloading the raw dataset:

```bash
python -m streamlit run app/app.py
```

Use the bundled historical examples in the app. For a complete rebuild, continue with the commands below.

The exact primary package versions are pinned in `pyproject.toml` and `requirements.txt`. The saved run records the actual Python/package versions and source/data hashes. The native OpenMP dependency is separate from Python packages.

```bash
# Public dataset download with a pinned complete-file SHA-256 check.
python scripts/download_data.py
python scripts/download_data.py --verify-only

# Full date-based selection, training, calibration, evaluation and sensitivity run.
python scripts/run_pipeline.py --seed 42 --epochs 15

# Historical examples and README table from the new artifacts.
python scripts/build_demo.py

# Verify metrics, threshold selection and saved inference against original records.
python scripts/verify_artifacts.py

# Rebuild and execute the analytical notebooks in the active environment.
python -m ipykernel install --user --name aml-capstone
python scripts/build_notebooks.py --execute --kernel aml-capstone

# Static checks, tests and historical app.
python -m black --check src scripts app tests
python -m ruff check src scripts app tests
python -m pytest
python -m streamlit run app/app.py
```

The download needs internet access. Raw data remains ignored by Git. The committed models, prediction artifacts and demo examples support a historical demonstration without downloading raw data; full retraining, raw audits and record-level replay verification require the pinned CSV.

For an ML-only run, use `--skip-mlp` with fresh `--models-dir` and `--results-dir` directories so old neural artifacts cannot be mixed with a new bundle. Most coursework demonstrations should use the complete saved bundle rather than changing its configuration.

## Results

The original test outputs were inspected during code review. These results are a **corrected retrospective reevaluation**, not a newly untouched external test. All model, calibration and threshold choices in the final implementation use the documented training/development partitions, never final test scores.

<!-- BEGIN GENERATED RESULTS -->
| Model | Raw AP | Raw ROC-AUC | Selected Brier | Accuracy | Precision | Recall | Calibration |
|---|---:|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.3200 | 0.5797 | 0.2396 | 68.17% | 33.99% | 19.67% | none |
| XGBoost | 0.3722 | 0.6367 | 0.1910 | 56.79% | 34.49% | 67.74% | none |
| PyTorch Embedding MLP | 0.3936 | 0.6506 | 0.2017 | 47.94% | 31.95% | 83.04% | none |
| Constant training prior | 0.2686 | 0.5000 | 0.1965 | 73.14% | — | 0.00% | constant |

The project-selected 70% development precision target was infeasible with minimum alert support for: Logistic Regression, XGBoost, PyTorch Embedding MLP. Their operating points use the declared development-F1 fallback.
<!-- END GENERATED RESULTS -->

AP means **Average Precision on raw scores**, not trapezoidal PR area. Brier loss uses the selected probability output, including identity calibration when that wins development selection. Lower Brier loss is better. Operational precision and recall use thresholds selected on the same probability scale as inference. The documented fallback maximizes development F1 with minimum alert support. The 70% target is a project configuration choice, not a faculty-specified requirement.

See [full metrics](results/metrics/evaluation_metrics.json), [feature ablation](results/metrics/feature_ablation.csv), [forward model selection and temporal folds](results/reports/temporal_search.json), [calibration selection](results/reports/calibration_selection.json), and [date/error slices](results/metrics/error_slices.csv).

## What makes the comparison valid

- Dates remain intact in chronological training/development/test partitions.
- Four training-only expanding date folds tune each candidate feature family's model parameters. A later, disjoint forward selection block chooses the final feature family and calibration method. Threshold selection and test have their own subsequent date blocks.
- Model fitting and every development role purge labels unavailable at their decision boundaries.
- A reconstructed arrival-time audit plus a declared 24-hour reporting lag controls when historical labels become visible. Daily local-midnight snapshots use strictly earlier release timestamps.
- Earlier development/test outcomes may enter later historical snapshots only after release. Estimator weights and decision policy remain fixed during test. Saved inference replays the same event database and timestamp filters.
- Imputation, standardization, one-hot vocabularies and embedding vocabularies fit on training only. Unseen categories have explicit fallback behavior.
- Scheduled-demand features count the entire timetable, including missing-outcome flights.
- Constant-prior comparison, unsuccessful ablations, raw/selected calibration curves and day-block confidence intervals accompany headline metrics.

The static timetable is assumed known at T-2h, and the reporting lag is an assumption because real publication timestamps are absent. See [methods and mathematical justification](docs/METHODS.md) and the [dataset card](docs/DATA_CARD.md).

## Repository map

```text
app/                    Historical Streamlit demonstrator
src/                    Data, time handling, features, models, evaluation and inference
scripts/                Download, reproduce, audit, verify and execute notebooks
notebooks/              Executed data audit, EDA and model-comparison walkthroughs
data/metadata/          Tracked airport timezone metadata
models/                 Version-matched models, preprocessors, policy and event store
results/figures/        EDA and evaluation plots
results/metrics/        Comparison, ablation, uncertainty and error metrics
results/predictions/    Full-precision development and test predictions
results/reports/        Audit, provenance, splits and sensitivity examples
docs/                   Method and data documentation, final report and slides
docs/report/            Final LaTeX report PDF only
docs/presentation/      Final 13-slide presentation PDF only
tests/                 Regression, artifact integration and app tests
```

`data/raw/flights.csv`, environments, caches and local logs are intentionally excluded. AI handoff bundles and duplicated result archives are not part of the submission. The only Git repository is this project folder.

## Tests and safeguards

The tests cover chronological partitions, schema/clock validation, forbidden inputs, availability cutoffs, past-only priors, daily replay, frozen references, timetable counts, score-scale consistency, model aliases, historical date guards, midnight rescheduling, duration preservation and app feedback. `verify_artifacts.py` independently reconstructs metrics and thresholds and compares saved-model inference with original test records.

The GitHub Actions workflow runs Python 3.12 formatting, lint and test checks on Linux and macOS, including OpenMP installation on macOS.

## Scope and interpretation

One year and three origins limit transfer. Weather, rotations, capacity, crew and connection constraints are absent. Schedule sensitivity is non-causal and does not demonstrate realized delay reduction.

## Final submission documents

- [Project report (PDF)](docs/report/T001_Report.pdf): eight pages, including the literature review, dataset and feature decisions, models, actual results and references.
- [Presentation (PDF)](docs/presentation/T001_Presentation.pdf): 13 slides covering the prediction task, implementation, evaluation, offline example and conclusion.

The report and presentation folders contain only their final PDFs. Editable document sources, preparation notes and build intermediates are kept outside this repository.

## Sources and license

Dataset: [Kaggle Flights](https://www.kaggle.com/datasets/matinsajadi/flights). Original schema and provenance: [nycflights13](https://nycflights13.tidyverse.org/reference/flights.html), sourced from BTS. Timezone provenance is recorded in the metadata file. Project code uses the existing [MIT license](LICENSE); dataset-provider terms apply separately.

## Team T001

| Member | Enrollment number |
|---|---|
| Sarthak Mishra | 2401010421 |
| Shane | 2401020062 |
| Shitanshu Tiwari | 2401010440 |
| Preetish Ubhrani | 2401020050 |
| Kshitiz | 2401010242 |
