# Fraud-E2E: cost-aware credit card fraud detection, end to end

A portfolio project that takes a small, highly imbalanced fraud dataset from first look to a deployable, monitored model, with honest uncertainty at every step.

> **Status:** P0 (setup) and P1 (data understanding + EDA) done. Modelling (P2) has not started; the evaluation protocol is pre-registered in [`notes/00_preregistration.md`](notes/00_preregistration.md).

## Data

- *Credit Card Fraud Detection 2026* by Udit Jain, [Kaggle](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026), licence **CC0 (public domain)**.
- A **public CC0 dataset of unspecified provenance**: the data card does not say whether rows are real or synthetic, so no result here is a claim about real-world card fraud.
- Size, fraud rate and quality-check results: [`reports/p1_data_understanding.json`](reports/p1_data_understanding.json).

**Known limitations**

- No customer or card ID → no customer-level features, no grouped split.
- No real timestamp (hour-of-day and day-of-week only) → no out-of-time validation; drift is simulated.
- Few fraud cases → every metric is reported with an interval.

## Approach

| Phase | What | Where |
|---|---|---|
| P0 | Repo, environment, locked data split | `scripts/make_splits.py`, `data/splits/` |
| P1 | Data quality, EDA (dev split only), business framing, cost model | [`01_data_understanding`](notebooks/01_data_understanding.ipynb), [`02_eda`](notebooks/02_eda.ipynb) |
| P2 | Logistic baseline → LightGBM / XGBoost, imbalance handling, calibration, cost-based threshold, SHAP | _planned_ |
| P3 | Package, MLflow, tests, FastAPI, Docker, CI | _planned_ |
| P4 | Simulated drift, PSI, model card | _planned_ |
| P5 | Static website with a threshold simulator | _planned_ |

**Evaluation discipline**

- 20% stratified holdout, locked before any label-based analysis and scored once at the end.
- Repeated stratified 5-fold CV (3 repeats) on the remaining 80%.
- PR-AUC is the primary metric; comparisons use paired per-fold differences with a corrected resampled t-test.
- Every number in the docs comes from a committed artifact in `reports/`.

## Reproduce

```bash
uv sync
make check       # lint + tests (including split/leakage guards)
make notebooks   # re-execute notebooks; artifacts land in reports/
make kaggle      # build self-contained Kaggle editions in kaggle/
```

## Repository layout

```
src/fraud_e2e/   library code (data, splits, stats, cost, plots, reporting)
notebooks/       executed notebooks (sources in notebooks/py/)
kaggle/          Kaggle editions of the notebooks
reports/         JSON/CSV results and figures
notes/           pre-registration, cost assumptions, experiment log
configs/         cost assumptions (configs/cost.yaml)
tests/           pytest suite
```

## Credits

Dataset by Udit Jain, released under CC0 on Kaggle.
