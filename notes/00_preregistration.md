# Pre-registration: evaluation protocol

**Written:** 2026-11-09, after structural data checks and **before** any label-based EDA or any model was fit.
**Status:** locked. Changes are allowed only as dated amendments at the bottom of this file, with the reason, and must be made before the result they would affect is seen.

## 1. Data split (locked)

- Source: `data/raw/credit_card_fraud_2026.csv` (sha256 `4e32b1ad…3da61`), 20,000 rows, 339 fraud.
- `scripts/make_splits.py` (seed 42) → `data/splits/splits.csv`
  (sha256 `5d6a7fadf723b6bd45c1c6882257bd3838face366169c7264e37c0cf4595f23e`).
- **Holdout:** 20% stratified (4,000 rows, 68 fraud). Not used for EDA, feature decisions, tuning, calibration or threshold choice. Scored **once** with the final pipeline at the end of P2; that result is reported whatever it is.
- **Dev:** remaining 80% (16,000 rows, 271 fraud). Repeated stratified 5-fold CV, 3 repeats (`fold_r0..fold_r2`, seeds 42/43/44) → 15 train/validation pairs, about 54 fraud per validation fold.
- Label-based EDA (fraud rate by feature, univariate AUC, etc.) uses **dev only**. Structural checks (schema, ranges, duplicates) may use all rows because they do not use the label.

Why a holdout despite only 339 fraud: P2 compares several models, imbalance strategies, calibrators and thresholds on the same CV folds. That selection makes CV scores optimistic. A 68-fraud holdout is noisy, but it is the only number not touched by selection. Its CI will be wide and will be reported as such.

## 2. Metrics

| Role | Metric | Notes |
|---|---|---|
| **Primary** | PR-AUC (average precision) | Computed per validation fold. |
| Secondary | ROC-AUC | Per fold. |
| Probability quality | Brier score, log loss, reliability curve | On out-of-fold (OOF) predictions. |
| Business | Expected cost (USD) at the chosen threshold, also per 1,000 transactions | Uses `configs/cost.yaml` only after `status: confirmed`. |
| Operating point | Precision, recall, alert rate at the chosen threshold | |

Reporting: mean and standard deviation across the 15 folds, plus a 95% interval. A PR-AUC "no-skill" reference equals the fraud rate of the fold and is shown next to every PR-AUC.

## 3. Comparing two pipelines

- Pipelines are compared on the **same 15 folds** (paired).
- Statistic: per-fold difference in PR-AUC, mean and 95% CI using the **corrected resampled t-test** (Nadeau & Bengio, 2003; variance inflated by `1/k + n_test/n_train` for repeated k-fold).
- **Decision rule:** a more complex pipeline is preferred over a simpler one only if the 95% CI of the difference is entirely above 0. Otherwise the simpler one is kept and the difference is reported as "not distinguishable". Simplicity order: logistic regression < LightGBM < XGBoost; no resampling < class weights < undersampling < SMOTE.
- Small differences are not interpreted. With ~54 fraud per fold, differences inside the interval are noise.

## 4. Planned experiments (P2)

1. **Baseline:** logistic regression (one-hot categoricals, standardised numerics, fit inside each fold), with and without `class_weight="balanced"`.
2. **Gradient boosting:** LightGBM, then XGBoost, default-ish parameters first.
3. **Imbalance handling on one fixed model** (the better boosting model from step 2, same hyper-parameters): none / class weight / random undersampling / SMOTE. Resampling is applied **to the training part of each fold only**, never to validation or holdout rows.
4. **Tuning:** a small fixed budget (≤ 30 configurations per model family), logged in `notes/experiment_log.md`. CV scores after tuning are labelled as optimistically biased; the holdout is the unbiased check.
5. **Calibration:** none vs sigmoid (Platt) vs isotonic, fitted with cross-fitting inside the training folds; chosen by Brier score on OOF predictions. Required if resampling is used, because resampling shifts predicted probabilities.
6. **Threshold:** chosen on dev OOF predictions by minimising total expected cost under the confirmed cost parameters. Reported with a sensitivity analysis over the cost parameters. Then applied unchanged to the holdout.
7. **Explainability:** SHAP on the final model (global importance + a few individual transactions).

## 5. Leakage controls

- `transaction_id` is never a feature.
- All preprocessing (encoders, scalers, resamplers, calibrators) is fit inside the training part of each fold.
- **Leak alarm:** any single feature with univariate dev ROC-AUC > 0.95, or any model with CV PR-AUC > 0.95, stops work until the cause is understood and written up.
- **Post-outcome features:** a column whose meaning implies knowledge of the outcome (for example a label assigned after an investigation) is a leakage candidate. Before modelling, each such column gets a written decision in `notes/` (keep / drop / report both). If kept, a model without it is also reported.

## 6. What this dataset cannot support (stated up front)

- No customer or card identifier → no customer-level features and no grouped split by customer. Rows are treated as independent transactions; if the same customer appears in train and validation, CV is optimistic in a way we cannot measure.
- No real timestamp (only hour-of-day and day-of-week) → no out-of-time validation; drift in P4 is simulated, not observed.
- Provenance is unspecified (real vs synthetic) → results describe this dataset only, not real-world card fraud performance.

## Amendments

_None yet._
