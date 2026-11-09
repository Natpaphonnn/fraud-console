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

### A1 — 2026-11-09 — Nat's decisions before P2 (no model has been fit yet)

1. **Holdout confirmed** (20%, stratified, seed 42, file above). It is not touched until the model **and** all thresholds are selected on CV. It is then scored **once**, and every holdout metric is reported with a stratified bootstrap 95% CI (2,000 resamples), however wide.
2. **Cost parameters accepted as illustrative** (`configs/cost.yaml`, `status: accepted`). Every notebook, the README and the website label them as *assumed values, not bank data*.
   - Sensitivity analysis: each parameter multiplied by ×0.5 and ×2 with the others at base. For each case report the cost-optimal threshold, expected cost per 1,000 transactions, recall, precision and alert rate, plus the cost of keeping the base threshold (regret). `fn_loss_fraction` ×2 = 2.0 is read as "a missed fraud costs twice its amount" (indirect losses), so its allowed range is widened to [0, 2].
   - Export a JSON with, per threshold, TP / FP / FN / TN counts and the USD amount of missed fraud, so the P5 website can recompute cost for any parameter values without a backend (cost is linear in the parameters).
3. **Policy.** Primary policy is two-tier: *approve* / *step-up* (review or extra authentication such as OTP/3DS). Supplementary three-tier policy: *approve* / *step-up* / *decline*.
   - Decline-specific cost parameters are **proposed in `notes/01_cost_assumptions.md` and must be approved by Nat before they are used**.
   - Both thresholds of the three-tier policy are chosen by expected cost on CV out-of-fold predictions only, then applied once to the holdout together with the two-tier threshold.
4. Raw CSV committed with `data/raw/SOURCE.md` and its sha256 (already done in P0).
5. `is_ai_generated_scam_attempt`: the final pipeline is also evaluated without it (paired comparison on the same 15 folds), as planned in §5.

### A2 — 2026-11-09 — implementation details fixed before any P2 result

- **Encoding.** One-hot for the 5 categoricals (all models); booleans as 0/1. Logistic regression additionally gets `log1p` on `amount_usd` and `account_balance_usd` (heavy right tails seen in P1) and standardisation of all numeric inputs, L2 penalty, `C = 1.0`. Trees get raw numerics.
- **Default boosting settings** (before tuning): 300 trees, learning rate 0.05, depth/leaves limited (LightGBM `num_leaves = 15`, XGBoost `max_depth = 4`), row and column subsampling 0.8, `min_child_samples`/`min_child_weight` ≈ 20 rows. No early stopping (it would need an extra inner split of ~43 fraud).
- **Order of comparisons** (each uses the §3 rule; the simpler option wins ties):
  1. Logistic: unweighted vs `class_weight="balanced"`.
  2. LightGBM vs XGBoost (both unweighted). The winner is "the boosting model".
  3. Imbalance handling on the boosting model: none / class weight (`scale_pos_weight = n_neg / n_pos`) / random undersampling of legit to 10:1 / SMOTENC (categorical-aware SMOTE) to a minority ratio of 0.1, `k = 5`. All inside the training folds.
  4. Tuning: 20 random configurations of the boosting model scored on repeat `r0` only; the best is then compared with the untuned version on all 15 folds. Tuned replaces untuned only if it passes the §3 rule.
  5. Best boosting pipeline vs best logistic pipeline. Boosting becomes the final model only if it passes the §3 rule; otherwise the logistic model is final.
  6. Ablation without `is_ai_generated_scam_attempt` on the final pipeline (reported, does not change the choice unless the leak alarm in §5 fires).
- **Calibration** of the final pipeline: none vs sigmoid vs isotonic, each fit by inner 5-fold `CalibratedClassifierCV` inside every outer training fold. Lowest mean Brier score wins; ties (CI of the paired difference includes 0) go to "none" then "sigmoid".
- **Threshold.** For each repeat the full dev OOF prediction set gives a cost-vs-threshold curve; the three curves are averaged and the minimum is the chosen threshold. Spread across repeats is reported.
- **Explainability.** TreeSHAP values from the booster's native `pred_contrib` (exact, no extra dependency), on the final model refit on all dev rows; one-hot columns are summed back to their source feature. SHAP explains the *uncalibrated* log-odds; this is stated next to every SHAP plot. If logistic regression is the final model, exact linear SHAP is used instead: coefficient × (standardised value − its dev mean).

### A3 — 2026-11-09 — before the holdout is scored

1. **Three-tier values accepted** by Nat as illustrative (`stepup_fraud_pass_rate` 0.2, `decline_fp_cost` 50 USD, `decline_tp_cost` 0 USD), with ×0.5 / ×2 sensitivity on the first two. The two-tier policy is also re-scored under the three-tier cost model for a like-for-like comparison.
2. **Threshold grid refined** (implementation of the A2 argmin, not a change of rule). The first run used a ~680-point grid; a check on a 9,000-point grid showed the coarse grid could miss the minimum when two separate regions have almost the same cost. The 1-D optimisation now uses a ~5,000-point grid; the 2-D three-tier search and the website export use the ~680-point grid. The range of thresholds within 1% of the minimum cost and the per-repeat optima are reported alongside the chosen threshold. All of this was done on dev OOF only; the holdout was still unseen.

### A4 — 2026-11-09 — after the holdout was scored (disclosure, not a change)

The holdout was scored once with the frozen choices in `notes/holdout_lock.json`. Afterwards, two **descriptive** metrics were added and are labelled post-hoc wherever shown: (a) the two-tier saving relative to "step-up nobody", because absolute cost per 1,000 is dominated by the few fraud amounts in a sample; (b) a like-for-like comparison of three-tier vs two-tier under the three-tier cost model (the two-tier threshold for that model, 0.1166, had been chosen on dev OOF before the holdout). Both reuse the saved holdout scores (`reports/p2_holdout_scores.csv.gz`); no model was refit and no threshold or choice was changed.

### A5 — 2026-11-09 — P5 fairness by age, written before any age-group result

Question: does the final pipeline (logistic regression, step-up threshold 0.0695) treat age groups
differently, and does it need `customer_age` at all? Model choice and threshold are **not** revisited
(the holdout has been used); P5 is descriptive.

- **Groups:** `customer_age` 18–29, 30–44, 45–59, 60+ (conventional bands). Fixed after a label-free look
  at the dev age distribution only (sizes 3,000 / 3,652 / 3,808 / 5,540 rows; ages 18–81).
- **Data:** dev out-of-fold scores of the final pipeline from CV repeat r0 (every dev row scored once, by a
  model that did not train on it). The holdout is **not** used. Repeats r1 and r2 are reported as point
  estimates only, as a stability check.
- **Threshold:** the project threshold 0.0695 for every group (no per-group thresholds).
- **Metrics per group:** rows, fraud cases, step-up rate, recall (TPR), precision, false-positive rate (good
  customers stepped up / all good customers). PR-AUC per group is descriptive only (base rates differ).
- **Uncertainty:** stratified bootstrap within each group (fraud and legit resampled separately), 2,000
  resamples, seed 42, percentile intervals.
- **Rule for "differs":** each group is compared with all other dev rows (group − rest), bootstrapping
  both sides independently. 4 groups × 4 metrics (step-up rate, recall, precision, FPR) = 16 comparisons,
  so a gap is called **real only if its Bonferroni-adjusted interval (1 − 0.05/16 = 99.69%) excludes 0**.
  Otherwise it is reported as **"not distinguishable"**. A non-significant gap is never reported as
  "fair" or "no difference", and the interval width is shown.
- **Model without `customer_age`:** the final spec with `drop = ["customer_age"]` on the same 15 folds,
  corrected resampled t-test on PR-AUC (§3 rule). Its per-group step-up rate and FPR at 0.0695 are
  reported descriptively.
- **Not claimed:** no causal or legal fairness claim. Age is the only person attribute in the data, and
  the data's provenance is unspecified.

### A6 — 2026-11-09 — P5 interaction / GAM check (optional), written before any result

Does the logistic model miss structure? Two extensions of the final pipeline, each compared with it on the
same 15 folds by the §3 rule on PR-AUC (the simpler model wins unless the 95% CI of the paired difference
excludes 0):

1. **GAM-like:** every numeric input expanded with cubic B-splines (5 quantile knots; after `log1p` for the
   two heavy-tailed columns), L2, `C = 1`.
2. **Red-flag interactions:** the 15 pairwise products of the 6 boolean flags added to the final inputs.

The final model is not changed by the outcome (the holdout has been used). A better extension is reported
as a lead for future work; a non-better one goes under "what didn't work".
