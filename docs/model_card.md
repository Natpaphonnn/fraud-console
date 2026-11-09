# Model card: Fraud-E2E step-up scoring model

| | |
|---|---|
| **Model** | logistic regression (L2, C = 1; one-hot categoricals, log1p on amount and balance, standardised numerics) — pipeline spec `9d9523a408a3` |
| **Task** | Score card transactions for fraud; scores above a threshold are sent to **step-up** (review or extra authentication), the rest are approved |
| **Owner** | Natpaphon Nukhunthod (portfolio project) |
| **Status** | Research / portfolio artifact. **Not deployed, not validated on production data.** |
| **Code** | `src/fraud_e2e/`, notebooks `03`–`06`, protocol `notes/00_preregistration.md` |

## Intended use

- Demonstrating an honest, cost-aware evaluation workflow for rare-event classification: locked splits, pre-registered comparisons, uncertainty on every metric, threshold chosen by expected cost, one-time holdout.
- Teaching and discussion material (portfolio, article, interview).

## Out of scope — do not use this model to

- Approve, decline or step up **real** card transactions, or make any decision about a real person.
- Estimate fraud losses or savings for a real institution: the costs are assumptions (see below) and the data provenance is unknown.
- Claim performance on other datasets, other regions, or other time periods. Nothing here has been tested out of time or out of distribution.
- Infer causes of fraud from the SHAP explanations; they describe what the model learned from this dataset only.

## Data

- *Credit Card Fraud Detection 2026* by Udit Jain (Kaggle), **CC0**. A **public CC0 dataset of unspecified provenance**: the data card does not say whether rows are real or synthetic.
- 20,000 transactions, 339 fraud (1.70%, 95% CI [1.53%, 1.88%]); 24 features (categorical, boolean, counts, continuous). `transaction_id` is never used as a feature.
- Split (locked before any label-based analysis): dev 16,000 rows / 271 fraud for 5-fold CV × 3 repeats; holdout 4,000 rows / 68 fraud, scored once.
- Data checks: 0 of 26 structural checks failed. Channel and device type look statistically independent (p = 0.44), and combinations unusual for real card traffic are common (largest: `card_present_channel_on_desktop`, 8.4% of rows). This is consistent with synthetic generation; provenance stays "unspecified".

## Performance

| Metric | CV on dev (15 folds, mean [95% CI]) | Holdout, once (68 fraud; bootstrap 95% CI) |
|---|---|---|
| PR-AUC (primary) | 0.415 [0.341, 0.490] | 0.468 [0.361, 0.585] |
| ROC-AUC | 0.942 [0.928, 0.957] | 0.950 [0.923, 0.972] |
| No-skill PR-AUC (= fraud rate) | 0.0169 | 0.0170 |
| Recall at threshold 0.0695 | 69.5% | 70.6% [60.3%, 80.9%] |
| Precision at threshold | 23.5% | 23.8% [20.0%, 27.8%] |
| Share of transactions stepped up | 5.01% | 5.05% [4.45%, 5.67%] |
| Cost saving vs stepping up nobody* | 47.6% | 37.0% [13.2%, 63.4%] (post-hoc, descriptive) |

\* Illustrative assumed costs (not bank data).

Fold-to-fold standard deviation of PR-AUC is 0.062, larger than most differences between pipelines; that is why pipelines are compared pairwise on the same folds (corrected resampled t-test) rather than by eyeballing their means.

## Decision threshold and costs

The threshold 0.0695 minimises expected cost on dev out-of-fold predictions (best threshold in each of the 3 CV repeats: 0.0673–0.0834). It was fixed before the holdout was scored.

**Illustrative assumed costs (not bank data).** Accepted for this project as round, illustrative values:

| Outcome | Assumed cost |
|---|---|
| Any step-up (review / OTP) | 5 USD |
| Good customer stepped up (extra friction) | +10 USD |
| Missed fraud | 1 × amount + 15 USD |

Each value was scaled ×0.5 and ×2: the cost-optimal threshold moved between 0.0292 and 0.1334; the largest extra cost from keeping the base threshold was 232 USD per 1,000 transactions (fn_loss_fraction ×2). **Re-derive the threshold with your own costs** rather than reusing this one.

A supplementary three-tier policy (approve / step-up ≥ 0.1166 / decline ≥ 0.4299; assumes 20% of fraud passes step-up and 50 USD per wrongly declined customer) was cheaper than two-tier on CV (-104 USD per 1,000) but **not distinguishable on the holdout** (-21 USD per 1,000, 95% CI [-103, +60]).

## Explainability

Exact linear SHAP values (coefficient × centred input). The top three features, `merchant_risk_score`, `velocity_score`, `cvv_retry_count`, carry 37% of mean |SHAP|. Example: the largest missed fraud in CV (3,236.34 USD) had one strong risk signal but lacked the other red flags. SHAP explains the model's log-odds, not causes of fraud.

## Model selection (pre-registered; simpler option kept unless the 95% CI of the paired difference excludes 0)

- XGBoost vs LightGBM: -0.004 [-0.035, +0.027] → not distinguishable.
- Imbalance handling vs none (LightGBM): class weight -0.023 [-0.058, +0.013]; undersampling -0.005 [-0.036, +0.026]; SMOTENC -0.177 [-0.229, -0.125].
- Tuning (20 trials): +0.050 [+0.014, +0.086].
- Best boosting (`lgbm_tuned`) minus logistic: -0.048 [-0.084, -0.012] → logistic kept.
- `is_ai_generated_scam_attempt` (with minus without): +0.028 [+0.011, +0.045] → kept.
- Calibration: neither sigmoid nor isotonic calibration improved the Brier score.

## Limitations and risks

- **Unknown provenance.** Patterns may be artefacts of how the data was generated (for example, no fraud below 5.14 USD although 1.9% of legitimate transactions are smaller; independent channel and device).
- **No customer or card ID.** No customer-level features; one customer may appear in train and validation, so CV can be optimistic by an unknown amount.
- **No real timestamps.** No out-of-time validation; temporal drift is untested.
- **Small positive class.** 339 fraud cases in total, 68 in the holdout; holdout cost intervals are wide.
- **Assumed costs.** All cost numbers are illustrative; real costs differ by institution and change the threshold.
- **Fairness not assessed.** `customer_age` is an input; no protected-attribute analysis was done. Any real use would need one.
- **Probabilities.** Sigmoid and isotonic calibration did not improve the Brier score on CV (Brier 0.013); calibration under drift was not checked.

## Reproduce

`uv sync && make notebooks && make docs` regenerates every number above from the locked split and seed 42. This card is generated by `scripts/render_docs.py` from `reports/*.json`.
