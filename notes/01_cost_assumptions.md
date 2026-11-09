# Cost assumptions (for threshold selection)

> **Illustrative assumed costs, not bank data.** Every notebook, the README and the website carry this label.

Values live in `configs/cost.yaml`; code reads them from there (`fraud_e2e.cost`). Code refuses to choose thresholds with a parameter set whose status is not `accepted`/`confirmed`.

## 1. Two-tier policy (primary): approve / step-up — **status: accepted (2026-11-09, illustrative)**

A transaction whose score is at or above the threshold goes to **step-up** (analyst review or extra authentication such as OTP / 3-D Secure). Otherwise it is approved.

| Outcome | Meaning | Cost formula |
|---|---|---|
| TP | fraud, stepped up | `review_cost_per_alert` |
| FP | legit, stepped up | `review_cost_per_alert + fp_friction_cost` |
| FN | fraud, approved | `fn_loss_fraction × amount_usd + fn_fixed_cost` |
| TN | legit, approved | 0 |

| Parameter | Value | Reasoning | Sensitivity |
|---|---|---|---|
| `review_cost_per_alert` | 5 USD | a few minutes of analyst time or an OTP / 3DS round-trip | ×0.5, ×2 |
| `fp_friction_cost` | 10 USD | good customer interrupted: support contact, abandoned purchase, small churn risk | ×0.5, ×2 |
| `fn_loss_fraction` | 1.0 | issuer absorbs the full amount of a missed fraud; ×2 = indirect losses on top | ×0.5, ×2 |
| `fn_fixed_cost` | 15 USD | chargeback processing per missed fraud | ×0.5, ×2 |

## 2. Three-tier policy (supplementary): approve / step-up / decline — **status: accepted (2026-11-09, illustrative)**

Two thresholds: score ≥ `t_stepup` → step-up; score ≥ `t_decline` → decline outright.

**Why an extra parameter is needed.** In the two-tier model step-up stops every fraud. If that stays true, declining is never cheaper than step-up (it only adds cost for good customers), and the optimiser would never use the decline tier. Real step-up is not perfect: a fraudster who controls the victim's phone (SIM swap, phished OTP) passes it. So the three-tier model adds the share of fraud that still gets through step-up. To compare fairly, the two-tier policy is re-scored under the same three-tier assumptions.

| Outcome | Cost formula |
|---|---|
| approve, fraud | `fn_loss_fraction × amount + fn_fixed_cost` (as before) |
| step-up, legit | `review_cost_per_alert + fp_friction_cost` (as before) |
| step-up, fraud | `review_cost_per_alert + stepup_fraud_pass_rate × (fn_loss_fraction × amount + fn_fixed_cost)` |
| decline, legit | `decline_fp_cost` |
| decline, fraud | `decline_tp_cost` |

| Proposed parameter | Value | Reasoning |
|---|---|---|
| `stepup_fraud_pass_rate` | 0.2 | about 1 in 5 frauds still passes OTP / 3DS (compromised phone or phished code); a round illustrative number |
| `decline_fp_cost` | 50 USD | a hard decline hurts a good customer far more than a step-up: lost sale and interchange, support call, and a real chance the customer moves to another card. Set at roughly 3× the step-up cost of a good customer (15 USD) |
| `decline_tp_cost` | 0 USD | automatic block, no review, no loss |

What these imply (rough, before any data): for a typical fraud amount, declining beats step-up only when the model is quite sure, around a fraud probability above ~0.6. So the decline tier should be small and reserved for the highest scores.

Sensitivity planned: `stepup_fraud_pass_rate` and `decline_fp_cost` each ×0.5 and ×2.

## How the values are used (pre-registered, A1/A2)

- Thresholds are chosen on dev out-of-fold predictions only (cost curves averaged over 3 CV repeats).
- Sensitivity: each parameter ×0.5 and ×2, threshold re-optimised, regret of keeping the base threshold reported.
- `reports/p2_threshold_web.json` holds per-threshold TP/FP/FN/TN and missed-fraud USD so the website can recompute cost for any values.
- The holdout is scored once, after both policies' thresholds are fixed.
