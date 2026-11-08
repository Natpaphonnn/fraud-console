# Cost assumptions (for threshold selection)

**Status: PROPOSED — waiting for Nat's confirmation.** Values live in `configs/cost.yaml`; code reads them from there (`fraud_e2e.cost.load_cost_params`). Nothing in P2 may use them for a decision until `status: confirmed`.

## Decision being priced

For every transaction the model outputs a score. If the score is above a threshold, the transaction is **flagged** (sent to review or step-up authentication). Otherwise it goes through.

| Outcome | Meaning | Cost formula |
|---|---|---|
| TP | fraud, flagged | `review_cost_per_alert` |
| FP | legit, flagged | `review_cost_per_alert + fp_friction_cost` |
| FN | fraud, missed | `fn_loss_fraction × amount_usd + fn_fixed_cost` |
| TN | legit, passed | 0 |

The cost of a missed fraud depends on its amount, so the threshold choice is amount-aware.

## Proposed values

These are round, illustrative numbers chosen for a portfolio project. They are **not** sourced from a bank and will be labelled as assumptions wherever they appear.

| Parameter | Proposed | Reasoning | Sensitivity range |
|---|---|---|---|
| `review_cost_per_alert` | 5 USD | a few minutes of analyst time or an OTP/step-up round-trip | 1 – 20 |
| `fp_friction_cost` | 10 USD | annoyed good customer: support contact, abandoned purchase, small churn risk | 0 – 50 |
| `fn_loss_fraction` | 1.0 | issuer absorbs the full amount of a missed fraud | 0.5 – 1.0 |
| `fn_fixed_cost` | 15 USD | chargeback processing per missed fraud | 0 – 30 |

## Questions for Nat

1. Is "flag = review / step-up" the right framing, or should a flag mean "decline outright"? (Decline would raise the FP friction cost.)
2. Are the four values acceptable as stated assumptions, or do you want different numbers?
3. Is the sensitivity range wide enough for the article?

## How they will be used (P2, pre-registered)

- Pick the threshold that minimises total cost on dev out-of-fold predictions.
- Report the same threshold's cost on the holdout, with a bootstrap CI.
- Sweep each parameter over its range and show how the best threshold and the saving vs "flag nothing" change.
