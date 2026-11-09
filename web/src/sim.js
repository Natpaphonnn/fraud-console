// Threshold simulator: pure functions, no DOM. Mirrors fraud_e2e.cost / fraud_e2e.threshold
// (two-tier policy: approve below the threshold, step up at or above it).
//
// `d` is the `sim` block of the page data (built from reports/p2_threshold_web.json):
//   threshold[i], tp[i], fp[i], fn[i], tn[i], fn_amount_usd[i]  (counts = mean over CV repeats)
//   n_rows, n_fraud, fraud_amount_usd
// `p` holds the four cost parameters (USD, illustrative).

export const PARAM_KEYS = [
  "review_cost_per_alert",
  "fp_friction_cost",
  "fn_loss_fraction",
  "fn_fixed_cost",
];

// Same bounds as fraud_e2e.cost.CostParams.
export function validateParams(p) {
  const errors = [];
  for (const k of PARAM_KEYS) {
    const v = p[k];
    if (typeof v !== "number" || !Number.isFinite(v)) errors.push(`${k} must be a number`);
    else if (v < 0) errors.push(`${k} must be ≥ 0`);
  }
  if (Number.isFinite(p.fn_loss_fraction) && p.fn_loss_fraction > 2) {
    errors.push("fn_loss_fraction must be ≤ 2");
  }
  return errors;
}

export function costAt(d, i, p) {
  return (
    (d.tp[i] + d.fp[i]) * p.review_cost_per_alert +
    d.fp[i] * p.fp_friction_cost +
    p.fn_loss_fraction * d.fn_amount_usd[i] +
    d.fn[i] * p.fn_fixed_cost
  );
}

// Cost of approving everything (no step-ups): every fraud is missed.
export function nothingCost(d, p) {
  return p.fn_loss_fraction * d.fraud_amount_usd + d.n_fraud * p.fn_fixed_cost;
}

export function metricsAt(d, i, p) {
  const tp = d.tp[i];
  const fp = d.fp[i];
  const fn = d.fn[i];
  const flagged = tp + fp;
  const cost = costAt(d, i, p);
  const nothing = nothingCost(d, p);
  const per1000 = 1000 / d.n_rows;
  return {
    index: i,
    threshold: d.threshold[i],
    recall: tp / (tp + fn),
    precision: flagged > 0 ? tp / flagged : NaN,
    alertRate: flagged / d.n_rows,
    flaggedPer1000: flagged * per1000,
    fraudCaughtPer1000: tp * per1000,
    fraudMissedPer1000: fn * per1000,
    missedUsdPer1000: d.fn_amount_usd[i] * per1000,
    costPer1000: cost * per1000,
    nothingPer1000: nothing * per1000,
    saving: nothing > 0 ? 1 - cost / nothing : NaN,
  };
}

export function costCurve(d, p) {
  return d.threshold.map((_, i) => costAt(d, i, p));
}

// First index of the minimum, like numpy.argmin.
export function bestIndex(d, p) {
  let best = 0;
  let bestCost = Infinity;
  for (let i = 0; i < d.threshold.length; i++) {
    const c = costAt(d, i, p);
    if (c < bestCost) {
      bestCost = c;
      best = i;
    }
  }
  return best;
}

export function nearestIndex(d, t) {
  let best = 0;
  for (let i = 1; i < d.threshold.length; i++) {
    if (Math.abs(d.threshold[i] - t) < Math.abs(d.threshold[best] - t)) best = i;
  }
  return best;
}

// Outcomes of a set of scored rows at threshold t (two-tier policy, same cost formula).
// Used for the live stream: the full 2,000-row sample and the rows played so far.
export function sampleOutcomes(scores, y, amount, t, p) {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  let fnUsd = 0;
  for (let i = 0; i < scores.length; i++) {
    const step = scores[i] >= t;
    if (y[i] === 1) {
      if (step) tp++;
      else {
        fn++;
        fnUsd += amount[i];
      }
    } else if (step) fp++;
    else tn++;
  }
  const n = scores.length;
  const flagged = tp + fp;
  const cost =
    flagged * p.review_cost_per_alert +
    fp * p.fp_friction_cost +
    p.fn_loss_fraction * fnUsd +
    fn * p.fn_fixed_cost;
  return {
    n,
    tp,
    fp,
    fn,
    tn,
    recall: tp + fn ? tp / (tp + fn) : NaN,
    precision: flagged ? tp / flagged : NaN,
    alertRate: n ? flagged / n : NaN,
    costPer1000: n ? (cost / n) * 1000 : NaN,
  };
}
