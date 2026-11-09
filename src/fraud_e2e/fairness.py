"""Group metrics by age band at a fixed threshold, with stratified bootstrap CIs (pre-reg A5).

All four compared metrics depend only on counts, so one stratified bootstrap resample of a group
(fraud and legit rows resampled separately, with replacement) is exactly two binomial draws:
flagged fraud ~ Bin(n_fraud, recall) and flagged legit ~ Bin(n_legit, fpr).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_e2e import config as C

AGE_BINS = [17, 29, 44, 59, 200]
AGE_LABELS = ["18–29", "30–44", "45–59", "60+"]
METRICS = ["stepup_rate", "recall", "precision", "fpr"]
N_COMPARISONS = len(AGE_LABELS) * len(METRICS)
ALPHA = 0.05
ALPHA_ADJ = ALPHA / N_COMPARISONS  # Bonferroni
N_BOOT = 20_000


def age_group(age) -> pd.Series:
    return pd.cut(pd.Series(age), AGE_BINS, labels=AGE_LABELS).astype(str)


def _counts(y, flag) -> tuple[int, int, int, int]:
    y, flag = np.asarray(y), np.asarray(flag, bool)
    return (
        int((y == 1).sum()),
        int((y == 0).sum()),
        int((flag & (y == 1)).sum()),
        int((flag & (y == 0)).sum()),
    )


def _rates(n_pos, n_neg, tp, fp):
    """Metrics from counts; works on scalars or arrays of resampled counts."""
    tp, fp = np.asarray(tp, float), np.asarray(fp, float)
    flagged = tp + fp
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "stepup_rate": flagged / (n_pos + n_neg),
            "recall": tp / n_pos if n_pos else np.full_like(tp, np.nan),
            "precision": np.where(flagged > 0, tp / flagged, np.nan),
            "fpr": fp / n_neg if n_neg else np.full_like(fp, np.nan),
        }


def _draws(n_pos, n_neg, tp, fp, rng, n_boot):
    tp_b = rng.binomial(n_pos, tp / n_pos if n_pos else 0.0, n_boot)
    fp_b = rng.binomial(n_neg, fp / n_neg if n_neg else 0.0, n_boot)
    return _rates(n_pos, n_neg, tp_b, fp_b)


def _ci(x, alpha):
    x = np.asarray(x, float)
    return float(np.nanquantile(x, alpha / 2)), float(np.nanquantile(x, 1 - alpha / 2))


def overall_rates(y, score, t: float) -> dict[str, float]:
    """The four metrics on all rows together (point values)."""
    return {k: float(v) for k, v in _rates(*_counts(y, np.asarray(score) >= t)).items()}


def group_metrics(
    y, score, groups, t: float, n_boot: int = N_BOOT, seed: int = C.SEED
) -> pd.DataFrame:
    """One row per group: counts, each metric with its 95% CI, and (descriptive) PR-AUC."""
    rng = np.random.default_rng(seed)
    y, score, groups = np.asarray(y), np.asarray(score, float), np.asarray(groups)
    rows = []
    for g in AGE_LABELS:
        m = groups == g
        n_pos, n_neg, tp, fp = _counts(y[m], score[m] >= t)
        point = _rates(n_pos, n_neg, tp, fp)
        boot = _draws(n_pos, n_neg, tp, fp, rng, n_boot)
        row = {
            "group": g,
            "n": int(m.sum()),
            "n_fraud": n_pos,
            "n_flagged": tp + fp,
            "tp": tp,
            "fp": fp,
        }
        for k in METRICS:
            lo, hi = _ci(boot[k], ALPHA)
            row |= {k: float(point[k]), f"{k}_ci_low": lo, f"{k}_ci_high": hi}
        row["pr_auc"] = (
            float(average_precision_score(y[m], score[m])) if 0 < n_pos < m.sum() else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def group_gaps(
    y, score, groups, t: float, n_boot: int = N_BOOT, seed: int = C.SEED
) -> pd.DataFrame:
    """Group minus all other rows, per metric: 95% CI, Bonferroni CI and the A5 verdict."""
    rng = np.random.default_rng(seed + 1)
    y, score, groups = np.asarray(y), np.asarray(score, float), np.asarray(groups)
    flag = score >= t
    rows = []
    for g in AGE_LABELS:
        m = groups == g
        a = _counts(y[m], flag[m])
        b = _counts(y[~m], flag[~m])
        pa, pb = _rates(*a), _rates(*b)
        da, db = _draws(*a, rng, n_boot), _draws(*b, rng, n_boot)
        for k in METRICS:
            diff = da[k] - db[k]
            lo, hi = _ci(diff, ALPHA)
            alo, ahi = _ci(diff, ALPHA_ADJ)
            real = alo > 0 or ahi < 0
            rows.append(
                {
                    "group": g,
                    "metric": k,
                    "group_value": float(pa[k]),
                    "rest_value": float(pb[k]),
                    "diff": float(pa[k] - pb[k]),
                    "ci_low": lo,
                    "ci_high": hi,
                    "ci_adj_low": alo,
                    "ci_adj_high": ahi,
                    "verdict": ("higher" if alo > 0 else "lower")
                    if real
                    else "not distinguishable",
                }
            )
    return pd.DataFrame(rows)
