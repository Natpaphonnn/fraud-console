"""Small statistics helpers for a rare-event target.

With only a few hundred positives, every rate and score here is reported with
an interval. Intervals are Wilson (for proportions) or stratified bootstrap
percentile intervals (for AUC), both seeded for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as ss

from fraud_e2e import config as C


def wilson_ci(k: np.ndarray | int, n: np.ndarray | int, alpha: float = 0.05):
    """Wilson score interval for k successes out of n. Vectorised."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    z = ss.norm.ppf(1 - alpha / 2)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo = np.where(k == 0, 0.0, np.clip(centre - half, 0, 1))  # exact bounds at the edges
    hi = np.where(k == n, 1.0, np.clip(centre + half, 0, 1))
    return lo, hi


def rate_table(df: pd.DataFrame, col: str, target: str = C.TARGET) -> pd.DataFrame:
    """Fraud rate per level of `col` with 95% Wilson CI and lift vs the overall rate."""
    g = df.groupby(col, observed=True)[target].agg(n="size", n_fraud="sum").reset_index()
    g["fraud_rate"] = g["n_fraud"] / g["n"]
    g["ci_low"], g["ci_high"] = wilson_ci(g["n_fraud"].to_numpy(), g["n"].to_numpy())
    base = df[target].mean()
    g["lift"] = g["fraud_rate"] / base
    g.insert(0, "feature", col)
    g = g.rename(columns={col: "level"})
    g["level"] = g["level"].astype(str)
    return g.sort_values("fraud_rate", ascending=False).reset_index(drop=True)


def chi2_association(df: pd.DataFrame, col: str, target: str = C.TARGET) -> dict:
    """Chi-square test of independence + Cramér's V between a discrete column and the target."""
    table = pd.crosstab(df[col], df[target])
    chi2, p, dof, _ = ss.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {
        "feature": col,
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": float(v),
        "n_levels": int(table.shape[0]),
    }


def rank_auc(x: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC of score x for binary labels y via the Mann-Whitney U statistic (ties averaged)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    ranks = ss.rankdata(x)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = C.SEED, alpha: float = 0.05
) -> tuple[float, float]:
    """Stratified bootstrap percentile CI for rank_auc.

    Positives and negatives are resampled separately so every replicate keeps the class counts.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(bool)
    pos, neg = x[y], x[~y]
    lab = np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        xb = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        aucs[b] = rank_auc(xb, lab)
    return float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))


def univariate_numeric(
    df: pd.DataFrame,
    cols: list[str],
    target: str = C.TARGET,
    n_boot: int = 1000,
    seed: int = C.SEED,
) -> pd.DataFrame:
    """Per numeric column: class medians/IQR and univariate ROC AUC with bootstrap CI.

    AUC > 0.5 means larger values are associated with fraud; AUC < 0.5 the opposite.
    `separation` = |AUC - 0.5| ranks features regardless of direction.
    """
    y = df[target].to_numpy()
    rows = []
    for i, c in enumerate(cols):
        x = df[c].to_numpy(dtype=float)
        auc = rank_auc(x, y)
        lo, hi = bootstrap_auc_ci(x, y, n_boot=n_boot, seed=seed + i)
        q = lambda a, p: float(np.quantile(a, p))  # noqa: E731
        xp, xn = x[y == 1], x[y == 0]
        rows.append(
            {
                "feature": c,
                "median_fraud": q(xp, 0.5),
                "iqr_fraud": [q(xp, 0.25), q(xp, 0.75)],
                "median_legit": q(xn, 0.5),
                "iqr_legit": [q(xn, 0.25), q(xn, 0.75)],
                "auc": auc,
                "auc_ci_low": lo,
                "auc_ci_high": hi,
                "separation": abs(auc - 0.5),
            }
        )
    return pd.DataFrame(rows).sort_values("separation", ascending=False).reset_index(drop=True)


def binned_rate(
    df: pd.DataFrame, col: str, n_bins: int = 10, target: str = C.TARGET
) -> pd.DataFrame:
    """Fraud rate by quantile bin of a numeric column, with Wilson CI.

    Low-cardinality integer columns (<= 15 distinct values) are grouped by exact value
    instead, with the top 1% folded into the 99th-percentile value so no bin is tiny.
    """
    if df[col].nunique() <= 15:
        cap = df[col].quantile(0.99)
        x = df[col].clip(upper=cap)
        g = df.groupby(x)[target].agg(n="size", n_fraud="sum").reset_index()
        g["bin_mid"] = g["bin_left"] = g["bin_right"] = g[col].astype(float)
    else:
        bins = pd.qcut(df[col], q=n_bins, duplicates="drop")
        g = df.groupby(bins, observed=True)[target].agg(n="size", n_fraud="sum").reset_index()
        g["bin_mid"] = [iv.mid for iv in g[col]]
        g["bin_left"] = [iv.left for iv in g[col]]
        g["bin_right"] = [iv.right for iv in g[col]]
    g["fraud_rate"] = g["n_fraud"] / g["n"]
    g["ci_low"], g["ci_high"] = wilson_ci(g["n_fraud"].to_numpy(), g["n"].to_numpy())
    g.insert(0, "feature", col)
    return g.drop(columns=col)
