"""Threshold selection by expected cost on out-of-fold (OOF) predictions.

The OOF frame has one row per (repeat, transaction) with columns
repeat, y, amount_usd, score. Each repeat is a full set of dev predictions, so
each gives its own cost curve; curves are averaged across repeats and the
minimum is the chosen threshold (pre-registration A2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_e2e.cost import (
    CostParams,
    ThreeTierParams,
    confusion_by_threshold,
    cost_from_confusion,
    require_usable,
    scaled,
)

NEAR_OPTIMAL_TOL = 0.01  # thresholds within 1% of the minimum cost are "near-optimal"


def threshold_grid() -> np.ndarray:
    """Fine grid for 1-D optimisation: dense near 0 (rare-class thresholds), covers [0, 1]."""
    return np.unique(np.r_[0.0, np.geomspace(1e-4, 1.0, 4001), np.linspace(0, 1, 1001)])


def web_grid(extra=()) -> np.ndarray:
    """Coarser grid for the website export and the 2-D three-tier search, plus extra points."""
    return np.unique(np.r_[0.0, np.geomspace(1e-4, 1.0, 481), np.linspace(0, 1, 201), extra])


def near_optimal_range(grid: np.ndarray, mean_curve: np.ndarray, tol=NEAR_OPTIMAL_TOL):
    """Lowest and highest threshold whose mean cost is within `tol` of the minimum."""
    ok = mean_curve <= mean_curve.min() * (1 + tol)
    return float(grid[ok].min()), float(grid[ok].max())


def confusion_per_repeat(oof: pd.DataFrame, grid: np.ndarray) -> dict[int, dict]:
    return {
        r: confusion_by_threshold(g["y"], g["score"], g["amount_usd"], grid)
        for r, g in oof.groupby("repeat")
    }


def cost_curves(conf: dict[int, dict], p: CostParams) -> np.ndarray:
    """Array (n_repeats, n_thresholds) of total cost in USD."""
    return np.vstack([cost_from_confusion(c, p) for c in conf.values()])


def operating_point(conf: dict[int, dict], i: int, n_rows: int) -> dict:
    """Mean over repeats of the operating-point statistics at grid index i."""
    tp = np.mean([c["tp"][i] for c in conf.values()])
    fp = np.mean([c["fp"][i] for c in conf.values()])
    fn = np.mean([c["fn"][i] for c in conf.values()])
    return {
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp) if tp + fp > 0 else float("nan"),
        "alert_rate": (tp + fp) / n_rows,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(oof: pd.DataFrame, p: CostParams, grid: np.ndarray | None = None) -> dict:
    require_usable(p)
    grid = threshold_grid() if grid is None else grid
    conf = confusion_per_repeat(oof, grid)
    curves = cost_curves(conf, p)
    mean_curve = curves.mean(axis=0)
    i = int(np.argmin(mean_curve))
    n_rows = int(oof.groupby("repeat").size().iloc[0])
    per_repeat_best = [float(grid[np.argmin(c)]) for c in curves]
    y = oof[oof["repeat"] == oof["repeat"].iloc[0]]
    nothing = float(
        (p.fn_loss_fraction * y.loc[y["y"] == 1, "amount_usd"]).sum()
        + p.fn_fixed_cost * y["y"].sum()
    )
    return {
        "threshold": float(grid[i]),
        "grid_index": i,
        "cost_usd": float(mean_curve[i]),
        "cost_per_1000_txn": float(mean_curve[i] / n_rows * 1000),
        "cost_sd_across_repeats": float(curves[:, i].std(ddof=1)) if len(curves) > 1 else None,
        "best_threshold_per_repeat": per_repeat_best,
        "near_optimal_range": near_optimal_range(grid, mean_curve),
        "flag_nothing_cost_usd": nothing,
        "saving_vs_flag_nothing": float(1 - mean_curve[i] / nothing),
        "n_rows": n_rows,
        **operating_point(conf, i, n_rows),
        "_curves": curves,
        "_grid": grid,
    }


def sensitivity(oof: pd.DataFrame, base: CostParams, factors=(0.5, 2.0)) -> pd.DataFrame:
    """Scale one parameter at a time; re-optimise the threshold; also cost of keeping base."""
    grid = threshold_grid()
    conf = confusion_per_repeat(oof, grid)
    n_rows = int(oof.groupby("repeat").size().iloc[0])
    base_i = int(np.argmin(cost_curves(conf, base).mean(axis=0)))
    rows = [{"parameter": "(base)", "factor": 1.0, "value": None}]
    for name in base.numeric():
        for f in factors:
            rows.append({"parameter": name, "factor": f, "value": getattr(base, name) * f})
    out = []
    for r in rows:
        p = base if r["parameter"] == "(base)" else scaled(base, r["parameter"], r["factor"])
        mean_curve = cost_curves(conf, p).mean(axis=0)
        i = int(np.argmin(mean_curve))
        lo, hi = near_optimal_range(grid, mean_curve)
        out.append(
            {
                **r,
                "best_threshold": float(grid[i]),
                "near_optimal_low": lo,
                "near_optimal_high": hi,
                "cost_per_1000_txn": float(mean_curve[i] / n_rows * 1000),
                "cost_per_1000_at_base_threshold": float(mean_curve[base_i] / n_rows * 1000),
                "regret_per_1000_txn": float((mean_curve[base_i] - mean_curve[i]) / n_rows * 1000),
                **{
                    k: v
                    for k, v in operating_point(conf, i, n_rows).items()
                    if k in ("recall", "precision", "alert_rate")
                },
            }
        )
    return pd.DataFrame(out)


def export_for_web(oof: pd.DataFrame, grid: np.ndarray | None = None) -> dict:
    """Per-threshold sufficient statistics (mean over repeats) for the static website."""
    grid = web_grid() if grid is None else grid
    conf = confusion_per_repeat(oof, grid)
    n_rows = int(oof.groupby("repeat").size().iloc[0])
    keys = ["tp", "fp", "fn", "tn", "fn_amount_usd"]
    mean = {k: np.mean([c[k] for c in conf.values()], axis=0) for k in keys}
    return {
        "n_rows": n_rows,
        "n_fraud": int(mean["tp"][0] + mean["fn"][0]),
        "fraud_amount_usd": float(
            oof.loc[oof["y"] == 1].groupby("repeat")["amount_usd"].sum().mean()
        ),
        "threshold": grid.round(6).tolist(),
        **{k: np.round(v, 3).tolist() for k, v in mean.items()},
        "cost_formula": (
            "(tp + fp) * review_cost_per_alert + fp * fp_friction_cost"
            " + fn_loss_fraction * fn_amount_usd + fn * fn_fixed_cost"
        ),
        "note": "Mean over 3 CV repeats of out-of-fold predictions on the dev split.",
    }


# --------------------------------------------------------------------------- three-tier


def _prefix_at_grid(y, s, a, grid, p: CostParams):
    order = np.argsort(s)
    ys, as_ = y[order].astype(bool), a[order]
    idx = np.searchsorted(s[order], grid, side="left")  # rows with score < t
    fn_cost = np.where(ys, p.fn_loss_fraction * as_ + p.fn_fixed_cost, 0.0)
    pre = lambda v: np.r_[0.0, np.cumsum(v)]  # noqa: E731
    return {
        "N": pre(np.ones(len(ys)))[idx],
        "L": pre(~ys)[idx],
        "F": pre(ys)[idx],
        "FNC": pre(fn_cost)[idx],
        "tot": {
            "N": len(ys),
            "L": float((~ys).sum()),
            "F": float(ys.sum()),
            "FNC": float(fn_cost.sum()),
        },
    }


def three_tier_cost_matrix(y, score, amount, grid, p: CostParams, q: ThreeTierParams):
    """cost[i, j] for t_stepup = grid[i], t_decline = grid[j] (inf where j < i).

    approve: score < t1; step-up: t1 <= score < t2; decline: score >= t2.
    """
    P = _prefix_at_grid(np.asarray(y), np.asarray(score, float), np.asarray(amount, float), grid, p)
    t = P["tot"]
    N, L, F, FNC = P["N"], P["L"], P["F"], P["FNC"]
    approve = FNC[:, None]
    stepup = (
        p.review_cost_per_alert * (N[None, :] - N[:, None])
        + p.fp_friction_cost * (L[None, :] - L[:, None])
        + q.stepup_fraud_pass_rate * (FNC[None, :] - FNC[:, None])
    )
    decline = q.decline_fp_cost * (t["L"] - L[None, :]) + q.decline_tp_cost * (t["F"] - F[None, :])
    cost = approve + stepup + decline
    cost[np.tril_indices(len(grid), -1)] = np.inf
    return cost


def choose_three_tier(
    oof: pd.DataFrame, p: CostParams, q: ThreeTierParams, grid: np.ndarray | None = None
) -> dict:
    require_usable(p, q)
    grid = web_grid() if grid is None else grid
    grid = np.r_[grid, np.inf]  # last column = "never decline" (two-tier)
    mats = [
        three_tier_cost_matrix(g["y"].to_numpy(), g["score"], g["amount_usd"], grid, p, q)
        for _, g in oof.groupby("repeat")
    ]
    mean = np.mean(mats, axis=0)
    i, j = np.unravel_index(np.argmin(mean), mean.shape)
    n_rows = int(oof.groupby("repeat").size().iloc[0])
    # two-tier under the same three-tier cost model = never decline (t2 beyond max score)
    two_tier_i = int(np.argmin(mean[:, -1]))
    return {
        "t_stepup": float(grid[i]),
        "t_decline": float(grid[j]),
        "cost_per_1000_txn": float(mean[i, j] / n_rows * 1000),
        "two_tier_same_model_threshold": float(grid[two_tier_i]),
        "two_tier_same_model_cost_per_1000": float(mean[two_tier_i, -1] / n_rows * 1000),
        "cost_sd_across_repeats_per_1000": float(
            np.std([m[i, j] for m in mats], ddof=1) / n_rows * 1000
        ),
        "best_pair_per_repeat": [
            [float(grid[a]), float(grid[b])]
            for a, b in (np.unravel_index(np.argmin(m), m.shape) for m in mats)
        ],
        "n_rows": n_rows,
    }
