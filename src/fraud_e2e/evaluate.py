"""One-time holdout evaluation, guarded by a lock file.

The lock records exactly what was evaluated (pipeline spec id + thresholds).
Re-running with the *same* frozen choices just reproduces the same numbers; any
attempt to evaluate a *different* pipeline or threshold on the holdout raises,
so the holdout cannot be used for selection by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_e2e import config as C
from fraud_e2e.cost import CostParams, outcome_costs, three_tier_actions, three_tier_costs
from fraud_e2e.features import feature_frame
from fraud_e2e.models import fit_pipeline, normalise_spec, spec_id

LOCK_PATH = C.ROOT / "notes" / "holdout_lock.json"


def fit_final(dev: pd.DataFrame, spec: dict):
    """Refit the chosen pipeline on all dev rows."""
    if "split" in dev and not dev["split"].eq("dev").all():
        raise ValueError("final model must be fit on dev rows only")
    spec = normalise_spec(spec)
    return fit_pipeline(spec, feature_frame(dev, spec["drop"]), dev[C.TARGET].to_numpy())


def _check_lock(frozen: dict, lock_path: Path, allow_create: bool) -> None:
    if lock_path.exists():
        locked = json.loads(lock_path.read_text())
        if locked != frozen:
            raise PermissionError(
                f"Holdout already evaluated with different choices ({lock_path}). "
                "Refusing to evaluate a new pipeline/threshold on the holdout."
            )
    elif allow_create:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(frozen, indent=2) + "\n")
    else:
        raise PermissionError("No holdout lock yet; pass allow_create=True for the one-time run")


def _metrics(y, s, amount, t_flag, p: CostParams, three_tier=None) -> dict:
    flag = s >= t_flag
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    out = {
        "pr_auc": average_precision_score(y, s),
        "roc_auc": roc_auc_score(y, s),
        "base_rate": float(y.mean()),
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "alert_rate": float(flag.mean()),
        "cost_per_1000_txn": float(outcome_costs(y, flag, amount, p).sum() / len(y) * 1000),
        "flag_nothing_cost_per_1000_txn": float(
            outcome_costs(y, np.zeros_like(flag), amount, p).sum() / len(y) * 1000
        ),
    }
    if three_tier is not None:
        q, t1, t2 = three_tier
        act = three_tier_actions(s, t1, t2)
        out["three_tier_cost_per_1000_txn"] = float(
            three_tier_costs(y, act, amount, p, q).sum() / len(y) * 1000
        )
        out["three_tier_decline_rate"] = float((act == 2).mean())
        out["three_tier_stepup_rate"] = float((act == 1).mean())
    return out


def bootstrap(fn, y, s, amount, n_boot=2000, seed=C.SEED) -> dict:
    """Stratified bootstrap (fraud and legit resampled separately) percentile 95% CIs.

    `fn(y, s, amount) -> dict of scalars` is evaluated on the original sample and on each
    resample.
    """
    rng = np.random.default_rng(seed)
    y, s, amount = np.asarray(y), np.asarray(s, float), np.asarray(amount, float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    point = fn(y, s, amount)
    reps = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        reps.append(fn(y[idx], s[idx], amount[idx]))
    reps = pd.DataFrame(reps)
    return {
        k: {
            "value": float(v),
            "ci_low": float(reps[k].quantile(0.025)),
            "ci_high": float(reps[k].quantile(0.975)),
        }
        for k, v in point.items()
    }


def bootstrap_metrics(y, s, amount, t_flag, p, three_tier=None, n_boot=2000, seed=C.SEED):
    """Pre-registered holdout metrics with stratified bootstrap CIs."""
    return bootstrap(
        lambda yy, ss_, aa: _metrics(yy, ss_, aa, t_flag, p, three_tier), y, s, amount, n_boot, seed
    )


def evaluate_holdout(
    df_all: pd.DataFrame,
    spec: dict,
    t_flag: float,
    p: CostParams,
    three_tier=None,
    lock_path: Path = LOCK_PATH,
    allow_create: bool = False,
    n_boot: int = 2000,
) -> dict:
    """Fit on dev, score holdout once, bootstrap CIs. three_tier = (ThreeTierParams, t1, t2)."""
    spec = normalise_spec(spec)
    frozen = {
        "spec_id": spec_id(spec),
        "spec": spec,
        "t_flag": round(float(t_flag), 10),
        "cost_params": p.to_dict(),
        "three_tier": None
        if three_tier is None
        else {
            "params": three_tier[0].to_dict(),
            "t_stepup": round(float(three_tier[1]), 10),
            "t_decline": round(float(three_tier[2]), 10),
        },
    }
    _check_lock(frozen, lock_path, allow_create)
    dev = df_all[df_all["split"] == "dev"]
    hold = df_all[df_all["split"] == "holdout"]
    model = fit_final(dev, spec)
    s = model.predict_proba(feature_frame(hold, spec["drop"]))[:, 1]
    y = hold[C.TARGET].to_numpy()
    res = bootstrap_metrics(y, s, hold[C.AMOUNT_COL].to_numpy(), t_flag, p, three_tier, n_boot)
    return {
        "frozen": frozen,
        "n_rows": len(hold),
        "n_fraud": int(y.sum()),
        "metrics": res,
        "_scores": pd.DataFrame({C.ID_COL: hold[C.ID_COL].to_numpy(), "y": y, "score": s}),
    }
