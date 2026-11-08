"""Business cost model for flag decisions.

Two-tier policy (primary): each scored transaction is approved or sent to
step-up (review / extra authentication).

  TP  fraud, step-up       -> review cost
  FP  legit, step-up       -> review cost + good-customer friction cost
  FN  fraud, approved      -> fn_loss_fraction x amount_usd + fixed chargeback cost
  TN  legit, approved      -> 0

Three-tier policy (supplementary) adds "decline" and lets a share of fraud pass
step-up; see `ThreeTierParams` and `three_tier_costs`.

All values come from configs/cost.yaml. They are ILLUSTRATIVE assumptions, not
bank data (`COST_LABEL`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from fraud_e2e import config as C

COST_LABEL = "Illustrative assumed costs (not bank data)"


@dataclass(frozen=True)
class CostParams:
    review_cost_per_alert: float  # USD per flagged transaction (TP and FP)
    fp_friction_cost: float  # USD extra per wrongly flagged good customer
    fn_loss_fraction: float  # multiplier on amount_usd lost on a missed fraud (0..2)
    fn_fixed_cost: float  # USD per missed fraud (chargeback handling)
    status: str = "proposed"  # proposed -> accepted (illustrative) / confirmed

    def __post_init__(self):
        for name in ("review_cost_per_alert", "fp_friction_cost", "fn_fixed_cost"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not 0 <= self.fn_loss_fraction <= 2:
            raise ValueError("fn_loss_fraction must be within [0, 2]")

    @property
    def usable(self) -> bool:
        return self.status in {"accepted", "confirmed"}

    def to_dict(self) -> dict:
        return asdict(self)

    def numeric(self) -> dict:
        d = self.to_dict()
        d.pop("status")
        return d


@dataclass(frozen=True)
class ThreeTierParams:
    stepup_fraud_pass_rate: float  # share of fraud that still passes step-up (0..1)
    decline_fp_cost: float  # USD per good customer hard-declined
    decline_tp_cost: float  # USD per fraud declined
    status: str = "proposed"

    def __post_init__(self):
        if not 0 <= self.stepup_fraud_pass_rate <= 1:
            raise ValueError("stepup_fraud_pass_rate must be within [0, 1]")
        if self.decline_fp_cost < 0 or self.decline_tp_cost < 0:
            raise ValueError("decline costs must be >= 0")

    @property
    def usable(self) -> bool:
        return self.status in {"accepted", "confirmed"}

    def to_dict(self) -> dict:
        return asdict(self)


def _load_yaml(path: Path | None) -> dict:
    path = Path(path or C.CONFIGS_DIR / "cost.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_cost_params(path: Path | None = None) -> CostParams:
    raw = _load_yaml(path)
    return CostParams(**raw["params"], status=raw.get("status", "proposed"))


def load_three_tier_params(path: Path | None = None) -> ThreeTierParams:
    raw = _load_yaml(path)["three_tier"]
    return ThreeTierParams(**raw["params"], status=raw.get("status", "proposed"))


def load_sensitivity_factors(path: Path | None = None) -> list[float]:
    return list(_load_yaml(path).get("sensitivity_factors", [0.5, 2.0]))


def require_usable(*params) -> None:
    for p in params:
        if not p.usable:
            raise PermissionError(
                f"{type(p).__name__} status is '{p.status}'; it must be accepted before use"
            )


def scaled(p: CostParams, name: str, factor: float) -> CostParams:
    return replace(p, **{name: getattr(p, name) * factor})


# --------------------------------------------------------------------------- two-tier


def outcome_costs(y_true, flagged, amount, p: CostParams) -> np.ndarray:
    """Per-transaction cost in USD for a given set of flag decisions."""
    y = np.asarray(y_true).astype(bool)
    f = np.asarray(flagged).astype(bool)
    a = np.asarray(amount, dtype=float)
    cost = np.zeros(len(y))
    cost[f] += p.review_cost_per_alert
    cost[f & ~y] += p.fp_friction_cost
    miss = y & ~f
    cost[miss] += p.fn_loss_fraction * a[miss] + p.fn_fixed_cost
    return cost


def total_cost(y_true, flagged, amount, p: CostParams) -> float:
    return float(outcome_costs(y_true, flagged, amount, p).sum())


def reference_costs(y_true, amount, p: CostParams) -> dict:
    """Costs of the two trivial policies every model must beat."""
    y = np.asarray(y_true)
    return {
        "flag_nothing": total_cost(y, np.zeros_like(y, bool), amount, p),
        "flag_everything": total_cost(y, np.ones_like(y, bool), amount, p),
    }


def confusion_by_threshold(y_true, score, amount, thresholds) -> dict[str, np.ndarray]:
    """TP/FP/FN/TN counts and missed-fraud USD for 'flag if score >= t', for every t.

    Cost is linear in these sufficient statistics, so they are all a front end needs to
    recompute cost for arbitrary parameters.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype=float)
    a = np.asarray(amount, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    order = np.argsort(s)
    s_sorted, y_sorted, a_sorted = s[order], y[order], a[order]
    # number of rows with score < t  (they are approved)
    n_below = np.searchsorted(s_sorted, t, side="left")
    cum_pos = np.r_[0, np.cumsum(y_sorted)]
    cum_pos_amt = np.r_[0, np.cumsum(np.where(y_sorted, a_sorted, 0.0))]
    fn = cum_pos[n_below]
    fn_amount = cum_pos_amt[n_below]
    tn = n_below - fn
    tp = y.sum() - fn
    fp = (len(y) - n_below) - tp
    return {
        "threshold": t,
        "tp": tp.astype(int),
        "fp": fp.astype(int),
        "fn": fn.astype(int),
        "tn": tn.astype(int),
        "fn_amount_usd": fn_amount,
    }


def cost_from_confusion(conf: dict, p: CostParams) -> np.ndarray:
    alerts = conf["tp"] + conf["fp"]
    return (
        alerts * p.review_cost_per_alert
        + conf["fp"] * p.fp_friction_cost
        + p.fn_loss_fraction * conf["fn_amount_usd"]
        + conf["fn"] * p.fn_fixed_cost
    )


# --------------------------------------------------------------------------- three-tier


def three_tier_costs(y_true, action, amount, p: CostParams, q: ThreeTierParams) -> np.ndarray:
    """Per-transaction cost for actions 0 = approve, 1 = step-up, 2 = decline.

    Step-up: review cost for everyone, friction for legit, and a share
    `stepup_fraud_pass_rate` of fraud still gets through (expected FN cost).
    Decline: `decline_fp_cost` for legit, `decline_tp_cost` for fraud.
    """
    y = np.asarray(y_true).astype(bool)
    act = np.asarray(action)
    a = np.asarray(amount, dtype=float)
    fn_cost = p.fn_loss_fraction * a + p.fn_fixed_cost
    cost = np.zeros(len(y))
    approve, stepup, decline = act == 0, act == 1, act == 2
    cost[approve & y] = fn_cost[approve & y]
    cost[stepup] += p.review_cost_per_alert
    cost[stepup & ~y] += p.fp_friction_cost
    cost[stepup & y] += q.stepup_fraud_pass_rate * fn_cost[stepup & y]
    cost[decline & ~y] = q.decline_fp_cost
    cost[decline & y] = q.decline_tp_cost
    return cost


def three_tier_actions(score, t_stepup: float, t_decline: float) -> np.ndarray:
    s = np.asarray(score, dtype=float)
    return np.where(s >= t_decline, 2, np.where(s >= t_stepup, 1, 0))
