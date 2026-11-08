"""Business cost model for an alert/decline decision.

Each transaction scored by the model ends in one of four outcomes:

  TP  fraud, flagged      -> analyst review cost
  FP  legit, flagged      -> analyst review cost + good-customer friction cost
  FN  fraud, not flagged  -> fraction of the amount lost + fixed chargeback handling
  TN  legit, not flagged  -> 0

All parameters live in configs/cost.yaml and are passed in; nothing is hard-coded
here. Values in that file are *assumptions* that must be confirmed (see
notes/01_cost_assumptions.md) before they drive any decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from fraud_e2e import config as C


@dataclass(frozen=True)
class CostParams:
    review_cost_per_alert: float  # USD per flagged transaction (TP and FP)
    fp_friction_cost: float  # USD extra per wrongly flagged good customer
    fn_loss_fraction: float  # share of amount_usd lost when fraud is missed (0..1)
    fn_fixed_cost: float  # USD per missed fraud (chargeback handling)
    status: str = "proposed"  # "proposed" until Nat confirms, then "confirmed"

    def __post_init__(self):
        for name in ("review_cost_per_alert", "fp_friction_cost", "fn_fixed_cost"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not 0 <= self.fn_loss_fraction <= 1:
            raise ValueError("fn_loss_fraction must be within [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


def load_cost_params(path: Path | None = None) -> CostParams:
    path = Path(path or C.CONFIGS_DIR / "cost.yaml")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return CostParams(**raw["params"], status=raw.get("status", "proposed"))


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
