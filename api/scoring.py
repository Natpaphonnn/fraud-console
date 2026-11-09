"""Pure-Python scoring with the exported logistic pipeline (no numpy, no project package).

Same model file as the website (reports/p4_web_model.json, `final_model`) and the same formula as
web/src/model.js and fraud_e2e.export.linear_scores:

    categorical -> coefficient of the row's category (one-hot)
    boolean     -> coef * (0|1)
    log_scaled  -> coef * (log1p(x) - mean) / scale
    scaled      -> coef * (x - mean) / scale
    logit = intercept + sum(contributions);  score = 1 / (1 + exp(-logit))
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

MODEL_FILE = "p4_web_model.json"
THRESHOLD_FILE = "p2_threshold.json"


@dataclass(frozen=True)
class Model:
    spec_id: str
    features: list[str]
    intercept: float
    categories: dict[str, dict[str, float]]  # feature -> {category: coef}
    boolean: dict[str, float]
    log_scaled: dict[str, tuple[float, float, float]]  # feature -> (coef, mean, scale)
    scaled: dict[str, tuple[float, float, float]]
    threshold: float
    threshold_basis: str = ""

    def contributions(self, row: dict) -> list[tuple[str, float]]:
        """[(feature, coef x model input)] in the pipeline's column order."""
        out = []
        for f, lut in self.categories.items():
            out.append((f, lut.get(str(row[f]), 0.0)))  # unseen category -> 0 (sklearn 'ignore')
        for f, c in self.boolean.items():
            out.append((f, c * float(bool(row[f]))))
        for f, (c, m, s) in self.log_scaled.items():
            out.append((f, (c * (math.log1p(float(row[f])) - m)) / s))
        for f, (c, m, s) in self.scaled.items():
            out.append((f, (c * (float(row[f]) - m)) / s))
        return out

    def logit(self, row: dict) -> float:
        total = 0.0
        for _, c in self.contributions(row):
            total += c
        return self.intercept + total

    def score(self, row: dict) -> float:
        return 1.0 / (1.0 + math.exp(-self.logit(row)))


def load_model(model_dir: Path) -> Model:
    meta = json.loads((model_dir / MODEL_FILE).read_text())
    m = meta["final_model"]
    thr = json.loads((model_dir / THRESHOLD_FILE).read_text())
    return Model(
        spec_id=meta["spec_id"],
        features=list(meta["features"]),
        intercept=float(m["intercept"]),
        categories={
            b["feature"]: dict(zip(b["categories"], map(float, b["coef"]), strict=True))
            for b in m["categorical"]
        },
        boolean={b["feature"]: float(b["coef"]) for b in m["boolean"]},
        log_scaled={b["feature"]: (b["coef"], b["mean"], b["scale"]) for b in m["log_scaled"]},
        scaled={b["feature"]: (b["coef"], b["mean"], b["scale"]) for b in m["scaled"]},
        threshold=float(thr["threshold"]),
        threshold_basis=(
            f"minimises expected cost on dev out-of-fold predictions; {thr['cost_label']}"
        ),
    )
