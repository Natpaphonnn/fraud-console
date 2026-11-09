"""Repeated stratified CV on the dev split using the locked fold assignment."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as ss
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from fraud_e2e import config as C
from fraud_e2e.features import feature_frame
from fraud_e2e.models import fit_pipeline, normalise_spec, spec_id

REPEATS = (0, 1, 2)


def fold_metrics(y, p) -> dict:
    p_clip = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p_clip, labels=[0, 1]),
        "base_rate": float(np.mean(y)),
    }


@dataclass
class CVResult:
    name: str
    spec: dict
    folds: pd.DataFrame  # one row per (repeat, fold)
    oof: pd.DataFrame  # one row per (repeat, transaction)
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        out = {"name": self.name, "spec_id": spec_id(self.spec), "n_folds": len(self.folds)}
        for m in ["pr_auc", "roc_auc", "brier", "log_loss"]:
            v = self.folds[m].to_numpy()
            out[f"{m}_mean"] = float(v.mean())
            out[f"{m}_sd"] = float(v.std(ddof=1))
        out["base_rate_mean"] = float(self.folds["base_rate"].mean())
        out["seconds"] = self.seconds
        return out


def run_cv(dev: pd.DataFrame, spec: dict, name: str, repeats=REPEATS) -> CVResult:
    """Fit on 4 folds, score the 5th, for every fold of every repeat. dev must be dev-only."""
    if "split" in dev and not dev["split"].eq("dev").all():
        raise ValueError("run_cv received non-dev rows; the holdout must not enter CV")
    spec = normalise_spec(spec)
    X = feature_frame(dev, spec["drop"])
    y = dev[C.TARGET].to_numpy()
    rows, oofs = [], []
    t0 = time.perf_counter()
    for r in repeats:
        fold_col = dev[f"fold_r{r}"].to_numpy()
        for k in np.unique(fold_col):
            tr, va = fold_col != k, fold_col == k
            model = fit_pipeline(spec, X[tr], y[tr])
            p = model.predict_proba(X[va])[:, 1]
            rows.append(
                {
                    "repeat": r,
                    "fold": int(k),
                    "n_train": int(tr.sum()),
                    "n_val": int(va.sum()),
                    "n_pos_val": int(y[va].sum()),
                    **fold_metrics(y[va], p),
                }
            )
            oofs.append(
                pd.DataFrame(
                    {
                        C.ID_COL: dev.loc[va, C.ID_COL].to_numpy(),
                        "repeat": r,
                        "fold": int(k),
                        "y": y[va],
                        "amount_usd": dev.loc[va, C.AMOUNT_COL].to_numpy(),
                        "score": p,
                    }
                )
            )
    return CVResult(
        name=name,
        spec=spec,
        folds=pd.DataFrame(rows),
        oof=pd.concat(oofs, ignore_index=True),
        seconds=time.perf_counter() - t0,
    )


def corrected_ttest(
    a: CVResult,
    b: CVResult,
    metric: str = "pr_auc",
    alpha: float = 0.05,
    lower_is_better: bool = False,
) -> dict:
    """Paired comparison b - a over identical folds, Nadeau & Bengio corrected resampled t-test.

    Variance of the mean difference is inflated from s^2/J to (1/J + n_test/n_train) s^2
    to account for overlapping training sets across folds and repeats.
    """
    key = ["repeat", "fold"]
    m = a.folds[key + [metric, "n_train", "n_val"]].merge(
        b.folds[key + [metric]], on=key, suffixes=("_a", "_b"), validate="one_to_one"
    )
    if len(m) != len(a.folds) or len(m) != len(b.folds):
        raise ValueError("results are not on the same folds")
    d = (m[f"{metric}_b"] - m[f"{metric}_a"]).to_numpy()
    J = len(d)
    ratio = float(m["n_val"].mean() / m["n_train"].mean())
    sd = d.std(ddof=1)
    se = np.sqrt((1 / J + ratio) * sd**2)
    mean = float(d.mean())
    tcrit = ss.t.ppf(1 - alpha / 2, J - 1)
    t = mean / se if se > 0 else np.inf * np.sign(mean)
    p = float(2 * ss.t.sf(abs(t), J - 1)) if np.isfinite(t) else 0.0
    lo, hi = mean - tcrit * se, mean + tcrit * se
    return {
        "a": a.name,
        "b": b.name,
        "metric": metric,
        "mean_diff_b_minus_a": mean,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p,
        "n_folds": J,
        "b_wins_per_folds": int((d > 0).sum()),
        "lower_is_better": lower_is_better,
        "verdict": _verdict(lo, hi, lower_is_better),
    }


def _verdict(lo: float, hi: float, lower_is_better: bool) -> str:
    if lower_is_better:
        lo, hi = -hi, -lo
    if lo > 0:
        return "b better"
    if hi < 0:
        return "a better"
    return "not distinguishable"


def pick(
    simpler: CVResult, complex_: CVResult, metric: str = "pr_auc", lower_is_better: bool = False
) -> tuple[CVResult, dict]:
    """Pre-registered rule: keep the simpler pipeline unless the complex one is clearly better."""
    test = corrected_ttest(simpler, complex_, metric, lower_is_better=lower_is_better)
    better = test["verdict"] == "b better"
    test["chosen"] = complex_.name if better else simpler.name
    return (complex_ if better else simpler), test


def metric_ci(res: CVResult, metric: str = "pr_auc", alpha: float = 0.05) -> dict:
    """Mean over folds with a 95% interval using the same corrected variance as the paired test."""
    v = res.folds[metric].to_numpy()
    J = len(v)
    ratio = float(res.folds["n_val"].mean() / res.folds["n_train"].mean())
    se = np.sqrt((1 / J + ratio) * v.var(ddof=1))
    t = ss.t.ppf(1 - alpha / 2, J - 1)
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)),
        "ci_low": float(v.mean() - t * se),
        "ci_high": float(v.mean() + t * se),
    }


CV_DIR = C.REPORTS_DIR / "cv"


def save_result(res: CVResult, folder=CV_DIR, with_oof: bool = True) -> None:
    """Fold metrics + spec (always committed) and OOF predictions (gzip CSV)."""
    folder.mkdir(parents=True, exist_ok=True)
    res.folds.to_csv(folder / f"{res.name}.folds.csv", index=False, float_format="%.6g")
    meta = {
        "name": res.name,
        "spec": res.spec,
        "spec_id": spec_id(res.spec),
        "seconds": round(res.seconds, 1),
        **res.extra,
    }
    (folder / f"{res.name}.spec.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    if with_oof:
        res.oof.to_csv(folder / f"{res.name}.oof.csv.gz", index=False, float_format="%.6g")


def load_result(name: str, folder=CV_DIR) -> CVResult:
    meta = json.loads((folder / f"{name}.spec.json").read_text())
    folds = pd.read_csv(folder / f"{name}.folds.csv")
    oof_path = folder / f"{name}.oof.csv.gz"
    oof = pd.read_csv(oof_path) if oof_path.exists() else pd.DataFrame()
    return CVResult(name=name, spec=meta["spec"], folds=folds, oof=oof, seconds=meta["seconds"])


def verdict_text(test: dict) -> str:
    """Human wording of a paired-test verdict, naming the winner."""
    if test["verdict"] == "b better":
        return f"`{test['b']}` is better"
    if test["verdict"] == "a better":
        return f"`{test['a']}` is better"
    return "not distinguishable"
