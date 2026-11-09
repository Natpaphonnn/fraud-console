"""Export the final logistic pipeline as plain JSON so the website can score rows itself.

Two kinds of model are exported, both with spec = reports/cv/final.spec.json:

* fold models: the 5 models of CV repeat 0. Each dev row is scored by the model whose
  training folds did NOT contain it, which reproduces the committed out-of-fold scores
  (reports/cv/final.oof.csv.gz, repeat 0). The live stream and its inspector use these.
* final model: refit on all dev rows (`evaluate.fit_final`, the model scored once on the
  holdout). "Try it yourself" uses this one.

An exported model is: intercept + coefficients per model input, the one-hot categories,
and the StandardScaler mean/scale (after log1p for LOG_COLS). Scoring is
    logit = intercept + Σ coef_j · input_j,   p = 1 / (1 + exp(-logit))
`linear_scores` implements exactly what web/src/model.js does.
"""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from fraud_e2e import config as C
from fraud_e2e.evaluate import fit_final
from fraud_e2e.features import feature_frame
from fraud_e2e.models import fit_pipeline, normalise_spec, spec_id

STREAM_N = 2000  # rows shipped to the page (stratified sample of dev, natural fraud rate)
STREAM_REPEAT = 0
SKLEARN_TOL = 1e-12  # exported formula vs sklearn predict_proba


def linear_export(pipe) -> dict:
    """Plain-JSON description of a fitted 'linear' pipeline (see features.make_preprocessor)."""
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    coef = model.coef_[0]
    names = list(prep.get_feature_names_out())
    by_name = dict(zip(names, coef, strict=True))
    out: dict = {"intercept": float(model.intercept_[0])}
    for name, trans, cols in prep.transformers_:
        if name == "cat":
            out["categorical"] = [
                {
                    "feature": f,
                    "categories": [str(c) for c in cats],
                    "coef": [float(by_name[f"cat__{f}_{c}"]) for c in cats],
                }
                for f, cats in zip(cols, trans.categories_, strict=True)
            ]
        elif name == "bool":
            out["boolean"] = [{"feature": f, "coef": float(by_name[f"bool__{f}"])} for f in cols]
        elif name in {"log", "num"}:
            scaler = trans[-1] if name == "log" else trans
            out["log_scaled" if name == "log" else "scaled"] = [
                {
                    "feature": f,
                    "mean": float(m),
                    "scale": float(s),
                    "coef": float(by_name[f"{name}__{f}"]),
                }
                for f, m, s in zip(cols, scaler.mean_, scaler.scale_, strict=True)
            ]
        elif name != "remainder":
            raise ValueError(f"unexpected transformer {name!r}")
    n_inputs = sum(len(b["categories"]) for b in out["categorical"]) + sum(
        len(out[k]) for k in ("boolean", "log_scaled", "scaled")
    )
    if n_inputs != len(coef):
        raise ValueError(f"export covers {n_inputs} of {len(coef)} coefficients")
    return out


def linear_contributions(m: dict, X: pd.DataFrame) -> pd.DataFrame:
    """coef × model input, summed per source feature (one-hot columns summed)."""
    out = {}
    for b in m["categorical"]:
        lut = dict(zip(b["categories"], b["coef"], strict=True))
        out[b["feature"]] = X[b["feature"]].astype(str).map(lut).fillna(0.0).to_numpy(float)
    for b in m["boolean"]:
        out[b["feature"]] = b["coef"] * X[b["feature"]].to_numpy(float)
    for b in m["log_scaled"]:
        out[b["feature"]] = (
            b["coef"] * (np.log1p(X[b["feature"]].to_numpy(float)) - b["mean"]) / b["scale"]
        )
    for b in m["scaled"]:
        out[b["feature"]] = b["coef"] * (X[b["feature"]].to_numpy(float) - b["mean"]) / b["scale"]
    return pd.DataFrame(out, index=X.index)


def linear_scores(m: dict, X: pd.DataFrame) -> np.ndarray:
    logit = m["intercept"] + linear_contributions(m, X).sum(axis=1).to_numpy()
    return 1.0 / (1.0 + np.exp(-logit))


def _checked(pipe, X: pd.DataFrame) -> dict:
    m = linear_export(pipe)
    diff = np.abs(linear_scores(m, X) - pipe.predict_proba(X)[:, 1]).max()
    if diff > SKLEARN_TOL:
        raise AssertionError(f"exported model differs from sklearn by {diff:.2e}")
    return m


def stream_sample(dev: pd.DataFrame, n: int = STREAM_N, seed: int = C.SEED) -> pd.DataFrame:
    """Stratified random sample of dev rows (fraud share as in dev), in random order."""
    rng = np.random.default_rng(seed)
    y = dev[C.TARGET].to_numpy()
    n_fraud = round(n * y.mean())
    idx = np.concatenate(
        [
            rng.choice(np.flatnonzero(y == 1), n_fraud, replace=False),
            rng.choice(np.flatnonzero(y == 0), n - n_fraud, replace=False),
        ]
    )
    return dev.iloc[rng.permutation(idx)].reset_index(drop=True)


def export_web_model(dev: pd.DataFrame, spec: dict, oof: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fold models + final model (JSON) and the scored stream sample (DataFrame)."""
    if not dev["split"].eq("dev").all():
        raise ValueError("export uses dev rows only")
    spec = normalise_spec(spec)
    if spec["model"] != "logreg" or spec["imbalance"] != "none" or spec["calibration"]:
        raise ValueError("only the uncalibrated, unweighted logistic pipeline can be exported")
    X, y = feature_frame(dev, spec["drop"]), dev[C.TARGET].to_numpy()
    fold_col = dev[f"fold_r{STREAM_REPEAT}"].to_numpy()
    folds = {}
    for k in np.unique(fold_col):
        tr, va = fold_col != k, fold_col == k
        folds[str(int(k))] = _checked(fit_pipeline(spec, X[tr], y[tr]), X[va])
    final = _checked(fit_final(dev, spec), X)

    s = stream_sample(dev)
    Xs = feature_frame(s, spec["drop"])
    fold_of = s[f"fold_r{STREAM_REPEAT}"].astype(int).to_numpy()
    score_oof = np.empty(len(s))
    for k, m in folds.items():
        mask = fold_of == int(k)
        score_oof[mask] = linear_scores(m, Xs[mask])
    committed = oof[oof["repeat"] == STREAM_REPEAT].set_index(C.ID_COL)["score"]
    rows = pd.DataFrame(
        {
            C.ID_COL: s[C.ID_COL],
            "fold": fold_of,
            C.TARGET: s[C.TARGET].astype(int),
            **{c: s[c] for c in C.FEATURE_COLS if c not in spec["drop"]},
            "score_oof": score_oof,
            "score_final": linear_scores(final, Xs),
            "score_oof_committed": committed.loc[s[C.ID_COL]].to_numpy(),
        }
    )
    for c in C.BOOLEAN_COLS:
        if c in rows:
            rows[c] = rows[c].astype(int)
    payload = {
        "description": "Logistic pipeline exported for in-browser scoring (see fraud_e2e.export).",
        "spec": spec,
        "spec_id": spec_id(spec),
        "features": [c for c in C.FEATURE_COLS if c not in spec["drop"]],
        "categorical": [c for c in C.CATEGORICAL_COLS if c not in spec["drop"]],
        "boolean": [c for c in C.BOOLEAN_COLS if c not in spec["drop"]],
        "stream": {
            "n_rows": len(rows),
            "n_fraud": int(rows[C.TARGET].sum()),
            "source": "dev split, stratified random sample, random order",
            "scores": f"out-of-fold: each row scored by the CV repeat-{STREAM_REPEAT} model "
            "that was not trained on it",
            "seed": C.SEED,
        },
        "fold_models": folds,
        "final_model": final,
        "sklearn_tolerance": SKLEARN_TOL,
    }
    return payload, rows


def write_stream_csv(rows: pd.DataFrame, path) -> None:
    """Full-precision CSV (floats round-trip exactly), gzip with mtime 0 (deterministic)."""
    data = rows.to_csv(index=False, lineterminator="\n").encode()
    with open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
        gz.write(data)
