"""Model pipelines described by a plain-dict spec, so every experiment is loggable.

spec = {
    "model": "logreg" | "lgbm" | "xgb",
    "params": {...},               # overrides of the defaults below
    "imbalance": "none" | "class_weight" | "undersample" | "smote",
    "drop": [...],                 # features to leave out (ablation)
    "calibration": None | "sigmoid" | "isotonic",
}
Samplers sit inside an imblearn Pipeline, so they only ever see training rows.
"""

from __future__ import annotations

import copy
import hashlib
import json

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fraud_e2e import config as C
from fraud_e2e.features import feature_list, make_preprocessor

N_JOBS = 4  # keep the Mac responsive for other projects

DEFAULTS: dict[str, dict] = {
    "logreg": {"C": 1.0, "max_iter": 2000},
    "lgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.0,
    },
    "xgb": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 20 * 0.0169 * (1 - 0.0169),  # ~20 rows' worth of hessian
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
}

UNDERSAMPLE_RATIO = 0.1  # minority / majority after undersampling (legit 10:1)
SMOTE_RATIO = 0.1  # minority / majority after SMOTENC
SMOTE_K = 5


def spec_id(spec: dict) -> str:
    """Short stable hash of a spec (for logs and the holdout lock)."""
    blob = json.dumps(spec, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def normalise_spec(spec: dict) -> dict:
    s = {"params": {}, "imbalance": "none", "drop": [], "calibration": None, **copy.deepcopy(spec)}
    s["drop"] = sorted(s["drop"])
    return s


def _estimator(spec: dict):
    name = spec["model"]
    params = {**DEFAULTS[name], **spec["params"]}
    weighted = spec["imbalance"] == "class_weight"
    if name == "logreg":
        return LogisticRegression(
            **params, class_weight="balanced" if weighted else None, random_state=C.SEED
        )
    if name == "lgbm":
        return LGBMClassifier(
            **params,
            class_weight="balanced" if weighted else None,
            random_state=C.SEED,
            n_jobs=N_JOBS,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
    if name == "xgb":
        # scale_pos_weight is set from the training fold by fit_pipeline when weighted
        return XGBClassifier(
            **params,
            tree_method="hist",
            random_state=C.SEED,
            n_jobs=N_JOBS,
            eval_metric="logloss",
        )
    raise ValueError(f"unknown model {name!r}")


def build_pipeline(spec: dict):
    spec = normalise_spec(spec)
    kind = "linear" if spec["model"] == "logreg" else "tree"
    steps = []
    if spec["imbalance"] == "undersample":
        steps.append(
            (
                "sampler",
                RandomUnderSampler(sampling_strategy=UNDERSAMPLE_RATIO, random_state=C.SEED),
            )
        )
    elif spec["imbalance"] == "smote":
        cols = feature_list(spec["drop"])
        cat_like = [c for c in cols if c in C.CATEGORICAL_COLS + C.BOOLEAN_COLS]
        steps.append(
            (
                "sampler",
                SMOTENC(
                    categorical_features=cat_like,
                    sampling_strategy=SMOTE_RATIO,
                    k_neighbors=SMOTE_K,
                    random_state=C.SEED,
                ),
            )
        )
    elif spec["imbalance"] not in {"none", "class_weight"}:
        raise ValueError(f"unknown imbalance strategy {spec['imbalance']!r}")
    # "transform" (A6) is only present in specs that use it, so older spec ids are unchanged
    prep = make_preprocessor(kind, spec["drop"], spec.get("transform"))
    steps += [("prep", prep), ("model", _estimator(spec))]
    pipe = Pipeline(steps)
    if spec["calibration"]:
        return CalibratedClassifierCV(pipe, method=spec["calibration"], cv=5, ensemble=True)
    return pipe


def fit_pipeline(spec: dict, X, y):
    """Build and fit; handles XGBoost's class weight, which needs the fold's class ratio."""
    spec = normalise_spec(spec)
    pipe = build_pipeline(spec)
    if spec["model"] == "xgb" and spec["imbalance"] == "class_weight":
        ratio = float((y == 0).sum() / (y == 1).sum())
        key = (
            "estimator__model__scale_pos_weight"
            if spec["calibration"]
            else "model__scale_pos_weight"
        )
        pipe.set_params(**{key: ratio})
    return pipe.fit(X, y)
