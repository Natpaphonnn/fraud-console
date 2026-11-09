"""Feature matrix and preprocessing (fit inside each training fold by the pipeline)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)

from fraud_e2e import config as C

LOG_COLS = ["amount_usd", "account_balance_usd"]  # heavy right tails (P1)


def feature_list(drop: Sequence[str] = ()) -> list[str]:
    unknown = set(drop) - set(C.FEATURE_COLS)
    if unknown:
        raise ValueError(f"Unknown features to drop: {sorted(unknown)}")
    return [c for c in C.FEATURE_COLS if c not in set(drop)]


def feature_frame(df: pd.DataFrame, drop: Sequence[str] = ()) -> pd.DataFrame:
    """Model input: feature columns only (never the ID or target), booleans as 0/1."""
    cols = feature_list(drop)
    X = df[cols].copy()
    for c in C.BOOLEAN_COLS:
        if c in X:
            X[c] = X[c].astype(np.int8)
    return X


TRANSFORMS = (None, "spline", "flag_pairs")  # P5 / pre-registration A6


def _spline():
    return SplineTransformer(n_knots=5, degree=3, knots="quantile")


def make_preprocessor(
    kind: str, drop: Sequence[str] = (), transform: str | None = None
) -> ColumnTransformer:
    """'linear': one-hot + log1p(heavy tails) + standardise; 'tree': one-hot, raw numerics.

    transform (linear only, A6): "spline" expands every numeric input with cubic B-splines
    (after log1p for LOG_COLS); "flag_pairs" adds the pairwise products of the boolean flags.
    """
    if transform not in TRANSFORMS:
        raise ValueError(f"unknown transform {transform!r}")
    if transform and kind != "linear":
        raise ValueError("transforms are defined for the linear preprocessor only")
    cols = feature_list(drop)
    cats = [c for c in C.CATEGORICAL_COLS if c in cols]
    bools = [c for c in C.BOOLEAN_COLS if c in cols]
    nums = [c for c in C.NUMERIC_COLS if c in cols]
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    if kind == "linear":
        logs = [c for c in nums if c in LOG_COLS]
        plain = [c for c in nums if c not in LOG_COLS]
        log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
        if transform == "spline":
            log_step, num_step = make_pipeline(log1p, _spline()), _spline()
        else:
            log_step, num_step = make_pipeline(log1p, StandardScaler()), StandardScaler()
        bool_step = (
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            if transform == "flag_pairs"
            else "passthrough"
        )
        transformers = [
            ("cat", onehot, cats),
            ("bool", bool_step, bools),
            ("log", log_step, logs),
            ("num", num_step, plain),
        ]
    elif kind == "tree":
        transformers = [
            ("cat", onehot, cats),
            ("bool", "passthrough", bools),
            ("num", "passthrough", nums),
        ]
    else:
        raise ValueError(f"unknown preprocessor kind {kind!r}")
    return ColumnTransformer(transformers, verbose_feature_names_out=True)


def source_feature(transformed_name: str, drop: Sequence[str] = ()) -> str:
    """Map an output column (e.g. 'cat__auth_method_PIN') back to its input feature."""
    _, _, rest = transformed_name.partition("__")
    if rest in C.FEATURE_COLS:
        return rest
    for c in sorted(feature_list(drop), key=len, reverse=True):
        if rest.startswith(c + "_"):
            return c
    raise KeyError(transformed_name)
