"""Diagnostics that explain modelling results (not used for selection)."""

from __future__ import annotations

import pandas as pd

from fraud_e2e import config as C
from fraud_e2e.features import feature_frame
from fraud_e2e.models import build_pipeline


def smote_synthetic_vs_real(dev: pd.DataFrame, spec: dict, repeat: int = 0, fold: int = 0):
    """Red-flag prevalence in real vs SMOTENC-generated fraud, for one training fold."""
    tr = dev[dev[f"fold_r{repeat}"] != fold]
    X, y = feature_frame(tr), tr[C.TARGET].to_numpy()
    sampler = build_pipeline({**spec, "imbalance": "smote"}).named_steps["sampler"]
    Xs, _ = sampler.fit_resample(X, y)
    synthetic = Xs.iloc[len(X) :]
    real = X[y == 1]
    cols = C.BOOLEAN_COLS + ["cvv_retry_count", "merchant_risk_score", "velocity_score"]
    out = pd.DataFrame(
        {
            "feature": cols,
            "real_fraud_mean": [float(real[c].mean()) for c in cols],
            "synthetic_fraud_mean": [float(synthetic[c].astype(float).mean()) for c in cols],
            "legit_mean": [float(X.loc[y == 0, c].mean()) for c in cols],
        }
    )
    out["n_real_fraud"], out["n_synthetic"] = len(real), len(synthetic)
    return out
