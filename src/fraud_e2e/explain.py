"""SHAP values without the `shap` package.

Tree models: exact TreeSHAP from the booster itself (LightGBM `pred_contrib`,
XGBoost `pred_contribs`). Logistic regression: exact linear SHAP,
coef x (x - mean(x)) on the standardised design matrix. Values are in log-odds
of the *uncalibrated* model. One-hot columns are summed back to their source
feature so a categorical feature gets one value per transaction.
"""

from __future__ import annotations

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from fraud_e2e.features import feature_frame, source_feature


def _base_pipeline(model):
    """The fitted pipeline to explain. For a calibrated ensemble use the first member."""
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model


def shap_values(model, df: pd.DataFrame, drop=()) -> tuple[pd.DataFrame, float]:
    """Per-row SHAP per source feature (log-odds), and the expected value (base)."""
    pipe = _base_pipeline(model)
    prep, est = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xt = prep.transform(feature_frame(df, drop))
    names = prep.get_feature_names_out()
    kind = type(est).__name__
    if kind == "LGBMClassifier":
        contrib = est.booster_.predict(Xt, pred_contrib=True)
        phi, base = contrib[:, :-1], float(contrib[0, -1])
    elif kind == "XGBClassifier":
        import xgboost as xgb

        contrib = est.get_booster().predict(xgb.DMatrix(Xt, feature_names=None), pred_contribs=True)
        phi, base = contrib[:, :-1], float(contrib[0, -1])
    elif kind == "LogisticRegression":
        mu = Xt.mean(axis=0)
        phi = (Xt - mu) * est.coef_[0]
        base = float(est.intercept_[0] + mu @ est.coef_[0])
    else:
        raise TypeError(f"no SHAP implementation for {kind}")
    by_col = pd.DataFrame(phi, columns=names, index=df.index)
    groups = [source_feature(n, drop) for n in names]
    return by_col.T.groupby(groups, sort=False).sum().T, base


def global_importance(phi: pd.DataFrame) -> pd.DataFrame:
    imp = phi.abs().mean().sort_values(ascending=False)
    return imp.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})
