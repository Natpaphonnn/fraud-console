"""Leakage guards: holdout never enters CV, preprocessing/resampling see training rows only."""

import numpy as np
import pandas as pd
import pytest

from fraud_e2e import config as C
from fraud_e2e.cv import run_cv
from fraud_e2e.data import load_raw
from fraud_e2e.evaluate import fit_final
from fraud_e2e.features import feature_frame
from fraud_e2e.models import build_pipeline, fit_pipeline
from fraud_e2e.splits import attach_splits, dev_only

FAST_LGBM = {"model": "lgbm", "params": {"n_estimators": 20}}


@pytest.fixture(scope="module")
def df_all():
    return attach_splits(load_raw())


@pytest.fixture(scope="module")
def dev(df_all):
    return dev_only(df_all)


def test_feature_frame_excludes_id_and_target(dev):
    X = feature_frame(dev)
    assert C.ID_COL not in X and C.TARGET not in X and "split" not in X
    assert not any(c.startswith("fold_") for c in X)


def test_run_cv_rejects_holdout_rows(df_all):
    with pytest.raises(ValueError, match="holdout"):
        run_cv(df_all, {"model": "logreg"}, "x", repeats=(0,))


def test_fit_final_rejects_holdout_rows(df_all):
    with pytest.raises(ValueError):
        fit_final(df_all, {"model": "logreg"})


def test_oof_covers_each_dev_row_once_per_repeat(dev):
    res = run_cv(dev, FAST_LGBM, "fast", repeats=(0,))
    assert res.oof[C.ID_COL].is_unique
    assert set(res.oof[C.ID_COL]) == set(dev[C.ID_COL])
    assert len(res.folds) == 5


def test_validation_rows_not_in_training_fold(dev):
    """The fold column used for validation is disjoint from the one used for training."""
    for r in range(3):
        f = dev[f"fold_r{r}"]
        for k in range(5):
            assert set(dev.loc[f == k, C.ID_COL]).isdisjoint(dev.loc[f != k, C.ID_COL])


def test_scaler_is_fit_on_training_rows_only(dev):
    tr = dev.iloc[:5000]
    pipe = fit_pipeline({"model": "logreg"}, feature_frame(tr), tr[C.TARGET].to_numpy())
    scaler = pipe.named_steps["prep"].named_transformers_["num"]
    assert np.allclose(scaler.mean_[0], tr[scaler.feature_names_in_[0]].mean())


@pytest.mark.parametrize("strategy", ["undersample", "smote"])
def test_resampling_only_changes_training_data(dev, strategy):
    """Samplers run in fit only; predict_proba returns one score per input row, unchanged order."""
    tr, va = dev.iloc[:8000], dev.iloc[8000:9000]
    spec = {**FAST_LGBM, "imbalance": strategy}
    pipe = fit_pipeline(spec, feature_frame(tr), tr[C.TARGET].to_numpy())
    p1 = pipe.predict_proba(feature_frame(va))[:, 1]
    p2 = pipe.predict_proba(feature_frame(va.iloc[::-1]))[:, 1][::-1]
    assert len(p1) == len(va)
    np.testing.assert_allclose(p1, p2)


def test_sampler_ratio(dev):
    tr = dev.iloc[:8000]
    pipe = build_pipeline({**FAST_LGBM, "imbalance": "undersample"})
    Xs, ys = pipe.named_steps["sampler"].fit_resample(feature_frame(tr), tr[C.TARGET].to_numpy())
    assert ys.sum() == tr[C.TARGET].sum()  # all fraud kept
    assert (ys == 0).sum() == pytest.approx(ys.sum() / 0.1, abs=1)


def test_ablation_drops_feature(dev):
    X = feature_frame(dev, drop=["is_ai_generated_scam_attempt"])
    assert "is_ai_generated_scam_attempt" not in X
    with pytest.raises(ValueError):
        feature_frame(dev, drop=["not_a_feature"])


def test_cv_is_deterministic(dev):
    a = run_cv(dev, FAST_LGBM, "a", repeats=(0,))
    b = run_cv(dev, FAST_LGBM, "b", repeats=(0,))
    pd.testing.assert_series_equal(a.oof["score"], b.oof["score"])


@pytest.mark.parametrize("model", ["logreg", "lgbm"])
def test_feature_names_map_back_to_sources(dev, model):
    from fraud_e2e.features import feature_list, source_feature

    spec = {**FAST_LGBM, "model": model} if model == "lgbm" else {"model": "logreg"}
    pipe = fit_pipeline(spec, feature_frame(dev.iloc[:3000]), dev[C.TARGET].iloc[:3000].to_numpy())
    names = pipe.named_steps["prep"].get_feature_names_out()
    assert {source_feature(n) for n in names} == set(feature_list())
