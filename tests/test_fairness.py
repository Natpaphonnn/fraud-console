"""Fairness helpers (A5) and A6 transforms: mechanics only, no project results."""

import json

import numpy as np
import pandas as pd
import pytest

from fraud_e2e import config as C
from fraud_e2e.fairness import AGE_LABELS, _counts, _draws, _rates, age_group, group_gaps
from fraud_e2e.features import make_preprocessor
from fraud_e2e.models import normalise_spec, spec_id


def test_age_bands_edges():
    assert list(age_group([18, 29, 30, 44, 45, 59, 60, 81])) == [
        "18–29", "18–29", "30–44", "30–44", "45–59", "45–59", "60+", "60+",
    ]  # fmt: skip


def test_rates_from_counts():
    r = _rates(n_pos=10, n_neg=90, tp=6, fp=9)
    assert r["recall"] == pytest.approx(0.6) and r["fpr"] == pytest.approx(0.1)
    assert r["stepup_rate"] == pytest.approx(0.15) and r["precision"] == pytest.approx(0.4)


def test_binomial_draws_match_row_resampling():
    """Two binomials per side = stratified row resampling (same mean and spread)."""
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(40), np.zeros(400)].astype(int)
    flag = np.r_[np.arange(40) < 25, np.arange(400) < 30]
    n_pos, n_neg, tp, fp = _counts(y, flag)
    fast = _draws(n_pos, n_neg, tp, fp, rng, 20_000)["recall"]
    pos = np.flatnonzero(y == 1)
    slow = np.array([flag[rng.choice(pos, len(pos))].mean() for _ in range(20_000)])
    assert fast.mean() == pytest.approx(slow.mean(), abs=0.005)
    assert fast.std() == pytest.approx(slow.std(), rel=0.05)


def test_gap_verdict_uses_bonferroni_interval():
    rng = np.random.default_rng(1)
    n = 8000
    groups = np.array(AGE_LABELS)[rng.integers(0, 4, n)]
    y = (rng.random(n) < 0.05).astype(int)
    score = rng.random(n) + 0.5 * (groups == "60+")  # 60+ stepped up far more often
    g = group_gaps(y, score, groups, t=0.9, n_boot=4000)
    row = g[(g.group == "60+") & (g.metric == "stepup_rate")].iloc[0]
    assert row.verdict == "higher" and row.ci_adj_low > 0
    assert (g.ci_adj_high - g.ci_adj_low >= g.ci_high - g.ci_low - 1e-12).all()


@pytest.mark.parametrize("transform", ["spline", "flag_pairs"])
def test_transforms_fit(transform):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.integers(0, 5, 200) for c in C.NUMERIC_COLS})
    for c in C.BOOLEAN_COLS:
        X[c] = rng.integers(0, 2, 200)
    for c in C.CATEGORICAL_COLS:
        X[c] = rng.choice(["a", "b"], 200)
    out = make_preprocessor("linear", transform=transform).fit_transform(X[C.FEATURE_COLS])
    base = make_preprocessor("linear").fit_transform(X[C.FEATURE_COLS])
    assert out.shape[0] == 200 and out.shape[1] > base.shape[1]


def test_final_spec_id_unchanged_by_transform_support():
    spec = json.loads((C.REPORTS_DIR / "cv" / "final.spec.json").read_text())
    assert spec_id(normalise_spec(spec["spec"])) == spec["spec_id"]
