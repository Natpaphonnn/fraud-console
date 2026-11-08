import json

import numpy as np
import pandas as pd
import pytest

from fraud_e2e.cost import (
    CostParams,
    ThreeTierParams,
    confusion_by_threshold,
    cost_from_confusion,
    outcome_costs,
    three_tier_actions,
    three_tier_costs,
)
from fraud_e2e.cv import CVResult, corrected_ttest
from fraud_e2e.evaluate import _check_lock
from fraud_e2e.threshold import choose_threshold, three_tier_cost_matrix

P = CostParams(5, 10, 1.0, 15, status="accepted")
Q = ThreeTierParams(0.2, 50, 0, status="accepted")


@pytest.fixture
def toy():
    rng = np.random.default_rng(3)
    n = 3000
    y = (rng.random(n) < 0.03).astype(int)
    s = np.clip(rng.normal(0.1 + 0.4 * y, 0.15), 0, 1)
    s[:20] = 0.25  # ties
    a = rng.lognormal(4, 1, n)
    return y, s, a


def test_confusion_matches_brute_force(toy):
    y, s, a = toy
    grid = np.array([0, 0.1, 0.25, 0.3, 0.9, 1.01])
    conf = confusion_by_threshold(y, s, a, grid)
    for i, t in enumerate(grid):
        brute = outcome_costs(y, s >= t, a, P).sum()
        assert cost_from_confusion(conf, P)[i] == pytest.approx(brute)


def test_three_tier_matrix_matches_brute_force(toy):
    y, s, a = toy
    grid = np.array([0, 0.1, 0.25, 0.4, 0.7, np.inf])
    m = three_tier_cost_matrix(y, s, a, grid, P, Q)
    for i, t1 in enumerate(grid):
        for j, t2 in enumerate(grid):
            if j < i:
                assert m[i, j] == np.inf
                continue
            brute = three_tier_costs(y, three_tier_actions(s, t1, t2), a, P, Q).sum()
            assert m[i, j] == pytest.approx(brute)


def test_two_tier_is_special_case_of_three_tier(toy):
    y, s, a = toy
    q0 = ThreeTierParams(0.0, 50, 0, status="accepted")
    act = three_tier_actions(s, 0.3, np.inf)
    assert three_tier_costs(y, act, a, P, q0).sum() == pytest.approx(
        outcome_costs(y, s >= 0.3, a, P).sum()
    )


def test_choose_threshold_requires_accepted_costs(toy):
    y, s, a = toy
    oof = pd.DataFrame({"repeat": 0, "y": y, "score": s, "amount_usd": a})
    with pytest.raises(PermissionError):
        choose_threshold(oof, CostParams(5, 10, 1, 15, status="proposed"))
    res = choose_threshold(oof, P)
    assert 0 <= res["threshold"] <= 1 and res["cost_usd"] <= res["flag_nothing_cost_usd"]


def _res(name, vals):
    folds = pd.DataFrame(
        {
            "repeat": [0] * len(vals),
            "fold": range(len(vals)),
            "pr_auc": vals,
            "n_train": 800,
            "n_val": 200,
        }
    )
    return CVResult(name, {}, folds, pd.DataFrame())


def test_corrected_ttest_direction_and_width():
    a = _res("a", [0.30, 0.32, 0.31, 0.29, 0.33])
    b = _res("b", [0.40, 0.42, 0.41, 0.39, 0.43])
    t = corrected_ttest(a, b)
    assert t["verdict"] == "b better" and t["mean_diff_b_minus_a"] == pytest.approx(0.10)
    noisy = _res("c", [0.20, 0.45, 0.25, 0.50, 0.30])
    assert corrected_ttest(a, noisy)["verdict"] == "not distinguishable"
    assert corrected_ttest(b, a, lower_is_better=True)["verdict"] == "b better"


def test_holdout_lock(tmp_path):
    lock = tmp_path / "lock.json"
    with pytest.raises(PermissionError):
        _check_lock({"a": 1}, lock, allow_create=False)
    _check_lock({"a": 1}, lock, allow_create=True)
    assert json.loads(lock.read_text()) == {"a": 1}
    _check_lock({"a": 1}, lock, allow_create=False)  # same choices: fine
    with pytest.raises(PermissionError):
        _check_lock({"a": 2}, lock, allow_create=True)
