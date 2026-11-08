import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fraud_e2e.cost import CostParams, load_cost_params, outcome_costs, reference_costs
from fraud_e2e.stats import bootstrap_auc_ci, rank_auc, wilson_ci


def test_rank_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.1
    x = rng.normal(size=500) + y
    x[:50] = np.round(x[:50])  # include ties
    assert rank_auc(x, y) == pytest.approx(roc_auc_score(y, x))


def test_wilson_ci_brackets_rate():
    lo, hi = wilson_ci(np.array([0, 5, 50]), np.array([100, 100, 100]))
    assert lo[0] == pytest.approx(0, abs=1e-12) and hi[0] > 0
    assert (lo <= np.array([0, 0.05, 0.5])).all() and (hi >= np.array([0, 0.05, 0.5])).all()


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    y = rng.random(2000) < 0.05
    x = rng.normal(size=2000) + y
    lo, hi = bootstrap_auc_ci(x, y, n_boot=200)
    assert lo < rank_auc(x, y) < hi


P = CostParams(review_cost_per_alert=5, fp_friction_cost=10, fn_loss_fraction=1.0, fn_fixed_cost=15)


def test_outcome_costs_each_cell():
    y = np.array([1, 0, 1, 0])
    flag = np.array([1, 1, 0, 0])
    amt = np.array([100.0, 100.0, 100.0, 100.0])
    np.testing.assert_allclose(outcome_costs(y, flag, amt, P), [5, 15, 115, 0])


def test_reference_costs():
    y = np.array([1, 0, 0])
    amt = np.array([50.0, 10.0, 10.0])
    ref = reference_costs(y, amt, P)
    assert ref["flag_nothing"] == pytest.approx(65)
    assert ref["flag_everything"] == pytest.approx(5 + 15 + 15)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        CostParams(
            review_cost_per_alert=-1, fp_friction_cost=0, fn_loss_fraction=1, fn_fixed_cost=0
        )
    with pytest.raises(ValueError):
        CostParams(
            review_cost_per_alert=1, fp_friction_cost=0, fn_loss_fraction=1.5, fn_fixed_cost=0
        )


def test_cost_yaml_loads():
    p = load_cost_params()
    assert p.status in {"proposed", "confirmed"}
