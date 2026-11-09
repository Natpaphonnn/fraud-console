"""Content figures plot real numbers: checked against the committed results (no rendering here)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from fraud_e2e import config as C
from fraud_e2e.data import load_raw
from fraud_e2e.splits import dev_only

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import content_figures as cf  # noqa: E402


@pytest.fixture(scope="module")
def dev():
    return dev_only(load_raw())


def test_cost_valley_matches_the_threshold_report():
    d = cf.cost_data()
    thr = json.loads((C.REPORTS_DIR / "p2_threshold.json").read_text())
    _, mult, cost, t = d["project"]
    assert mult == 1.0
    assert t == pytest.approx(thr["threshold"], abs=1e-6)
    assert cost == pytest.approx(thr["cost_per_1000_txn"], abs=0.05)


def test_landscape_scores_come_from_the_model(dev):
    d = cf.landscape_data(dev, n=12)
    assert np.allclose(d["Z"], np.log10(d["P"]))
    assert 0 < d["P"].min() < d["P"].max() < 1
    # more merchant risk and more velocity never lower the score (both coefficients positive)
    assert (np.diff(d["P"], axis=0) >= 0).all() and (np.diff(d["P"], axis=1) >= 0).all()


def test_galaxy_uses_every_dev_row_and_no_holdout(dev):
    d = cf.galaxy_data(dev)
    assert d["P"].shape == (len(dev), 3)
    assert int(d["y"].sum()) == int(dev[C.TARGET].sum())
    assert 0 < d["var"] <= 1
