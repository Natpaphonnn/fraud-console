"""In-browser model export: faithful to the pipeline, dev rows only, reproducible."""

import json

import numpy as np
import pandas as pd
import pytest

from fraud_e2e import config as C
from fraud_e2e.data import load_raw
from fraud_e2e.export import STREAM_N, linear_scores
from fraud_e2e.features import feature_frame
from fraud_e2e.splits import dev_only, load_splits


@pytest.fixture(scope="module")
def model() -> dict:
    return json.loads((C.REPORTS_DIR / "p4_web_model.json").read_text())


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return pd.read_csv(C.REPORTS_DIR / "p4_web_stream.csv.gz")


def test_stream_rows_are_dev_only(rows):
    split = load_splits().set_index(C.ID_COL)["split"]
    assert split.loc[rows[C.ID_COL]].eq("dev").all()
    assert rows[C.ID_COL].is_unique and len(rows) == STREAM_N


def test_stream_keeps_the_dev_fraud_rate(rows):
    dev = dev_only(load_raw())
    assert rows[C.TARGET].sum() == round(STREAM_N * dev[C.TARGET].mean())


def test_python_formula_reproduces_stored_scores(model, rows):
    X = feature_frame(rows.astype({c: bool for c in C.BOOLEAN_COLS}))
    np.testing.assert_allclose(
        linear_scores(model["final_model"], X), rows["score_final"], rtol=0, atol=1e-12
    )
    for k, m in model["fold_models"].items():
        mask = rows["fold"] == int(k)
        np.testing.assert_allclose(
            linear_scores(m, X[mask]), rows.loc[mask, "score_oof"], rtol=0, atol=1e-12
        )


def test_fold_models_reproduce_committed_oof(rows):
    # committed OOF (reports/cv/final.oof.csv.gz) is rounded to 6 significant digits
    rel = (rows["score_oof"] - rows["score_oof_committed"]).abs() / rows["score_oof_committed"]
    assert rel.max() <= 5.1e-6


def test_export_covers_every_feature_once(model):
    m = model["final_model"]
    feats = [b["feature"] for k in ("categorical", "boolean", "log_scaled", "scaled") for b in m[k]]
    assert sorted(feats) == sorted(C.FEATURE_COLS) == sorted(model["features"])
    assert (
        model["spec_id"]
        == json.loads((C.REPORTS_DIR / "cv" / "final.spec.json").read_text())["spec_id"]
    )
