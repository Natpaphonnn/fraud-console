"""Scoring API: same scores as Python (<= 1e-9), schema validation, decisions, contributions."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app import app, get_model
from api.schemas import Transaction
from fraud_e2e import config as C

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((REPO / "api" / "example_request.json").read_text())


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def stream() -> pd.DataFrame:
    return pd.read_csv(C.REPORTS_DIR / "p4_web_stream.csv.gz")


def payload(row: pd.Series) -> dict:
    out = {}
    for f in get_model().features:
        v = row[f]
        out[f] = bool(v) if f in C.BOOLEAN_COLS else str(v) if f in C.CATEGORICAL_COLS else float(v)
    return out


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    spec = json.loads((C.REPORTS_DIR / "cv" / "final.spec.json").read_text())
    assert body["status"] == "ok" and body["model"] == spec["spec_id"] and body["n_features"] == 24


def test_schema_matches_model_features():
    assert list(Transaction.model_fields) == get_model().features


def test_scores_match_python_for_every_shipped_row(client, stream):
    """The API, the website and Python score the same transactions identically (<= 1e-9)."""
    worst = 0.0
    thr = json.loads((C.REPORTS_DIR / "p2_threshold.json").read_text())["threshold"]
    for _, row in stream.iterrows():
        body = client.post("/predict", json=payload(row)).json()
        worst = max(worst, abs(body["score"] - row["score_final"]))
        assert body["decision"] == ("step-up" if body["score"] >= thr else "approve")
    assert worst <= 1e-9, worst


def test_contributions_add_up_and_are_the_largest(client):
    body = client.post("/predict", json=EXAMPLE).json()
    m = get_model()
    contribs = dict(m.contributions(EXAMPLE))
    assert math.isclose(m.intercept + sum(contribs.values()), body["logit"], abs_tol=1e-12)
    assert math.isclose(1 / (1 + math.exp(-body["logit"])), body["score"], abs_tol=1e-12)
    top = [c["feature"] for c in body["top_contributions"]]
    assert top == sorted(contribs, key=lambda f: abs(contribs[f]), reverse=True)[:3]
    assert body["threshold_basis"].endswith("Illustrative assumed costs (not bank data)")


def test_example_request_is_a_real_dev_row(client, stream):
    assert EXAMPLE == {k: type(EXAMPLE[k])(v) for k, v in payload(stream.iloc[0]).items()}
    body = client.post("/predict", json=EXAMPLE).json()
    assert abs(body["score"] - stream.iloc[0]["score_final"]) <= 1e-9


@pytest.mark.parametrize(
    "change",
    [
        {"amount_usd": -5},
        {"merchant_risk_score": 101},
        {"used_vpn": "maybe"},
        {"unexpected_field": 1},
        {"channel": "Carrier pigeon"},
    ],
)
def test_invalid_requests_are_rejected(client, change):
    assert client.post("/predict", json={**EXAMPLE, **change}).status_code == 422


def test_missing_field_is_rejected(client):
    bad = dict(EXAMPLE)
    bad.pop("amount_usd")
    assert client.post("/predict", json=bad).status_code == 422


def test_ci_smoke_script_logic(client, monkeypatch, capsys):
    """scripts/ci_api_smoke.py (run against the container in CI) passes against the app itself."""
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    import ci_api_smoke

    def call(url, body=None):
        path = url.split("http://test", 1)[1]
        r = client.get(path) if body is None else client.post(path, json=body)
        return r.status_code, r.json()

    monkeypatch.setattr(ci_api_smoke, "call", call)
    assert ci_api_smoke.main("http://test") == 0
    assert "FAIL" not in capsys.readouterr().out
