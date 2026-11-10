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


def test_landscape_shows_both_plateaus_and_a_flat_logodds_plane(dev):
    d = cf.landscape_data(dev, n=15)
    assert d["P"].min() < 0.01 and d["P"].max() > 0.95  # low plateau, high plateau, cliff between
    assert d["P"].min() < d["thr"] < d["P"].max()
    L = d["L"]  # a logistic model is exactly a plane in log-odds
    assert np.allclose(np.diff(L, 2, axis=0), 0, atol=1e-9) and np.allclose(
        np.diff(L, 2, axis=1), 0, atol=1e-9
    )
    fraud = dev[dev[C.TARGET] == 1][C.BOOLEAN_COLS].mean().sort_values(ascending=False)
    assert d["flags_on"] == list(fraud.index[: cf.LANDSCAPE_FLAGS])


def test_captions_credit_the_data_and_label_costs(dev):
    data = {
        "landscape": cf.landscape_data(dev, n=10),
        "cost": cf.cost_data(),
        "galaxy": cf.galaxy_data(dev),
    }
    for name, (_, key, *_rest) in cf.FIGURES.items():
        t = cf.texts(name, data[key])
        assert "Udit Jain" in t["credit"] and "CC0" in t["credit"]
    assert (
        "Illustrative assumed costs (not bank data)"
        in cf.texts("cost_valley", data["cost"])["credit"]
    )


def test_galaxy_uses_every_dev_row_and_no_holdout(dev):
    d = cf.galaxy_data(dev)
    assert d["P"].shape == (len(dev), 3)
    assert int(d["y"].sum()) == int(dev[C.TARGET].sum())
    assert 0 < d["var"] <= 1


def test_annotated_captions_quote_the_reports_and_stay_readable(dev):
    land = cf.landscape_data(dev, n=15)
    land["_contour"] = cf.threshold_contour(land)
    web = json.loads((C.REPORTS_DIR / "p2_threshold_web.json").read_text())
    thr = json.loads((C.REPORTS_DIR / "p2_threshold.json").read_text())
    caps = {
        "landscape": cf.annotations("landscape", land),
        "cost_valley": cf.annotations("cost_valley", cf.cost_data()),
        "galaxy": cf.annotations("galaxy", cf.galaxy_data(dev)),
    }
    assert caps["cost_valley"][2]["text"].startswith(
        f"Cheapest point ≈ ${thr['cost_per_1000_txn']:,.0f}"
    )
    assert caps["cost_valley"][2]["small"] == "Illustrative assumed costs (not bank data)"
    assert caps["galaxy"][0]["text"] == f"{web['n_rows']:,} transactions"
    assert caps["galaxy"][1]["text"] == f"Gold = fraud ({web['n_fraud']:,})"
    galaxy_last = caps["galaxy"][2]["text"].replace(" · ", " ")
    assert galaxy_last == "Fraud sits apart — the model learns this"
    assert cf.SMALL_PT >= cf.CAPTION_PT / 2  # the cost note is readable on a phone
    targets = {name: [c for c in cs if "target" in c] for name, cs in caps.items()}
    assert all(len(t) == 1 for t in targets.values())  # threshold, cheapest point, fraud cluster
    for cs in caps.values():
        assert len(cs) == len(cf.ANNOT_WINDOWS) == 3
        for c in cs:
            assert all(len(line.split()) <= 8 for line in c["text"].split(" · "))


def test_annotation_timing_holds_each_caption_and_loops_seamlessly():
    total, fps = cf.ANNOT_SECONDS, cf.CLIP_FPS
    times = np.arange(total * fps) / fps
    alphas = np.array(
        [[cf.caption_alpha(t, s, e, total) for s, e in cf.ANNOT_WINDOWS] for t in times]
    )
    assert alphas[0, 0] == 1.0  # text from the very first frame
    held = (alphas >= 1.0).sum(axis=0) / fps
    assert (held >= 1.5).all() and (held <= 2.5).all(), held  # ~1.5-2 s fully visible each
    assert cf.FADE <= 0.3  # quick fade-in: partly transparent for at most 0.3 s
    fading = ((alphas > 0) & (alphas < 1)).sum(axis=0) / fps
    assert (fading <= 2 * 0.3 + 1e-9).all(), fading  # fade-in + fade-out per caption
    assert ((alphas > 0).sum(axis=1) <= 1).all()  # one caption at a time
    # the last frame flows into the first: caption 1 is fading in at the end of the loop
    assert 0 < alphas[-1, 0] < 1 and alphas[-1, 1:].sum() == 0


@pytest.mark.parametrize("size", [(1080, 1080), (1080, 1920)])
def test_annotated_captions_fit_the_safe_area_without_lone_words(dev, size):
    import matplotlib.pyplot as plt

    w, h = size
    left, right, _, _ = cf.safe_box(w, h)
    cf.register_mono()
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    land = cf.landscape_data(dev, n=10)
    land["_contour"] = cf.threshold_contour(land)
    data = {"landscape": land, "cost_valley": cf.cost_data(), "galaxy": cf.galaxy_data(dev)}
    for name, d in data.items():
        for cap in cf.annotations(name, d):
            lines = cf.wrap(fig, cap["text"], cf.CAPTION_PT, (right - left) * w)
            assert len(lines) <= 3, lines
            assert len(lines) == 1 or len(lines[-1].split()) > 1, lines  # no lone last word
            if "small" in cap:
                assert len(cf.wrap(fig, cap["small"], cf.SMALL_PT, (right - left) * w)) <= 2
    plt.close(fig)
