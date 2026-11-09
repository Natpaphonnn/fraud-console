"""Static project page: fresh, labelled, self-contained, and its numbers come from reports/."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fraud_e2e.cost import COST_LABEL
from fraud_e2e.summary import load_links
from fraud_e2e.web import SITE, build_site, page_data, stale_files

REPO = Path(__file__).resolve().parents[1]
DATASET_URL = "https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026"


@pytest.fixture(scope="module")
def page() -> str:
    return build_site()["index.html"].decode()


def test_committed_site_is_fresh():
    assert stale_files() == [], "run `make docs` and commit web/site/"


def test_template_fully_rendered(page):
    html_only = re.sub(r'<script type="application/json".*?</script>', "", page, flags=re.S)
    assert not re.search(r"\{\{|\}\}", html_only)


def test_required_labels_and_credit(page):
    assert page.count(COST_LABEL) >= 3  # hero badge, simulator callout, footer
    assert "Public CC0 dataset of unspecified provenance" in page
    assert "Udit Jain" in page and DATASET_URL in page
    assert "creativecommons.org/publicdomain/zero/1.0" in page


def test_self_contained_no_third_party_assets(page):
    # scripts, stylesheets and icons are local files; no CDN, fonts or trackers
    for tag, attr in (("script", "src"), ("link", "href")):
        for url in re.findall(rf"<{tag}\b[^>]*\b{attr}=\"([^\"]+)\"", page):
            assert not re.match(r"[a-z]+:|//", url), url
    allowed = {DATASET_URL, "https://creativecommons.org/publicdomain/zero/1.0/"}
    allowed |= {u for u in load_links().values() if u}
    for url in re.findall(r"href=\"(https?://[^\"]+)\"", page):
        assert url in allowed, url


def test_embedded_data_is_the_report(page):
    block = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', page, re.S)
    data = json.loads(block.group(1).replace("<\\/", "</"))
    web = json.loads((REPO / "reports" / "p2_threshold_web.json").read_text())
    for k in ("threshold", "tp", "fp", "fn", "tn", "fn_amount_usd", "default_params"):
        assert data["sim"][k] == web[k], k
    assert data == page_data()


def test_static_numbers_match_reports(page):
    thr = json.loads((REPO / "reports" / "p2_threshold.json").read_text())
    hold = json.loads((REPO / "reports" / "p2_holdout.json").read_text())
    assert f"{thr['threshold']:.4f}" in page
    assert f"{hold['metrics']['pr_auc']['value']:.3f}" in page
    assert f"{100 * thr['recall']:.1f}%" in page


def test_links_must_be_https(tmp_path):
    bad = tmp_path / "links.yaml"
    bad.write_text('github: "http://example.com"\n')
    with pytest.raises(ValueError, match="https"):
        load_links(bad)
    bad.write_text('twitter: "https://example.com"\n')
    with pytest.raises(KeyError):
        load_links(bad)


def test_vercel_config_is_valid_json():
    cfg = json.loads((SITE / "vercel.json").read_text())
    assert any(h["key"] == "Content-Security-Policy" for h in cfg["headers"][0]["headers"])


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_simulator_js():
    r = subprocess.run(
        ["node", "--test", "web/tests/"], cwd=REPO, capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0, r.stdout + r.stderr
