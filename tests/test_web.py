"""Static project page: fresh, labelled, self-contained, and its numbers come from reports/."""

import json
import re
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
    # (rel="canonical" only names the page's own public URL; it loads nothing)
    for tag, attr in (("script", "src"), ("link", "href")):
        for m in re.finditer(rf"<{tag}\b[^>]*\b{attr}=\"([^\"]+)\"[^>]*>", page):
            if 'rel="canonical"' in m.group(0):
                continue
            assert not re.match(r"[a-z]+:|//", m.group(1)), m.group(1)
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


def test_local_paths_are_relative(page):
    # the site is served under a sub-path (GitHub Pages: /fraud-console/), so no root-absolute URLs
    refs = re.findall(r'(?:href|src)="([^"]+)"', page)
    for url in refs:
        assert not url.startswith("/"), url
    for name in ("app.js", "model.js", "sim.js", "flow.js", "viz.js"):
        js = build_site()[name].decode()
        for spec in re.findall(r'from\s+"([^"]+)"', js):
            assert spec.startswith("./"), (name, spec)
        assert "fetch(" not in js


def test_csp_meta_matches_vercel_policy(page):
    meta = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)">', page)
    assert meta and "script-src 'self'" in meta.group(1) and "frame-ancestors" not in meta.group(1)
