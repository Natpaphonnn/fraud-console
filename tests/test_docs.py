"""Generated documents: every placeholder resolves, and cost figures carry the label."""

import re
from pathlib import Path

import pytest

from fraud_e2e.summary import build_context, render

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = sorted((REPO / "docs" / "templates").glob("*.tmpl"))


def test_render_rejects_unknown_keys():
    with pytest.raises(KeyError, match="nope"):
        render("value: {{nope}}", {"a": "1"})


def test_render_substitutes():
    assert render("x={{ a }}, y={{b}}", {"a": "1", "b": "2"}) == "x=1, y=2"


@pytest.mark.parametrize("tmpl", TEMPLATES, ids=lambda p: p.name)
def test_templates_render_completely(tmpl):
    out = render(tmpl.read_text(), build_context())
    assert not re.search(r"\{\{|\}\}", out)


@pytest.mark.parametrize("name", ["README.md.tmpl", "model_card.md.tmpl"])
def test_english_docs_label_costs_as_illustrative(name):
    ctx = build_context()
    assert ctx["cost_label"] in render((REPO / "docs" / "templates" / name).read_text(), ctx)
