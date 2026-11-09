"""The Kaggle story notebook renders completely, is self-contained and reads the Kaggle input."""

import ast
import re
import sys
from pathlib import Path

import jupytext

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from build_kaggle import STORY, build_story  # noqa: E402


def test_story_renders_and_is_self_contained(tmp_path, monkeypatch):
    import build_kaggle

    monkeypatch.setattr(build_kaggle, "OUT", tmp_path)
    nb = jupytext.read(build_story())
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    ast.parse(code)  # valid Python after rendering
    text = "\n".join(c.source for c in nb.cells)
    assert not re.search(r"\{\{\s*\w+\s*\}\}", text)  # no unrendered placeholder
    assert "fraud_e2e" not in code  # no project package needed on Kaggle
    assert '"/kaggle/input"' in code and "credit_card_fraud_2026.csv" in code
    assert "Illustrative assumed costs (not bank data)" in text and "unspecified" in text
    assert "Udit Jain" in text and "CC0" in text
    assert STORY.exists()
