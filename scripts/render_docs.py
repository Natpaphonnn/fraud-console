"""Render generated documents from templates + committed artifacts (no hand-typed numbers).

uv run python scripts/render_docs.py          # write all documents
uv run python scripts/render_docs.py --check  # exit 1 if any document is stale
"""

from __future__ import annotations

import sys
from pathlib import Path

from fraud_e2e.summary import build_context, render

REPO = Path(__file__).resolve().parents[1]
T = REPO / "docs" / "templates"
DOCS = {
    T / "README.md.tmpl": REPO / "README.md",
    T / "model_card.md.tmpl": REPO / "docs" / "model_card.md",
}
# Local-only documents (git-ignored): rendered when their template exists on this machine.
PRIVATE = {
    T / "interview_notes_th.md.tmpl": REPO / "notes" / "interview_notes_th.md",
}
DOCS |= {t: o for t, o in PRIVATE.items() if t.exists()}


def main(check: bool) -> int:
    ctx = build_context()
    stale = []
    for tmpl, out in DOCS.items():
        text = render(tmpl.read_text(), ctx)
        if check:
            if not out.exists() or out.read_text() != text:
                stale.append(out.relative_to(REPO))
        else:
            out.write_text(text)
            print("rendered", out.relative_to(REPO))
    if stale:
        print("stale (run `make docs`):", *stale)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
