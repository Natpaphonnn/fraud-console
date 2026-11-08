"""Build self-contained Kaggle versions of the notebooks.

Kaggle cannot see this repo, so each generated notebook starts with a bootstrap
cell that writes the `fraud_e2e` package and `configs/cost.yaml` into
/kaggle/working and recreates the locked split (deterministic; guarded by
tests/test_splits.py). The rest of the notebook is identical to the repo version.

Output: kaggle/<name>.ipynb (not executed; Kaggle runs it). Upload is manual.

    uv run python scripts/build_kaggle.py
"""

from __future__ import annotations

from pathlib import Path

import jupytext
import nbformat

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "fraud_e2e"
NB_SRC = REPO / "notebooks" / "py"
OUT = REPO / "kaggle"
DATASET_URL = "https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026"


def bootstrap_source() -> str:
    files = {f"src/fraud_e2e/{p.name}": p.read_text() for p in sorted(SRC.glob("*.py"))}
    files["configs/cost.yaml"] = (REPO / "configs" / "cost.yaml").read_text()
    lines = [
        "# Bootstrap: write the project package into the working directory (auto-generated).",
        "import os, sys",
        "from pathlib import Path",
        'WORK = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))',
        f"FILES = {files!r}",
        "for rel, text in FILES.items():",
        "    p = WORK / rel",
        "    p.parent.mkdir(parents=True, exist_ok=True)",
        "    p.write_text(text)",
        'sys.path.insert(0, str(WORK / "src"))',
        "from fraud_e2e import config as C",
        "from fraud_e2e.data import load_raw",
        "from fraud_e2e.splits import make_splits, save_splits",
        "if not C.SPLITS_CSV.exists():",
        "    save_splits(make_splits(load_raw()))",
        'print("fraud_e2e ready; data:", C.RAW_CSV)',
    ]
    return "\n".join(lines)


def build(py_path: Path) -> Path:
    nb = jupytext.read(py_path)
    header = nbformat.v4.new_markdown_cell(
        "> **Kaggle edition.** Generated from the project repository by "
        "`scripts/build_kaggle.py`. Attach the dataset "
        f"[Credit Card Fraud Detection 2026]({DATASET_URL}) (CC0) before running. "
        "The first code cell installs the project's helper package into `/kaggle/working`."
    )
    boot = nbformat.v4.new_code_cell(bootstrap_source())
    nb.cells = [nb.cells[0], header, boot, *nb.cells[1:]]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    OUT.mkdir(exist_ok=True)
    out = OUT / f"{py_path.stem}.ipynb"
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    for py in sorted(NB_SRC.glob("*.py")):
        print("built", build(py).relative_to(REPO))
