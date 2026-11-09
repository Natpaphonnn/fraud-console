"""Run Kaggle notebooks in a simulated Kaggle layout (no repo on the path), report time and errors.

    uv run python scripts/test_kaggle.py [kaggle/fraud_story.ipynb ...]      # (make kaggle-test)

Layout: <tmp>/input/credit-card-fraud-detection-2026/credit_card_fraud_2026.csv and
<tmp>/working, via KAGGLE_INPUT_DIR / KAGGLE_WORKING_DIR (Kaggle: /kaggle/input, /kaggle/working).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "raw" / "credit_card_fraud_2026.csv"
DEFAULT = ["kaggle/fraud_story.ipynb", "kaggle/08_fairness.ipynb", "kaggle/09_interactions.ipynb"]


def run(nb_rel: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inp = root / "input" / "credit-card-fraud-detection-2026"
        inp.mkdir(parents=True)
        shutil.copy(DATA, inp / DATA.name)
        work = root / "working"
        work.mkdir()
        nb = work / Path(nb_rel).name
        shutil.copy(REPO / nb_rel, nb)
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FRAUD_E2E_ROOT")}
        env |= {"KAGGLE_INPUT_DIR": str(root / "input"), "KAGGLE_WORKING_DIR": str(work)}
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
             "--inplace", "--ExecutePreprocessor.timeout=900", str(nb)],
            cwd=work, env=env, capture_output=True, text=True,
        )  # fmt: skip
        secs = time.time() - t0
        if r.returncode:
            print(f"FAIL {nb_rel} ({secs:.0f} s)\n{r.stderr[-1500:]}")
            return False
        cells = json.loads(nb.read_text())["cells"]
        errors = [o for c in cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
        src = "".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
        text = nb.read_text()
        uses_repo = str(Path.home()) in text or str(REPO) in text
        ok = not errors and not uses_repo
        print(f"{'ok  ' if ok else 'FAIL'} {nb_rel}: {secs:.0f} s, errors={len(errors)}, "
              f"imports project package={'fraud_e2e' in src and 'sys.path.insert' not in src}, "
              f"local paths in outputs={uses_repo}")  # fmt: skip
        return ok


if __name__ == "__main__":
    targets = sys.argv[1:] or DEFAULT
    sys.exit(0 if all([run(t) for t in targets]) else 1)
