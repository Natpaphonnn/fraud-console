"""Export the final logistic pipeline + a scored dev sample for the website (dev rows only).

    uv run python scripts/export_web_model.py      # (make web-model)

Writes reports/p4_web_model.json and reports/p4_web_stream.csv.gz. Deterministic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fraud_e2e import config as C
from fraud_e2e.data import load_raw
from fraud_e2e.export import export_web_model, write_stream_csv
from fraud_e2e.reporting import write_json
from fraud_e2e.splits import dev_only

if __name__ == "__main__":
    dev = dev_only(load_raw())
    spec = json.loads((C.REPORTS_DIR / "cv" / "final.spec.json").read_text())["spec"]
    oof = pd.read_csv(C.REPORTS_DIR / "cv" / "final.oof.csv.gz")
    payload, rows = export_web_model(dev, spec, oof)
    gap = np.abs(rows["score_oof"] - rows["score_oof_committed"]) / rows["score_oof_committed"]
    n, k = len(rows), rows[C.TARGET].sum()
    print(f"{n} rows, {k} fraud; max rel. gap to committed OOF {gap.max():.1e}")
    print("wrote", write_json(payload, "p4_web_model").relative_to(C.ROOT))
    path = C.REPORTS_DIR / "p4_web_stream.csv.gz"
    write_stream_csv(rows, path)
    print("wrote", path.relative_to(C.ROOT))
