"""Write analysis results as committed artifacts (JSON / CSV).

Documents and the website quote numbers only from these files, so every
artifact carries the data hash and seed it was produced with.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_e2e import config as C


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def write_json(payload: dict, name: str, folder: Path = C.REPORTS_DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    body = {"_meta": {"data_sha256": C.RAW_SHA256, "seed": C.SEED}, **payload}
    path.write_text(json.dumps(body, indent=2, default=_default, sort_keys=False) + "\n")
    return path


def write_csv(df: pd.DataFrame, name: str, folder: Path = C.REPORTS_DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    df.to_csv(path, index=False, float_format="%.6g")
    return path


def read_json(name: str, folder: Path = C.REPORTS_DIR) -> dict:
    return json.loads((folder / f"{name}.json").read_text())
