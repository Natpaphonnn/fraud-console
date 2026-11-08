"""Locked data splits.

Design (fixed before any label-based EDA, see notes/00_preregistration.md):
  * 20% stratified holdout, touched once at the very end of modelling.
  * The remaining 80% ("dev") gets repeated stratified 5-fold CV assignments
    (3 repeats, different shuffles) so fold-to-fold variance can be reported.
The assignments are written to data/splits/splits.csv and committed; every
later stage reads that file instead of re-splitting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from fraud_e2e import config as C

HOLDOUT_FRACTION = 0.20
N_FOLDS = 5
N_REPEATS = 3


def make_splits(df: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """Return one row per transaction: split ('dev'/'holdout') and fold_r{k} for dev rows."""
    ids = df[C.ID_COL].to_numpy()
    y = df[C.TARGET].to_numpy()
    dev_ids, _ = train_test_split(ids, test_size=HOLDOUT_FRACTION, stratify=y, random_state=seed)
    out = pd.DataFrame({C.ID_COL: ids, "split": "holdout"})
    out.loc[out[C.ID_COL].isin(dev_ids), "split"] = "dev"

    dev_mask = out["split"].eq("dev").to_numpy()
    y_dev = y[dev_mask]
    for r in range(N_REPEATS):
        fold = np.full(len(out), -1, dtype=int)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed + r)
        dev_fold = np.empty(dev_mask.sum(), dtype=int)
        for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(y_dev)), y_dev)):
            dev_fold[val_idx] = k
        fold[dev_mask] = dev_fold
        out[f"fold_r{r}"] = fold
    return out


def save_splits(splits: pd.DataFrame, path: Path = C.SPLITS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(path, index=False)


def load_splits(path: Path = C.SPLITS_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def attach_splits(df: pd.DataFrame, splits: pd.DataFrame | None = None) -> pd.DataFrame:
    splits = load_splits() if splits is None else splits
    merged = df.merge(splits, on=C.ID_COL, how="left", validate="one_to_one")
    if merged["split"].isna().any():
        raise ValueError("Some transactions have no split assignment")
    return merged


def dev_only(df: pd.DataFrame) -> pd.DataFrame:
    """The only slice that label-based EDA and model selection may look at."""
    if "split" not in df.columns:
        df = attach_splits(df)
    return df[df["split"].eq("dev")].reset_index(drop=True)
