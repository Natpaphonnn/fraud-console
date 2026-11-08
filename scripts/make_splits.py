"""Create and lock the holdout + repeated CV fold assignment (run once).

Refuses to overwrite an existing split file unless --force is given, so the
locked folds cannot change silently.
"""

import argparse

from fraud_e2e import config as C
from fraud_e2e.data import load_raw
from fraud_e2e.splits import make_splits, save_splits

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

if C.SPLITS_CSV.exists() and not args.force:
    raise SystemExit(f"{C.SPLITS_CSV} already exists; refusing to overwrite (use --force).")

df = load_raw()
splits = make_splits(df)
save_splits(splits)
summary = (
    splits.merge(df[[C.ID_COL, C.TARGET]], on=C.ID_COL)
    .groupby("split")[C.TARGET]
    .agg(["size", "sum", "mean"])
)
print(summary)
