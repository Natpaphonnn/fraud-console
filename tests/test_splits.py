"""Guards for the locked split: no overlap, stratification held, file unchanged."""

import pytest

from fraud_e2e import config as C
from fraud_e2e.data import load_raw, sha256_of
from fraud_e2e.splits import N_FOLDS, N_REPEATS, attach_splits, dev_only, load_splits, make_splits

LOCKED_SPLITS_SHA256 = "5d6a7fadf723b6bd45c1c6882257bd3838face366169c7264e37c0cf4595f23e"


@pytest.fixture(scope="module")
def df():
    return load_raw()


@pytest.fixture(scope="module")
def splits():
    return load_splits()


def test_locked_split_file_unchanged():
    assert sha256_of(C.SPLITS_CSV) == LOCKED_SPLITS_SHA256


def test_split_is_reproducible(df, splits):
    regenerated = make_splits(df)
    assert regenerated.equals(splits)


def test_every_row_assigned_once(df, splits):
    assert len(splits) == len(df)
    assert splits[C.ID_COL].is_unique
    assert set(splits["split"]) == {"dev", "holdout"}


def test_holdout_has_no_fold(splits):
    hold = splits[splits["split"] == "holdout"]
    for r in range(N_REPEATS):
        assert (hold[f"fold_r{r}"] == -1).all()


def test_folds_partition_dev(splits):
    dev = splits[splits["split"] == "dev"]
    for r in range(N_REPEATS):
        assert set(dev[f"fold_r{r}"]) == set(range(N_FOLDS))


def test_stratification(df):
    d = attach_splits(df)
    overall = df[C.TARGET].mean()
    for _, part in d.groupby("split"):
        assert abs(part[C.TARGET].mean() - overall) < 0.001
    dev = d[d["split"] == "dev"]
    for r in range(N_REPEATS):
        counts = dev.groupby(f"fold_r{r}")[C.TARGET].sum()
        assert counts.max() - counts.min() <= 1


def test_dev_only_excludes_holdout(df, splits):
    dev = dev_only(df)
    hold_ids = set(splits.loc[splits["split"] == "holdout", C.ID_COL])
    assert hold_ids.isdisjoint(dev[C.ID_COL])
    assert len(dev) == (splits["split"] == "dev").sum()
