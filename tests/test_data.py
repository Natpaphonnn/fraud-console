import pandas as pd
import pytest

from fraud_e2e import config as C
from fraud_e2e.data import load_raw, plausibility_checks, quality_checks


@pytest.fixture(scope="module")
def df():
    return load_raw()


def test_raw_file_hash_matches(df):
    assert len(df) == 20_000
    assert df.shape[1] == 26


def test_hash_mismatch_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("transaction_id,is_fraud\n1,0\n")
    with pytest.raises(ValueError, match="sha256"):
        load_raw(bad)


def test_all_quality_checks_pass(df):
    failed = [q["check"] for q in quality_checks(df) if not q["passed"]]
    assert failed == []


def test_quality_checks_catch_problems(df):
    broken = df.head(100).copy()
    broken.loc[0, "time_of_day_hour"] = 25
    broken.loc[1, "amount_usd"] = None
    broken = pd.concat([broken, broken.iloc[[2]]])
    failed = {q["check"] for q in quality_checks(broken) if not q["passed"]}
    assert {"range_time_of_day_hour", "no_missing_values", "id_unique"} <= failed


def test_id_is_not_a_feature():
    assert C.ID_COL not in C.FEATURE_COLS
    assert C.TARGET not in C.FEATURE_COLS
    assert len(C.FEATURE_COLS) == len(set(C.FEATURE_COLS)) == 24


def test_plausibility_rules_run(df):
    out = plausibility_checks(df)
    assert all(0 <= r["share_of_rows"] <= 1 for r in out)
