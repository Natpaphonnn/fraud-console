"""Loading the raw CSV and data-quality checks.

Every check returns a plain dict so results can be dumped to JSON and quoted in
documents without retyping numbers by hand.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from fraud_e2e import config as C

# Allowed domain per column. Numeric ranges are *plausibility* bounds chosen
# before looking at the data (e.g. an hour must be 0-23); they are not fitted.
EXPECTED_LEVELS: dict[str, set[str]] = {
    "merchant_category": {
        "Online Retail",
        "Groceries",
        "Restaurants",
        "Electronics",
        "Fuel",
        "Travel",
        "Utilities",
        "Healthcare",
        "Gaming",
        "Streaming",
        "Gift Cards",
        "Crypto Exchange",
    },
    "card_type": {"Visa", "Mastercard", "Amex", "RuPay", "Discover"},
    "auth_method": {"3D Secure", "OTP", "PIN", "Biometric", "No Authentication"},
    "channel": {"Online", "POS", "Contactless", "In-App", "ATM"},
    "device_type": {
        "Android Phone",
        "iPhone",
        "POS Terminal",
        "Windows PC",
        "Mac",
        "Tablet",
        "Smart Watch",
        "ATM Machine",
    },
}

RANGES: dict[str, tuple[float, float]] = {
    "amount_usd": (0.01, 1e6),
    "hours_since_last_txn": (0, 24 * 365),
    "txn_count_last_24h": (0, 500),
    "distance_from_home_km": (0, 20_100),  # half Earth's circumference
    "card_age_months": (0, 600),
    "customer_age": (18, 110),
    "account_balance_usd": (0, 1e8),
    "cvv_retry_count": (0, 10),
    "velocity_score": (0, 100),
    "time_of_day_hour": (0, 23),
    "day_of_week": (0, 6),
    "merchant_risk_score": (0, 100),
    "prior_disputes": (0, 100),
}

# Cross-column combinations that would be odd for real card traffic. They are
# reported, not removed: the dataset is used as published.
PLAUSIBILITY_RULES: dict[str, str] = {
    "atm_channel_on_non_atm_device": "channel == 'ATM' and device_type != 'ATM Machine'",
    "atm_device_on_non_atm_channel": "device_type == 'ATM Machine' and channel != 'ATM'",
    "card_present_channel_on_desktop": (
        "channel in ['POS', 'Contactless'] and device_type in ['Windows PC', 'Mac']"
    ),
    "shipping_mismatch_at_atm": "channel == 'ATM' and billing_shipping_mismatch",
    "amount_exceeds_account_balance": "amount_usd > account_balance_usd",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path | None = None, verify_hash: bool = True) -> pd.DataFrame:
    """Read the canonical CSV with explicit dtypes."""
    path = Path(path or C.RAW_CSV)
    if verify_hash:
        got = sha256_of(path)
        if got != C.RAW_SHA256:
            raise ValueError(f"{path} sha256 {got} != expected {C.RAW_SHA256}")
    dtypes = {c: "category" for c in C.CATEGORICAL_COLS}
    dtypes.update({c: "bool" for c in C.BOOLEAN_COLS})
    df = pd.read_csv(path, dtype=dtypes)
    df[C.TARGET] = df[C.TARGET].astype("int8")
    return df


def _check(name: str, passed: bool, **detail) -> dict:
    return {"check": name, "passed": bool(passed), **detail}


def quality_checks(df: pd.DataFrame) -> list[dict]:
    """Structural checks that do not look at the label's relationship with features."""
    out: list[dict] = []
    expected_cols = [C.ID_COL, *C.FEATURE_COLS, C.TARGET]
    out.append(
        _check(
            "columns_match_schema",
            set(df.columns) == set(expected_cols),
            missing=sorted(set(expected_cols) - set(df.columns)),
            unexpected=sorted(set(df.columns) - set(expected_cols)),
        )
    )
    out.append(_check("row_count", True, n_rows=len(df), n_cols=df.shape[1]))

    n_null = df.isna().sum()
    out.append(
        _check(
            "no_missing_values",
            int(n_null.sum()) == 0,
            missing_by_col={k: int(v) for k, v in n_null.items() if v},
        )
    )

    ids = df[C.ID_COL]
    out.append(_check("id_unique", ids.is_unique, n_unique=int(ids.nunique())))
    out.append(
        _check(
            "id_contiguous_1_to_n",
            bool((ids.sort_values().to_numpy() == range(1, len(df) + 1)).all()),
        )
    )

    n_dup = int(df.drop(columns=C.ID_COL).duplicated().sum())
    out.append(_check("no_duplicate_rows_excluding_id", n_dup == 0, n_duplicates=n_dup))
    n_dup_x = int(df[C.FEATURE_COLS].duplicated().sum())
    out.append(_check("no_duplicate_feature_vectors", n_dup_x == 0, n_duplicates=n_dup_x))

    labels = sorted(df[C.TARGET].unique().tolist())
    out.append(
        _check(
            "target_is_binary",
            labels == [0, 1],
            values=labels,
            n_positive=int(df[C.TARGET].sum()),
            positive_rate=float(df[C.TARGET].mean()),
        )
    )

    for col, allowed in EXPECTED_LEVELS.items():
        seen = set(df[col].astype(str).unique())
        out.append(
            _check(
                f"levels_{col}",
                seen <= allowed,
                unexpected=sorted(seen - allowed),
                n_levels=len(seen),
            )
        )

    for col, (lo, hi) in RANGES.items():
        s = df[col]
        bad = int(((s < lo) | (s > hi)).sum())
        out.append(
            _check(
                f"range_{col}",
                bad == 0,
                n_out_of_range=bad,
                observed_min=float(s.min()),
                observed_max=float(s.max()),
                allowed=[lo, hi],
            )
        )
    return out


def plausibility_checks(df: pd.DataFrame) -> list[dict]:
    """Count rows matching each implausible cross-column combination."""
    out = []
    for name, expr in PLAUSIBILITY_RULES.items():
        n = int(df.eval(expr).sum())
        out.append({"rule": name, "expression": expr, "n_rows": n, "share_of_rows": n / len(df)})
    return out
