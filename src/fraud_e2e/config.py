"""Project-wide constants: paths, seed, column groups.

Paths resolve to the repo layout locally and to the Kaggle input mount when the
code runs inside a Kaggle notebook, so the same notebooks work in both places.
"""

from __future__ import annotations

import os
from pathlib import Path

SEED = 42

# Repo root = two levels above this file (src/fraud_e2e/config.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Kaggle mounts datasets under /kaggle/input/<slug>/ (layout varies), outputs go to
# /kaggle/working. Env overrides exist only so the Kaggle build can be tested locally.
_KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
_KAGGLE_WORKING = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
_RAW_NAME = "credit_card_fraud_2026.csv"
ON_KAGGLE = _KAGGLE_INPUT.is_dir() and "FRAUD_E2E_ROOT" not in os.environ

if ON_KAGGLE:
    ROOT = _KAGGLE_WORKING
    _found = sorted(_KAGGLE_INPUT.rglob(_RAW_NAME))
    if not _found:
        raise FileNotFoundError(f"Attach the dataset: no {_RAW_NAME} under {_KAGGLE_INPUT}")
    RAW_CSV = _found[0]
else:
    ROOT = Path(os.environ.get("FRAUD_E2E_ROOT", _REPO_ROOT))
    RAW_CSV = ROOT / "data" / "raw" / _RAW_NAME

SPLITS_CSV = ROOT / "data" / "splits" / "splits.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIGS_DIR = ROOT / "configs"

# SHA-256 of the canonical CSV (see data/raw/SOURCE.md). Loading fails loudly if
# the file on disk is not byte-identical to the one the analysis was built on.
RAW_SHA256 = "4e32b1ad4880cae452d1f2e5aa53224d38eb913c185455c00a3759df5eb3da61"

ID_COL = "transaction_id"
TARGET = "is_fraud"
AMOUNT_COL = "amount_usd"

CATEGORICAL_COLS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]

BOOLEAN_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

# Integer-valued columns with few distinct values (counts / clock fields).
DISCRETE_COLS = [
    "txn_count_last_24h",
    "cvv_retry_count",
    "prior_disputes",
    "time_of_day_hour",
    "day_of_week",
]

CONTINUOUS_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
]

NUMERIC_COLS = CONTINUOUS_COLS + DISCRETE_COLS
FEATURE_COLS = CATEGORICAL_COLS + BOOLEAN_COLS + NUMERIC_COLS
