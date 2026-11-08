# Experiment log

Every run that produces a number is logged here (date, what, artifact, outcome). Numbers are not copied here; follow the artifact path.

| Date | Phase | ID | What | Artifact | Outcome / decision |
|---|---|---|---|---|---|
| 2026-11-09 | P1 | P1-01 | Lock holdout (20%) + 5-fold × 3 repeats CV on dev | `data/splits/splits.csv` | Locked; sha256 in `notes/00_preregistration.md` and `tests/test_splits.py` |
| 2026-11-09 | P1 | P1-02 | Structural and plausibility checks (all rows, no label use) | `reports/p1_data_understanding.json` | All structural checks pass; implausible channel/device combos reported, not removed |
| 2026-11-09 | P1 | P1-03 | Label-based EDA on dev only: univariate AUC, rate tables, red flags, amount concentration | `reports/p1_eda_summary.json`, `reports/p1_*.csv` | No leakage alarm; `is_ai_generated_scam_attempt` kept (also test without in P2); all 24 features kept |
