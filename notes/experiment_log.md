# Experiment log

Every run that produces a number is logged here (date, what, artifact, outcome). Numbers are not copied here; follow the artifact path.

| Date | Phase | ID | What | Artifact | Outcome / decision |
|---|---|---|---|---|---|
| 2026-11-09 | P1 | P1-01 | Lock holdout (20%) + 5-fold × 3 repeats CV on dev | `data/splits/splits.csv` | Locked; sha256 in `notes/00_preregistration.md` and `tests/test_splits.py` |
| 2026-11-09 | P1 | P1-02 | Structural and plausibility checks (all rows, no label use) | `reports/p1_data_understanding.json` | All structural checks pass; implausible channel/device combos reported, not removed |
| 2026-11-09 | P1 | P1-03 | Label-based EDA on dev only: univariate AUC, rate tables, red flags, amount concentration | `reports/p1_eda_summary.json`, `reports/p1_*.csv` | No leakage alarm; `is_ai_generated_scam_attempt` kept (also test without in P2); all 24 features kept |
| 2026-11-09 | P2 | P2-01 | Logistic baseline, unweighted vs balanced (15 folds) | `reports/p2_baseline.json`, `reports/cv/logreg*.folds.csv` | Unweighted kept (balanced worse on PR-AUC, CI excludes 0) |
| 2026-11-09 | P2 | P2-02 | LightGBM vs XGBoost; imbalance none / class weight / undersample / SMOTENC; 20-trial tuning on r0; boosting vs logistic | `reports/p2_modeling.json`, `reports/p2_decisions.csv`, `reports/p2_model_summary.csv` | LightGBM kept (XGB not distinguishable); no imbalance method beat "none"; SMOTENC clearly worse; tuning helped LightGBM; **logistic beat tuned LightGBM → final = logreg** |
| 2026-11-09 | P2 | P2-03 | SMOTENC diagnostic (red-flag prevalence real vs synthetic fraud) | `reports/p2_smote_diagnostic.csv` | Majority-vote on categorical/boolean features dilutes red flags in synthetic fraud |
| 2026-11-09 | P2 | P2-04 | Ablation without `is_ai_generated_scam_attempt`; calibration none/sigmoid/isotonic | `reports/p2_modeling.json` | Flag adds PR-AUC (CI > 0), kept; calibration not distinguishable → none |
| 2026-11-09 | P2 | P2-05 | Two-tier threshold by expected cost on dev OOF; ×0.5/×2 sensitivity; web export | `reports/p2_threshold_web.json`, `reports/p2_cost_sensitivity.csv` | First run on coarse grid; refined grid + near-optimal range (amendment A3) before holdout |
| 2026-11-09 | P2 | P2-06 | Three-tier values accepted by Nat; three-tier thresholds on dev OOF; one-time holdout | `reports/p2_three_tier.json`, `reports/p2_holdout.json`, `notes/holdout_lock.json` | Holdout consistent with CV (CIs overlap); scored once |
| 2026-11-09 | P2 | P2-07 | Post-hoc descriptive holdout metrics (A4) | `reports/p2_holdout_posthoc.json` | Three-tier vs two-tier not distinguishable on holdout |
| 2026-11-09 | P4 | P4-01 | Export final logistic pipeline for in-browser scoring: 5 fold models (CV repeat 0) + final model (all dev); stratified sample of 2,000 dev rows (no holdout rows) | `reports/p4_web_model.json`, `reports/p4_web_stream.csv.gz` | Export = sklearn within 1e-12; refit reproduces committed OOF up to its 6-digit rounding; JS = Python within 1e-9 (Node test) |
