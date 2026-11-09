"""One place that turns committed artifacts (reports/*) into formatted numbers.

README, model card, interview notes and notebook 00 all quote numbers through
`build_context()`, so no document ever contains a hand-typed result.
Templates use `{{key}}` placeholders; `render()` fails on any unknown key.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from fraud_e2e import config as C


def _j(name: str) -> dict:
    return json.loads((C.REPORTS_DIR / f"{name}.json").read_text())


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(C.REPORTS_DIR / f"{name}.csv")


def f3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def signed(x: float, nd: int = 3) -> str:
    return f"{x:+.{nd}f}"


def ci3(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def ci_pct(lo: float, hi: float, nd: int = 1) -> str:
    return f"[{100 * lo:.{nd}f}%, {100 * hi:.{nd}f}%]"


def ci_signed(lo: float, hi: float, nd: int = 3) -> str:
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def usd(x: float) -> str:
    return f"{x:,.0f} USD"


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def build_context() -> dict[str, str]:
    c: dict[str, str] = {}
    p1, eda = _j("p1_data_understanding"), _j("p1_eda_summary")
    allrow = next(r for r in p1["class_balance"] if r["subset"] == "all")
    dev = next(r for r in p1["class_balance"] if r["subset"] == "dev")
    hold_bal = next(r for r in p1["class_balance"] if r["subset"] == "holdout")
    c |= {
        "n_rows": f"{allrow['n']:,}",
        "n_fraud": f"{allrow['n_fraud']}",
        "fraud_rate": f"{allrow['fraud_rate_%']:.2f}%",
        "fraud_rate_ci": f"[{allrow['ci95_low_%']:.2f}%, {allrow['ci95_high_%']:.2f}%]",
        "legit_per_fraud": f"{(allrow['n'] - allrow['n_fraud']) / allrow['n_fraud']:.0f}",
        "n_features": str(sum(len(v) for v in p1["feature_groups"].values())),
        "dev_n": f"{dev['n']:,}",
        "dev_fraud": f"{dev['n_fraud']}",
        "holdout_n": f"{hold_bal['n']:,}",
        "holdout_fraud": f"{hold_bal['n_fraud']}",
        "n_checks": f"{p1['n_quality_checks']}",
        "n_checks_failed": f"{p1['n_quality_checks_failed']}",
        "chan_dev_p": f"{p1['channel_device_independence']['chi2_p_value']:.2f}",
        "chan_dev_gap_pp": f"{p1['channel_device_independence']['max_share_gap_pp']:.1f}",
    }
    worst = max(p1["plausibility_checks"], key=lambda r: r["n_rows"])
    c |= {"implausible_rule": worst["rule"], "implausible_share": pct(worst["share_of_rows"])}

    ls = eda["leakage_screen"]
    c |= {"max_uni_auc": f3(ls["max_direction_free_auc"]), "max_uni_feature": ls["max_feature"]}
    c["ai_flag_fraud_rate"] = pct(eda["ai_scam_flag"]["fraud_rate"])
    rf = eda["red_flags"]
    c |= {
        "flags0_rate": pct(rf[0]["fraud_rate"], 2),
        "flags_top_level": str(rf[-1]["level"]),
        "flags_top_rate": pct(rf[-1]["fraud_rate"]),
        "flags_top_ci": ci_pct(rf[-1]["ci_low"], rf[-1]["ci_high"]),
    }
    logits = [_logit(r["fraud_rate"]) for r in rf]
    steps = [b - a for a, b in zip(logits, logits[1:], strict=False)]
    c |= {
        "flag_logodds_steps": ", ".join(f"{s:+.2f}" for s in steps),
        "flag_odds_multipliers": ", ".join(f"×{math.exp(s):.1f}" for s in steps),
    }
    amt = eda["amount_at_risk"]
    c |= {
        "top10_fraud_usd_share": pct(amt["top10pct_share_of_fraud_usd"], 0),
        "fraud_min_usd": f"{amt['fraud_min_usd']:.2f} USD",
        "legit_below_fraud_min": pct(amt["legit_share_below_fraud_min"]),
        "max_spearman": f"{eda['max_abs_spearman']['value']:.2f}",
        "max_spearman_pair": " / ".join(f"`{x}`" for x in eda["max_abs_spearman"]["pair"]),
    }

    mod = _j("p2_modeling")
    fin = mod["final_pipeline"]
    dec = {d["step"]: d for d in mod["decisions"]}
    web = _j("p2_threshold_web")
    noskill = web["n_fraud"] / web["n_rows"]
    c |= {
        "final_model": "logistic regression"
        if fin["spec"]["model"] == "logreg"
        else fin["spec"]["model"],
        "final_spec_id": fin["spec_id"],
        "cv_pr_auc": f3(fin["pr_auc"]["mean"]),
        "cv_pr_auc_ci": ci3(fin["pr_auc"]["ci_low"], fin["pr_auc"]["ci_high"]),
        "cv_pr_auc_sd": f3(fin["pr_auc"]["sd"]),
        "cv_roc_auc": f3(fin["roc_auc"]["mean"]),
        "cv_roc_auc_ci": ci3(fin["roc_auc"]["ci_low"], fin["roc_auc"]["ci_high"]),
        "noskill": f"{noskill:.4f}",
        "cv_lift": f"{fin['pr_auc']['mean'] / noskill:.0f}",
    }

    def d(step: str, key: str):
        x = dec[step]
        c[f"{key}_diff"] = signed(x["mean_diff_b_minus_a"])
        c[f"{key}_ci"] = ci_signed(x["ci_low"], x["ci_high"])
        c[f"{key}_wins"] = f"{x['b_wins_per_folds']}/{x['n_folds']}"
        c[f"{key}_verdict"] = x["verdict"]

    d("1. boosting library", "xgb_vs_lgbm")
    d("2. imbalance: class_weight", "cw")
    d("2. imbalance: undersample", "us")
    d("2. imbalance: smote", "smote")
    d("3. tuning", "tuning")
    d("4. boosting vs logistic", "boost_vs_lr")
    d("5. ablation (reported only)", "ai_flag")
    summ = _csv("p2_model_summary").set_index("pipeline")
    best_boost = dec["4. boosting vs logistic"]["b"]
    c |= {
        "best_boost_name": best_boost,
        "best_boost_pr_auc": f3(summ.loc[best_boost, "pr_auc_mean"]),
        "lgbm_pr_auc": f3(summ.loc["lgbm", "pr_auc_mean"]),
        "smote_pr_auc": f3(summ.loc["lgbm_smote", "pr_auc_mean"]),
        "lr_balanced_brier": f"{summ.loc['logreg_balanced', 'brier_mean']:.3f}",
        "lr_brier": f"{summ.loc['logreg', 'brier_mean']:.3f}",
    }
    trials = _csv("p2_tuning_trials")
    top5 = trials.head(5)["num_leaves"]
    c |= {
        "tuning_top5_leaves": ", ".join(str(int(v)) for v in top5),
        "tuning_n_trials": str(len(trials)),
    }
    sd = _csv("p2_smote_diagnostic").set_index("feature")
    flags = sd.loc[[f for f in C.BOOLEAN_COLS if f in sd.index]]
    c |= {
        "smote_flag_ratio": f"{(flags['synthetic_fraud_mean'] / flags['real_fraud_mean']).mean():.2f}",
        "smote_ipmis_real": pct(sd.loc["ip_country_mismatch", "real_fraud_mean"], 0),
        "smote_ipmis_syn": pct(sd.loc["ip_country_mismatch", "synthetic_fraud_mean"], 0),
        "smote_foreign_real": pct(sd.loc["is_foreign_transaction", "real_fraud_mean"], 0),
        "smote_foreign_syn": pct(sd.loc["is_foreign_transaction", "synthetic_fraud_mean"], 0),
    }
    cal = [x for k, x in dec.items() if k.startswith("6.")]
    c["calibration_verdict"] = (
        "neither sigmoid nor isotonic calibration improved the Brier score"
        if all(x["verdict"] == "not distinguishable" for x in cal)
        else "calibration changed the Brier score (see notebook 04)"
    )

    thr = _j("p2_threshold")
    cp = thr["cost_params"]
    c |= {
        "threshold": f"{thr['threshold']:.4f}",
        "thr_repeat_range": f"{min(thr['best_threshold_per_repeat']):.4f}–{max(thr['best_threshold_per_repeat']):.4f}",
        "cv_recall": pct(thr["recall"]),
        "cv_precision": pct(thr["precision"]),
        "cv_alert_rate": pct(thr["alert_rate"], 2),
        "cv_cost_per_1000": usd(thr["cost_per_1000_txn"]),
        "cv_nothing_per_1000": usd(thr["flag_nothing_cost_usd"] / thr["n_rows"] * 1000),
        "cv_saving": pct(thr["saving_vs_flag_nothing"]),
        "alerts_per_fraud": f"{1 / thr['precision']:.1f}",
        "cost_review": f"{cp['review_cost_per_alert']:g}",
        "cost_friction": f"{cp['fp_friction_cost']:g}",
        "cost_fn_frac": f"{cp['fn_loss_fraction']:g}",
        "cost_fn_fixed": f"{cp['fn_fixed_cost']:g}",
        "cost_label": thr["cost_label"],
    }
    sens = _csv("p2_cost_sensitivity")
    sn = sens[sens["parameter"] != "(base)"]
    w = sn.loc[sn["regret_per_1000_txn"].idxmax()]
    c |= {
        "sens_thr_min": f"{sn['best_threshold'].min():.4f}",
        "sens_thr_max": f"{sn['best_threshold'].max():.4f}",
        "sens_worst_param": f"{w['parameter']} ×{w['factor']:g}",
        "sens_worst_regret": usd(w["regret_per_1000_txn"]),
    }

    tt = _j("p2_three_tier")
    c |= {
        "tt_stepup": f"{tt['t_stepup']:.4f}",
        "tt_decline": f"{tt['t_decline']:.4f}",
        "tt_pass_rate": f"{tt['params']['stepup_fraud_pass_rate']:g}",
        "tt_pass_rate_pct": pct(tt["params"]["stepup_fraud_pass_rate"], 0),
        "tt_decline_fp": f"{tt['params']['decline_fp_cost']:g}",
        "tt_decline_tp": f"{tt['params']['decline_tp_cost']:g}",
        "tt_cv_diff": f"{tt['cost_per_1000_txn'] - tt['two_tier_same_model_cost_per_1000']:+,.0f} USD",
    }

    h = _j("p2_holdout")
    m = h["metrics"]
    post = _j("p2_holdout_posthoc")
    sv = post["two_tier_saving_vs_nothing"]
    t3 = post["three_tier_model__three_minus_two_per_1000"]
    c |= {
        "ho_n": f"{h['n_rows']:,}",
        "ho_fraud": f"{h['n_fraud']}",
        "ho_pr_auc": f3(m["pr_auc"]["value"]),
        "ho_pr_auc_ci": ci3(m["pr_auc"]["ci_low"], m["pr_auc"]["ci_high"]),
        "ho_roc_auc": f3(m["roc_auc"]["value"]),
        "ho_roc_auc_ci": ci3(m["roc_auc"]["ci_low"], m["roc_auc"]["ci_high"]),
        "ho_noskill": f"{m['base_rate']['value']:.4f}",
        "ho_recall": pct(m["recall"]["value"]),
        "ho_recall_ci": ci_pct(m["recall"]["ci_low"], m["recall"]["ci_high"]),
        "ho_precision": pct(m["precision"]["value"]),
        "ho_precision_ci": ci_pct(m["precision"]["ci_low"], m["precision"]["ci_high"]),
        "ho_alert_rate": pct(m["alert_rate"]["value"], 2),
        "ho_alert_rate_ci": ci_pct(m["alert_rate"]["ci_low"], m["alert_rate"]["ci_high"], 2),
        "ho_saving": pct(sv["value"]),
        "ho_saving_ci": ci_pct(sv["ci_low"], sv["ci_high"]),
        "ho_tt_diff": f"{t3['value']:+,.0f} USD",
        "ho_tt_diff_ci": f"[{t3['ci_low']:+,.0f}, {t3['ci_high']:+,.0f}]",
    }

    shap = _j("p2_shap")
    gi = shap["global_importance"]
    c["shap_top3"] = ", ".join(f"`{x['feature']}`" for x in gi[:3])
    c["shap_top3_share"] = pct(
        sum(x["mean_abs_shap"] for x in gi[:3]) / sum(x["mean_abs_shap"] for x in gi), 0
    )
    ex = {e["label"]: e for e in shap["examples"]}
    c["shap_missed_amount"] = f"{ex['missed fraud']['amount_usd']:,.2f} USD"
    return c


_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render(template: str, ctx: dict[str, str] | None = None) -> str:
    ctx = build_context() if ctx is None else ctx
    missing = sorted({k for k in _PLACEHOLDER.findall(template) if k not in ctx})
    if missing:
        raise KeyError(f"template uses unknown keys: {missing}")
    return _PLACEHOLDER.sub(lambda mt: ctx[mt.group(1)], template)


def render_file(template_path: Path, out_path: Path, header: str = "") -> str:
    text = header + render(template_path.read_text())
    out_path.write_text(text)
    return text
