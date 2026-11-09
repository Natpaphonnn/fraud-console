# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 04 · Modelling: boosting, imbalance handling, tuning, calibration
#
# **Questions this notebook answers**
#
# 1. Does gradient boosting (LightGBM / XGBoost) rank fraud better than the logistic baseline?
# 2. Which way of handling the 1:58 class imbalance works best on the same model —
#    nothing, class weights, undersampling, or SMOTE?
# 3. Does a small tuning budget help?
# 4. How much does `is_ai_generated_scam_attempt` contribute?
# 5. Are the predicted probabilities trustworthy, or do they need calibration?
#
# **Protocol** (`notes/00_preregistration.md`, A2). Every comparison is paired on the same 15 folds
# with the corrected resampled t-test, and **the simpler option is kept unless the more complex one
# is better with a 95% CI entirely on the right side of 0**. Order of simplicity:
# logistic < LightGBM < XGBoost; none < class weight < undersampling < SMOTE.
# Holdout rows are never used here.

# %%
from dataclasses import replace

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from fraud_e2e import config as C
from fraud_e2e import plots as P
from fraud_e2e.cv import (
    load_result,
    metric_ci,
    pick,
    run_cv,
    save_result,
    verdict_text,
)
from fraud_e2e.data import load_raw
from fraud_e2e.diagnostics import smote_synthetic_vs_real
from fraud_e2e.models import DEFAULTS, normalise_spec, spec_id
from fraud_e2e.reporting import write_csv, write_json
from fraud_e2e.splits import dev_only

P.set_style()
dev = dev_only(load_raw())
decisions = []  # every pre-registered comparison, in order


def decide(simpler, complex_, step, metric="pr_auc", lower_is_better=False):
    chosen, test = pick(simpler, complex_, metric, lower_is_better)
    decisions.append({"step": step, **test})
    display(
        Markdown(
            f"**{step}** — `{complex_.name}` − `{simpler.name}` {metric}: "
            f"{test['mean_diff_b_minus_a']:+.4f} (95% CI {test['ci_low']:+.4f} to {test['ci_high']:+.4f}; "
            f"`{complex_.name}` ahead in {test['b_wins_per_folds']}/{test['n_folds']} folds) → "
            f"{verdict_text(test)}; keep **`{chosen.name}`**."
        )
    )
    return chosen


def summary_table(results):
    rows = []
    for r in results:
        row = {"pipeline": r.name}
        for m in ["pr_auc", "roc_auc", "brier"]:
            ci = metric_ci(r, m)
            row[f"{m}_mean"], row[f"{m}_ci_low"], row[f"{m}_ci_high"], row[f"{m}_sd"] = (
                ci["mean"],
                ci["ci_low"],
                ci["ci_high"],
                ci["sd"],
            )
        rows.append(row)
    return pd.DataFrame(rows)


baseline = load_result("logreg")
print(f"dev: {len(dev):,} rows; baseline loaded: {baseline.name}")

# %% [markdown]
# ## 1. LightGBM vs XGBoost (both unweighted, default settings)

# %%
lgbm = run_cv(dev, {"model": "lgbm"}, "lgbm")
xgb = run_cv(dev, {"model": "xgb"}, "xgb")
for r in (lgbm, xgb):
    save_result(r)
boost = decide(lgbm, xgb, "1. boosting library")

# %% [markdown]
# ## 2. Imbalance handling on the boosting model
#
# Same model and settings, only the imbalance strategy changes. Resampling happens inside the
# training part of each fold (imblearn pipeline), never on validation rows. Candidates are tried in
# order of simplicity; each replaces the current choice only if it beats it under the rule.

# %%
bname = boost.spec["model"]
imb = {"none": boost}
for strategy in ["class_weight", "undersample", "smote"]:
    imb[strategy] = run_cv(dev, {"model": bname, "imbalance": strategy}, f"{bname}_{strategy}")
    save_result(imb[strategy])

current = imb["none"]
for strategy in ["class_weight", "undersample", "smote"]:
    current = decide(current, imb[strategy], f"2. imbalance: {strategy}")
imb_best = current

# %%
fig, axes = P.plt.subplots(1, 2, figsize=(13, 3.6))
P.fold_dots(
    list(imb.values()),
    "pr_auc",
    ax=axes[0],
    xlabel="PR-AUC per fold (dots); mean ± corrected 95% CI",
)
P.fold_dots(
    list(imb.values()), "brier", ax=axes[1], xlabel="Brier score per fold (lower is better)"
)
axes[0].set_title("Ranking quality")
axes[1].set_title("Probability quality")
fig.tight_layout()
P.save(fig, "p2_imbalance_folds")
P.plt.show()

# %% [markdown]
# Resampling and weighting tell the model fraud is far more common than it is, so its raw
# probabilities are inflated (higher Brier). That is fixable by calibration; ranking (PR-AUC) is
# what the comparison above decides on.

# %% [markdown]
# **Why SMOTE hurts here — a diagnostic.** SMOTENC builds synthetic fraud by interpolating numeric
# features between a fraud case and its nearest fraud neighbours, and sets each categorical/boolean
# feature to the most common value among those neighbours. The table compares the synthetic cases with
# the real fraud they were made from (one training fold).

# %%
smote_diag = smote_synthetic_vs_real(dev, {"model": bname})
write_csv(smote_diag, "p2_smote_diagnostic")
smote_diag.round(3)

# %%
flags = smote_diag[smote_diag["feature"].isin(C.BOOLEAN_COLS)]
ratio = (flags["synthetic_fraud_mean"] / flags["real_fraud_mean"]).mean()
cvv = smote_diag.set_index("feature").loc["cvv_retry_count"]
display(
    Markdown(
        f"On average a red-flag feature is present in synthetic fraud only **{ratio:.2f}×** as often as "
        f"in real fraud (CVV retries: {cvv['synthetic_fraud_mean']:.2f} vs {cvv['real_fraud_mean']:.2f}). "
        "Each flag is absent in most fraud cases, so the neighbours' majority vote usually sets it "
        "to 0. The synthetic 'fraud' therefore looks "
        "much more like legitimate traffic, which teaches the model the wrong boundary."
    )
)

# %% [markdown]
# ## 3. Small tuning budget
#
# 20 random configurations of the chosen boosting pipeline, scored on repeat `r0` only (5 folds).
# The best one is then run on all 15 folds and compared with the untuned pipeline. Because the
# configuration was picked on `r0`, the tuned CV score is optimistically biased; the holdout is
# the unbiased check.

# %%
rng = np.random.default_rng(C.SEED)
if bname == "lgbm":
    space = {
        "num_leaves": [7, 15, 31, 63],
        "learning_rate": lambda: float(np.round(10 ** rng.uniform(-2, -1), 4)),
        "n_estimators": [200, 400, 800],
        "min_child_samples": [10, 20, 50, 100],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.9],
        "reg_lambda": [0.0, 0.1, 1.0, 10.0],
    }
else:
    space = {
        "max_depth": [2, 3, 4, 6],
        "learning_rate": lambda: float(np.round(10 ** rng.uniform(-2, -1), 4)),
        "n_estimators": [200, 400, 800],
        "min_child_weight": [0.1, 0.33, 1.0, 3.0],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.9],
        "reg_lambda": [0.1, 1.0, 10.0],
    }


def sample():
    return {k: (v() if callable(v) else v[rng.integers(len(v))]) for k, v in space.items()}


configs = [sample() for _ in range(20)]
trials = []
for i, params in enumerate(configs):
    spec = {**imb_best.spec, "params": params}
    r = run_cv(dev, spec, f"trial_{i:02d}", repeats=(0,))
    trials.append(
        {
            "trial": i,
            **params,
            "pr_auc_r0_mean": r.folds["pr_auc"].mean(),
            "pr_auc_r0_sd": r.folds["pr_auc"].std(ddof=1),
        }
    )
trials = pd.DataFrame(trials).sort_values("pr_auc_r0_mean", ascending=False).reset_index(drop=True)
write_csv(trials, "p2_tuning_trials")
untuned_r0 = imb_best.folds.query("repeat == 0")["pr_auc"].mean()
display(Markdown(f"Untuned pipeline on `r0`: PR-AUC {untuned_r0:.4f}. Top 5 trials:"))
trials.head(5)

# %%
best_params = configs[int(trials.iloc[0]["trial"])]
tuned = run_cv(dev, {**imb_best.spec, "params": best_params}, f"{imb_best.name}_tuned")
save_result(tuned)
boost_best = decide(imb_best, tuned, "3. tuning")

# %% [markdown]
# ## 4. Best boosting pipeline vs the logistic baseline

# %%
final_family = decide(baseline, boost_best, "4. boosting vs logistic")

# %%
fig, axes = P.plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
P.fold_dots(
    list({r.name: r for r in [baseline, lgbm, xgb, imb_best, tuned]}.values()),
    "pr_auc",
    ax=axes[0],
    xlabel="PR-AUC per fold (dots); mean ± corrected 95% CI",
)
axes[0].set_title("Candidates on the same 15 folds")
P.pr_curves([baseline, boost_best], ax=axes[1])
axes[1].set_title("Precision-recall, out-of-fold (repeat 0)")
fig.tight_layout()
P.save(fig, "p2_model_comparison")
P.plt.show()

# %% [markdown]
# ## 5. Ablation: without `is_ai_generated_scam_attempt`
#
# Pre-registered in P1: the column behaves like an ordinary risk signal, so it stays in the model;
# here we measure how much it adds. This comparison is reported, it does not change the choice.

# %%
abl_spec = {**final_family.spec, "drop": ["is_ai_generated_scam_attempt"]}
ablation = run_cv(dev, abl_spec, f"{final_family.name}_no_ai_flag")
save_result(ablation, with_oof=False)
_, abl_test = pick(ablation, final_family, "pr_auc")
abl_test["step"] = "5. ablation (reported only)"
abl_test.pop("chosen")
decisions.append(abl_test)
display(
    Markdown(
        f"With the flag − without: PR-AUC {abl_test['mean_diff_b_minus_a']:+.4f} "
        f"(95% CI {abl_test['ci_low']:+.4f} to {abl_test['ci_high']:+.4f}) → {verdict_text(abl_test)}."
    )
)

# %% [markdown]
# ## 6. Calibration
#
# Each candidate wraps the final pipeline in `CalibratedClassifierCV` (inner 5-fold, inside every
# outer training fold). Lower Brier score is better; the order of simplicity is none < sigmoid <
# isotonic.

# %%
cal = {"none": final_family}
for method in ["sigmoid", "isotonic"]:
    cal[method] = run_cv(
        dev, {**final_family.spec, "calibration": method}, f"{final_family.name}_{method}"
    )
    save_result(cal[method], with_oof=False)

current = cal["none"]
for method in ["sigmoid", "isotonic"]:
    current = decide(
        current, cal[method], f"6. calibration: {method}", metric="brier", lower_is_better=True
    )
final = current

# %%
from sklearn.calibration import calibration_curve  # noqa: E402

fig, ax = P.plt.subplots(figsize=(5.5, 4.5))
for i, (method, r) in enumerate(cal.items()):
    o = r.oof
    prob_true, prob_pred = calibration_curve(o["y"], o["score"], n_bins=10, strategy="quantile")
    ax.plot(
        prob_pred * 100, prob_true * 100, marker="o", ms=4, lw=2, color=P.SERIES[i], label=method
    )
lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, lim], [0, lim], color=P.NEUTRAL, ls="--", lw=1, label="Perfect calibration")
ax.set_xscale("symlog", linthresh=1)
ax.set_yscale("symlog", linthresh=1)
ax.set_xlabel("Mean predicted fraud probability in bin (%)")
ax.set_ylabel("Observed fraud rate in bin (%)")
ax.set_title("Reliability (OOF, 10 quantile bins, 3 repeats pooled)")
ax.legend()
P.save(fig, "p2_calibration")
P.plt.show()

# %% [markdown]
# ## 7. Freeze the final pipeline

# %%
final_out = replace(final, name="final")
save_result(final_out)

all_results = [
    baseline,
    load_result("logreg_balanced"),
    lgbm,
    xgb,
    *list(imb.values())[1:],
    tuned,
    ablation,
    *list(cal.values())[1:],
]
table = summary_table(all_results)
write_csv(table, "p2_model_summary")
write_csv(pd.DataFrame(decisions), "p2_decisions")
final_spec = normalise_spec(final.spec)
write_json(
    {
        "final_pipeline": {
            "name": final.name,
            "spec": final_spec,
            "spec_id": spec_id(final_spec),
            **{m: metric_ci(final_out, m) for m in ["pr_auc", "roc_auc", "brier"]},
        },
        "defaults": DEFAULTS,
        "decisions": decisions,
        "ablation_ai_flag": abl_test,
        "tuning": {"n_trials": len(trials), "scored_on": "repeat r0", "best_params": best_params},
    },
    "p2_modeling",
)
table.round(4)

# %% [markdown]
# ## 8. Scope of the conclusion: why logistic regression wins *here*
#
# The claim is narrow: **on this dataset, with these features and this protocol, logistic
# regression ranks fraud at least as well as gradient boosting.** It is not a claim that linear
# models beat boosting for fraud in general. Three observations from P1 and from the tuning run
# explain why the result is plausible for this data, and none of them needs to hold elsewhere.
#
# 1. **Signals add up on the log-odds scale.** Logistic regression models the log-odds as a sum of
#    feature effects. P1 showed that fraud odds multiply by a large factor with each extra red
#    flag, which is close to that shape.
# 2. **Little structure for trees to exploit.** Numeric features are at most moderately correlated
#    and the binned fraud rates of the strongest features in P1 rise steadily; channel × device looked
#    independent. Trees
#    earn their keep on interactions and thresholds; there is little of either to find.
# 3. **Tuning pushed the trees towards simplicity.** The best tuning trials used very small trees,
#    i.e. close to an additive model, and still did not catch up.
#
# What could change the result elsewhere: customer history (spend velocity per card, usual merchants),
# real timestamps, or genuine feature interactions; all are absent here (see notebook 01).

# %%
import json as _json  # noqa: E402

_eda = _json.loads((C.REPORTS_DIR / "p1_eda_summary.json").read_text())
_rf = _eda["red_flags"]
_logit = [np.log(r["fraud_rate"] / (1 - r["fraud_rate"])) for r in _rf]
_steps = np.diff(_logit)
_top_leaves = trials.head(5)["num_leaves"].astype(int).tolist() if bname == "lgbm" else None
_noskill = final_out.folds["base_rate"].mean()
_fin = metric_ci(final_out, "pr_auc")
_bst = metric_ci(boost_best, "pr_auc")
display(
    Markdown(f"""
**Evidence, computed:**

- Fraud rate by number of red flags (dev, P1): {", ".join(f"{r['level']}: {100 * r['fraud_rate']:.2f}%" for r in _rf)}.
  Each extra flag multiplies the odds by {", ".join(f"×{np.exp(x):.1f}" for x in _steps)}: large and
  same-signed, shrinking somewhat at the top (the last group is "4 or more" and small).
- Largest absolute Spearman correlation between numeric features: {_eda["max_abs_spearman"]["value"]:.2f}.
- `num_leaves` of the 5 best tuning trials: {_top_leaves}.

**PR-AUC in context.** A random score has PR-AUC equal to the fraud rate, about **{_noskill:.3f}**
here. Logistic regression reaches **{_fin["mean"]:.3f}** (95% CI {_fin["ci_low"]:.3f}–{_fin["ci_high"]:.3f}),
about **{_fin["mean"] / _noskill:.0f}×** that; the best boosting pipeline reaches {_bst["mean"]:.3f}
({_bst["ci_low"]:.3f}–{_bst["ci_high"]:.3f}). PR-AUC is far from 1 for every model: on these features,
a large share of fraud looks like ordinary traffic.
""")
)

# %% [markdown]
# ## Summary

# %%
fin = metric_ci(final_out, "pr_auc")
fin_roc = metric_ci(final_out, "roc_auc")
fin_brier = metric_ci(final_out, "brier")
base_ci = metric_ci(baseline, "pr_auc")
boost_ci = metric_ci(boost_best, "pr_auc")
d4 = next(d for d in decisions if d["step"].startswith("4."))
lines = "\n".join(
    f"  - {d['step']}: {verdict_text(d)} → `{d['chosen']}`" for d in decisions if "chosen" in d
)
display(
    Markdown(f"""
- **Final pipeline: `{final.name}`** — PR-AUC **{fin["mean"]:.3f}** (95% CI {fin["ci_low"]:.3f}–{fin["ci_high"]:.3f}),
  ROC-AUC {fin_roc["mean"]:.3f} ({fin_roc["ci_low"]:.3f}–{fin_roc["ci_high"]:.3f}),
  Brier {fin_brier["mean"]:.4f}.
- **Boosting vs logistic:** best boosting pipeline `{d4["b"]}` (PR-AUC {boost_ci["mean"]:.3f}) minus
  logistic `{d4["a"]}` (PR-AUC {base_ci["mean"]:.3f}) = {d4["mean_diff_b_minus_a"]:+.4f}
  (95% CI {d4["ci_low"]:+.4f} to {d4["ci_high"]:+.4f}) → {verdict_text(d4)}. The P1 red-flag
  pattern (each flag multiplies the odds) is exactly what a logistic model represents, which is a
  plausible reason trees add nothing here.
- **SMOTE** cut PR-AUC to {metric_ci(imb["smote"], "pr_auc")["mean"]:.3f}: synthetic fraud carried
  red flags only {ratio:.2f}× as often as real fraud (diagnostic above).
- **Decisions in pre-registered order:**
{lines}
- **`is_ai_generated_scam_attempt`:** with − without = {abl_test["mean_diff_b_minus_a"]:+.4f}
  (95% CI {abl_test["ci_low"]:+.4f} to {abl_test["ci_high"]:+.4f}); {verdict_text(abl_test)}.
- The tuned score was selected on repeat `r0`, so it is optimistic; the holdout (notebook 05)
  gives the unbiased number.
""")
)
