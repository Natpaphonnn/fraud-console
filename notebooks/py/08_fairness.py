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
# # 08 · Fairness by age
#
# **Questions this notebook answers**
#
# 1. At the project threshold, are customers of different ages stepped up at different rates? Is fraud
#    caught at different rates? Are good customers bothered (false positives) at different rates?
# 2. Which of those gaps are clear enough to call real, and which cannot be told apart from noise?
# 3. Does the model need `customer_age` at all?
#
# **Method, fixed before any result** (`notes/00_preregistration.md`, amendment A5):
#
# - Age bands 18–29, 30–44, 45–59, 60+ (chosen from a label-free look at the age distribution).
# - Dev out-of-fold scores of the final pipeline from CV repeat r0, so every dev row is scored once by a
#   model that did not train on it. **The holdout is not used.** One threshold for everyone: the project's.
# - Per band: step-up rate, recall, precision, false-positive rate (FPR), each with a stratified bootstrap
#   95% interval, plus the number of fraud cases. With so few fraud cases per band, intervals are wide.
# - Each band is compared with all other dev rows. 4 bands × 4 metrics = 16 comparisons, so a gap counts as
#   **real only if its Bonferroni-adjusted interval excludes 0**. Otherwise it is "not distinguishable". That
#   is **not** evidence of fairness.
# - The model without `customer_age` is compared on the same 15 folds (paired, corrected t-test on PR-AUC).
#
# No causal or legal claim is made. Age is the only person attribute in the data, and the data's
# provenance is unspecified.

# %%
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from fraud_e2e import config as C
from fraud_e2e import plots as P
from fraud_e2e.cv import corrected_ttest, load_result, metric_ci, run_cv, save_result
from fraud_e2e.data import load_raw
from fraud_e2e.fairness import (
    AGE_LABELS,
    ALPHA_ADJ,
    METRICS,
    N_BOOT,
    N_COMPARISONS,
    age_group,
    group_gaps,
    group_metrics,
    overall_rates,
)
from fraud_e2e.reporting import read_json, write_csv, write_json
from fraud_e2e.splits import dev_only
from fraud_e2e.stats import wilson_ci

P.set_style()
dev = dev_only(load_raw())
final = load_result("final")
thr = read_json("p2_threshold")["threshold"]
oof = final.oof.merge(dev[[C.ID_COL, "customer_age"]], on=C.ID_COL, validate="many_to_one")
oof["age_group"] = age_group(oof["customer_age"]).to_numpy()
r0 = oof[oof["repeat"] == 0].reset_index(drop=True)
assert len(r0) == len(dev) and r0[C.ID_COL].is_unique
print(f"final pipeline: {final.spec['model']}; threshold {thr:.4f}; dev rows (r0 OOF): {len(r0):,}")

LABEL = {
    "stepup_rate": "Step-up rate",
    "recall": "Recall",
    "precision": "Precision",
    "fpr": "False-positive rate",
}

# %% [markdown]
# ## 1. How many rows and fraud cases per band?

# %%
sizes = r0.groupby("age_group")["y"].agg(n="size", n_fraud="sum").reindex(AGE_LABELS).reset_index()
lo, hi = wilson_ci(sizes["n_fraud"], sizes["n"])
sizes["fraud_rate_%"] = 100 * sizes["n_fraud"] / sizes["n"]
sizes["ci95_low_%"], sizes["ci95_high_%"] = 100 * lo, 100 * hi
sizes.round(2)

# %% [markdown]
# ## 2. Metrics per band at the project threshold (95% bootstrap intervals)

# %%
gm = group_metrics(r0["y"], r0["score"], r0["age_group"], thr)
overall = overall_rates(r0["y"], r0["score"], thr)
show = gm[["group", "n", "n_fraud", "n_flagged"]].copy()
for k in METRICS:
    show[LABEL[k]] = [
        f"{100 * v:.1f}% [{100 * a:.1f}, {100 * b:.1f}]"
        for v, a, b in zip(gm[k], gm[f"{k}_ci_low"], gm[f"{k}_ci_high"], strict=True)
    ]
show

# %%
fig, axes = P.plt.subplots(1, 4, figsize=(12, 3.3), sharey=True)
ypos = np.arange(len(AGE_LABELS))[::-1]
for ax, k in zip(axes, METRICS, strict=True):
    v, a, b = 100 * gm[k], 100 * gm[f"{k}_ci_low"], 100 * gm[f"{k}_ci_high"]
    ax.errorbar(
        v, ypos, xerr=[v - a, b - v], fmt="o", color=P.INK, ecolor=P.INK_2, capsize=0, lw=1.5
    )
    ax.axvline(100 * overall[k], color=P.SERIES[1], ls="--", lw=1, label="all dev rows")
    ax.set_title(LABEL[k])
    ax.set_xlabel("% (95% interval)")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(
    ypos, [f"{g}\n({n} fraud)" for g, n in zip(gm["group"], gm["n_fraud"], strict=True)]
)
axes[0].set_ylabel("Age band")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=9
)
fig.suptitle(
    f"Outcomes by age band at threshold {thr:.4f} (dev, out-of-fold, CV repeat r0)", y=1.04
)
P.save(fig, "p5_fairness")
P.plt.show()

# %% [markdown]
# ## 3. Is any band different from the rest? (pre-registered rule)
#
# Difference = band − all other dev rows. The verdict uses the Bonferroni interval
# (1 − 0.05/16 ≈ 99.69%); the 95% interval is shown for reference.

# %%
gaps = group_gaps(r0["y"], r0["score"], r0["age_group"], thr)
gtab = gaps.assign(
    metric=gaps["metric"].map(LABEL),
    diff_pp=100 * gaps["diff"],
    ci95_pp=[
        f"[{100 * a:+.1f}, {100 * b:+.1f}]"
        for a, b in zip(gaps["ci_low"], gaps["ci_high"], strict=True)
    ],
    ci_adj_pp=[
        f"[{100 * a:+.1f}, {100 * b:+.1f}]"
        for a, b in zip(gaps["ci_adj_low"], gaps["ci_adj_high"], strict=True)
    ],
)[["group", "metric", "diff_pp", "ci95_pp", "ci_adj_pp", "verdict"]]
write_csv(gaps, "p5_fairness_gaps")
gtab.round(2)

# %%
real = gaps[gaps["verdict"] != "not distinguishable"]
# reported for transparency: gaps whose plain 95% interval excludes 0 (NOT the pre-registered rule)
hint = gaps[(gaps["ci_low"] > 0) | (gaps["ci_high"] < 0)]
widest = gaps.assign(width=gaps["ci_adj_high"] - gaps["ci_adj_low"]).sort_values("width").iloc[-1]
print(f"gaps called real: {len(real)} of {N_COMPARISONS}")
print(f"gaps whose unadjusted 95% interval excludes 0: {len(hint)}")
print(
    f"widest adjusted interval: {widest['group']} {widest['metric']} ({100 * widest['width']:.1f} pp)"
)

# %% [markdown]
# ## 4. Stability: the same metrics on CV repeats r1 and r2 (point estimates)

# %%
stab = []
for r in (0, 1, 2):
    sub = oof[oof["repeat"] == r]
    t = group_metrics(sub["y"], sub["score"], sub["age_group"], thr, n_boot=1)
    stab.append(t[["group", *METRICS]].assign(repeat=f"r{r}"))
stab = pd.concat(stab)
stab_wide = stab.pivot(index="group", columns="repeat", values=METRICS).reindex(AGE_LABELS)
(100 * stab_wide).round(1)

# %% [markdown]
# ## 5. Does the model need `customer_age`?

# %%
no_age = run_cv(dev, {**final.spec, "drop": ["customer_age"]}, "logreg_no_age")
save_result(no_age, with_oof=False)
test = corrected_ttest(final, no_age, "pr_auc")
na_r0 = no_age.oof[no_age.oof["repeat"] == 0].merge(r0[[C.ID_COL, "age_group"]], on=C.ID_COL)
gm_na = group_metrics(na_r0["y"], na_r0["score"], na_r0["age_group"], thr)
cmp_age = pd.DataFrame(
    {
        "group": AGE_LABELS,
        "step-up % (with age)": 100 * gm["stepup_rate"],
        "step-up % (without)": 100 * gm_na["stepup_rate"],
        "FPR % (with age)": 100 * gm["fpr"],
        "FPR % (without)": 100 * gm_na["fpr"],
    }
)
print(
    f"PR-AUC without age minus with age: {test['mean_diff_b_minus_a']:+.3f} "
    f"[{test['ci_low']:+.3f}, {test['ci_high']:+.3f}] -> {test['verdict']}"
)
cmp_age.round(2)

# %%
shap = read_json("p2_shap")["global_importance"]
age_rank = next(i for i, x in enumerate(shap, 1) if x["feature"] == "customer_age")
age_share = next(x["mean_abs_shap"] for x in shap if x["feature"] == "customer_age") / sum(
    x["mean_abs_shap"] for x in shap
)
print(
    f"customer_age: SHAP rank {age_rank} of {len(shap)}, {100 * age_share:.1f}% of total mean |SHAP|"
)

# %%
saved = write_json(
    {
        "preregistration": "notes/00_preregistration.md#A5",
        "data": "dev out-of-fold scores of the final pipeline, CV repeat r0; holdout not used",
        "threshold": thr,
        "age_bands": AGE_LABELS,
        "n_boot": N_BOOT,
        "n_comparisons": N_COMPARISONS,
        "alpha_adjusted": ALPHA_ADJ,
        "overall": overall,
        "sizes": sizes.to_dict(orient="records"),
        "groups": gm.to_dict(orient="records"),
        "gaps": gaps.to_dict(orient="records"),
        "n_gaps_real": int(len(real)),
        "n_gaps_95_excluding_0": int(len(hint)),
        "stability": stab.to_dict(orient="records"),
        "no_age": {
            "test": test,
            "pr_auc_with_age": metric_ci(final, "pr_auc"),
            "pr_auc_without_age": metric_ci(no_age, "pr_auc"),
            "groups": gm_na.to_dict(orient="records"),
        },
        "customer_age_shap": {"rank": age_rank, "of": len(shap), "share": age_share},
    },
    "p5_fairness",
)
print("saved", saved.relative_to(C.ROOT))

# %% [markdown]
# ## Summary


# %%
def pp(x):
    return f"{100 * x:.1f}%"


def band_line(row):
    return (
        f"| {row['group']} | {row['n']:,} | {row['n_fraud']} | {pp(row['stepup_rate'])} | "
        f"{pp(row['recall'])} [{pp(row['recall_ci_low'])}, {pp(row['recall_ci_high'])}] | "
        f"{pp(row['fpr'])} [{pp(row['fpr_ci_low'])}, {pp(row['fpr_ci_high'])}] |"
    )


def gap_txt(r, lo="ci_low", hi="ci_high"):
    return (
        f"{r['group']} {LABEL[r['metric']].lower()} {100 * r['diff']:+.2f} pp "
        f"[{100 * r[lo]:+.2f}, {100 * r[hi]:+.2f}]"
    )


if len(real):
    verdict_txt = (
        f"{len(real)} of {N_COMPARISONS} gaps pass the pre-registered rule (Bonferroni interval): "
        + "; ".join(gap_txt(r, "ci_adj_low", "ci_adj_high") for _, r in real.iterrows())
        + "."
    )
else:
    verdict_txt = (
        f"None of the {N_COMPARISONS} gaps passes the pre-registered rule, so all are **not "
        "distinguishable** with this much data. That is not evidence that the model is fair."
    )
if len(hint):
    hint_txt = (
        f"Without the multiple-comparison correction, {len(hint)} gaps have a 95% interval that excludes 0: "
        + "; ".join(gap_txt(r) for _, r in hint.iterrows())
        + ". "
        + (
            "They point the same way (older bands above the rest, younger bands below), but"
            if all((r["diff"] > 0) == (r["group"] in ("45–59", "60+")) for _, r in hint.iterrows())
            else "They"
        )
        + " do not pass the pre-registered rule: a lead to re-check on more data, not a finding."
    )
else:
    hint_txt = "No gap excludes 0 even at the unadjusted 95% level."

band = sizes.set_index("age_group")
su = gm.set_index("group")
sn = gm_na.set_index("group")
age_effect_txt = (
    f"Removing age changes who is stepped up: 18–29 goes from {pp(su.loc['18–29', 'stepup_rate'])} to "
    f"{pp(sn.loc['18–29', 'stepup_rate'])}, 60+ from {pp(su.loc['60+', 'stepup_rate'])} to "
    f"{pp(sn.loc['60+', 'stepup_rate'])} (descriptive). Part of the age pattern follows the base rate: "
    f"fraud is {band.loc['60+', 'fraud_rate_%']:.2f}% of 60+ rows vs {band.loc['18–29', 'fraud_rate_%']:.2f}% "
    "of 18–29 rows on dev."
)

display(
    Markdown(f"""
**Answers** (dev, out-of-fold, threshold {thr:.4f}; 95% bootstrap intervals; holdout not used)

| Age band | Rows | Fraud | Step-up rate | Recall | False-positive rate |
|---|---|---|---|---|---|
{chr(10).join(band_line(r) for _, r in gm.iterrows())}

1. **Per band.** Fraud cases per band range from {gm["n_fraud"].min()} to {gm["n_fraud"].max()}, so recall
   intervals are wide (up to {100 * (gm["recall_ci_high"] - gm["recall_ci_low"]).max():.0f} pp).
2. **Gaps.** {verdict_txt} The widest Bonferroni interval spans {100 * widest["width"]:.1f} pp
   ({widest["group"]}, {LABEL[widest["metric"]].lower()}). {hint_txt}
3. **Without `customer_age`.** PR-AUC changes by {test["mean_diff_b_minus_a"]:+.3f}
   [{test["ci_low"]:+.3f}, {test["ci_high"]:+.3f}] (paired, 15 folds): **{test["verdict"]}**.
   Age ranks {age_rank} of {len(shap)} features by mean |SHAP| ({100 * age_share:.1f}% of the total).
   {age_effect_txt} The final model is unchanged (the holdout has been used); dropping age costs no
   distinguishable PR-AUC and is an option where using age is not acceptable.
""")
)
