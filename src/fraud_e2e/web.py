"""Build the static project page (web/site/) from web/src/ and committed reports.

Every number on the page comes from `reports/*.json|csv`: prose numbers via
`fraud_e2e.summary.build_context()`, charts and the threshold simulator via the
data block built here. web/site/ is entirely generated; edit web/src/.

Languages: web/src/i18n/{en,th}.json hold the words only. `{{key}}` placeholders are filled
here from the results (so the dictionaries contain no hand-typed numbers), `{var}`
placeholders are filled in the browser. The page HTML is prerendered in English
(`[[key]]` markers in the template); web/site/i18n/<lang>.json lets the browser switch.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

from fraud_e2e import config as C
from fraud_e2e.cost import load_sensitivity_factors
from fraud_e2e.fairness import AGE_LABELS, ALPHA
from fraud_e2e.splits import N_FOLDS, N_REPEATS
from fraud_e2e.summary import LINK_LABELS, build_context, load_links, render

WEB = C.ROOT / "web"
SRC = WEB / "src"
SITE = WEB / "site"
LANGS = ("en", "th")
STATIC = (
    "style.css",
    "theme.js",
    "i18n.js",
    "app.js",
    "sim.js",
    "model.js",
    "flow.js",
    "viz.js",
    "favicon.svg",
    "vercel.json",
    "fonts/fraudthai-400.woff2",
    "fonts/fraudthai-500.woff2",
    "fonts/fraudthai-600.woff2",
    "fonts/OFL.txt",
)
TEMPLATE = "index.html.tmpl"
SHAP_TOP = 10
# UI constants shared with the browser (page data `ui`) and quoted in the text.
DOT_EVERY = 10  # "fraud + sampled legit" view: one legit dot in DOT_EVERY
MIN_LANE_PX = 2  # decision-flow lanes are at least this wide so small ones stay visible
AXIS_MIN, AXIS_MAX = "0.0001", "1"  # log score axis
JS_TOL = "1e-9"  # browser vs Python score tolerance (web/tests/model.test.mjs)
FN_FRAC_MAX = 2  # upper bound of fn_loss_fraction (fraud_e2e.cost.CostParams)
OFL = "SIL Open Font License 1.1"

# Pre-registered PR-AUC comparisons shown on the page: (decision step, dictionary key).
FOREST_ROWS = [
    ("2. imbalance: smote", "smote"),
    ("2. imbalance: class_weight", "class_weight"),
    ("2. imbalance: undersample", "undersample"),
    ("1. boosting library", "xgb"),
    ("3. tuning", "tuning"),
    ("4. boosting vs logistic", "boost_vs_lr"),
    ("5. ablation (reported only)", "ai_flag"),
]
VERDICT = {"b better": "better", "a better": "worse", "not distinguishable": "none"}
FAIR_COLS = ("stepup_rate", "recall", "precision", "fpr")


def _json(name: str) -> dict:
    return json.loads((C.REPORTS_DIR / f"{name}.json").read_text())


def _inline_md(text: str) -> str:
    """HTML-escape a context value and turn `code` spans into <code>."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(text, quote=True))


def tr(template: str, **values) -> str:
    """Fill {var} placeholders (the browser does the same in i18n.js)."""
    return re.sub(r"\{(\w+)\}", lambda m: str(values[m.group(1)]), template)


def load_dict(lang: str) -> dict[str, str]:
    return json.loads((SRC / "i18n" / f"{lang}.json").read_text())


# ----------------------------------------------------------------------------- data blocks


def sim_data() -> dict:
    """Simulator inputs: the web export of the CV cost curve, unchanged."""
    w = _json("p2_threshold_web")
    keys = ("threshold", "tp", "fp", "fn", "tn", "fn_amount_usd")
    n = {len(w[k]) for k in keys}
    if len(n) != 1:
        raise ValueError(f"p2_threshold_web.json arrays have different lengths: {n}")
    return {
        "source": "reports/p2_threshold_web.json",
        "note": w["note"],
        "cost_label": w["cost_label"],
        "n_rows": w["n_rows"],
        "n_fraud": w["n_fraud"],
        "fraud_amount_usd": round(w["fraud_amount_usd"], 2),
        "default_params": w["default_params"],
        "chosen_threshold": w["chosen_threshold"],
        **{k: w[k] for k in keys},
    }


def model_data() -> dict:
    m = _json("p4_web_model")
    return {
        "source": "reports/p4_web_model.json",
        "spec_id": m["spec_id"],
        "features": m["features"],
        "fold_models": m["fold_models"],
        "final_model": m["final_model"],
    }


def stream_data() -> dict:
    """The scored dev sample, column-wise; categoricals as codes into `levels`."""
    m = _json("p4_web_model")
    rows = pd.read_csv(C.REPORTS_DIR / "p4_web_stream.csv.gz")
    levels = {f: sorted(rows[f].astype(str).unique()) for f in m["categorical"]}
    cols = {}
    for f in m["features"]:
        if f in levels:
            code = {v: i for i, v in enumerate(levels[f])}
            cols[f] = [code[str(v)] for v in rows[f]]
        elif f in m["boolean"]:
            cols[f] = rows[f].astype(int).tolist()
        else:
            cols[f] = rows[f].tolist()
    return {
        "source": "reports/p4_web_stream.csv.gz",
        **m["stream"],
        "id": rows[C.ID_COL].astype(int).tolist(),
        "fold": rows["fold"].astype(int).tolist(),
        "y": rows[C.TARGET].astype(int).tolist(),
        "levels": levels,
        "cols": cols,
    }


def page_data() -> dict:
    return {
        "sim": sim_data(),
        "model": model_data(),
        "stream": stream_data(),
        "ui": {"dot_every": DOT_EVERY, "min_lane_px": MIN_LANE_PX, "fn_frac_max": FN_FRAC_MAX},
    }


# ----------------------------------------------------------------- language-specific HTML


def forest_rows(d: dict[str, str]) -> list[dict]:
    dec = {x["step"]: x for x in _json("p2_modeling")["decisions"]}
    tests = [(dec[step], key) for step, key in FOREST_ROWS]
    p5 = _json("p5_interactions")["variants"]  # P5 (pre-registration A6)
    tests += [(p5["spline"]["test"], "spline"), (p5["flag_pairs"]["test"], "flag_pairs")]
    rows = []
    for t, key in tests:
        if t["metric"] != "pr_auc":
            raise ValueError(f"{key}: expected a PR-AUC comparison")
        v = VERDICT[t["verdict"]]
        rows.append(
            {
                "label": d[f"frag.forest.{key}"],
                "diff": t["mean_diff_b_minus_a"],
                "lo": t["ci_low"],
                "hi": t["ci_high"],
                "verdict": v,
                "verdict_text": d[f"frag.verdict.{v}"],
            }
        )
    return rows


def _pos(x: float, lo: float, hi: float) -> str:
    return f"{100 * (x - lo) / (hi - lo):.2f}%"


def forest_html(rows: list[dict], d: dict[str, str], n_folds: int, ci: str) -> str:
    lo = min(r["lo"] for r in rows)
    hi = max(r["hi"] for r in rows)
    step = 0.05
    dlo = step * (int(lo / step) - (1 if lo < 0 else 0))
    dhi = step * (int(hi / step) + (1 if hi > 0 else 0))
    dlo, dhi = min(dlo, -step), max(dhi, step)
    items = []
    for r in rows:
        hollow = " dot--hollow" if r["verdict"] == "none" else ""
        items.append(
            f'<li><div class="row-head"><span class="label">{html.escape(r["label"])}</span>'
            f'<span class="value">{r["diff"]:+.3f} [{r["lo"]:+.3f}, {r["hi"]:+.3f}]'
            f'<span class="verdict verdict--{r["verdict"]}">{html.escape(r["verdict_text"])}'
            "</span></span></div>"
            f'<div class="plot" aria-hidden="true">'
            f'<span class="zero" style="left:{_pos(0, dlo, dhi)}"></span>'
            f'<span class="ci-bar" style="left:{_pos(r["lo"], dlo, dhi)};'
            f'width:{100 * (r["hi"] - r["lo"]) / (dhi - dlo):.2f}%"></span>'
            f'<span class="dot{hollow}" style="left:{_pos(r["diff"], dlo, dhi)}"></span></div></li>'
        )
    n_ticks = round((dhi - dlo) / step)
    ticks = [dlo + i * step for i in range(n_ticks + 1)]
    labelled = [t for t in ticks if abs(round(t / 0.1) * 0.1 - t) < 1e-9 or abs(t) < 1e-9]

    def tick(t: float) -> str:
        frac = (t - dlo) / (dhi - dlo)
        cls = ' class="end"' if frac > 0.98 else ' class="start"' if frac < 0.02 else ""
        text = "0" if abs(t) < 1e-9 else f"{t:+.1f}"
        return f'<span{cls} style="left:{_pos(t, dlo, dhi)}">{text}</span>'

    axis = "".join(tick(t) for t in labelled)
    sub = html.escape(tr(d["frag.forest.sub"], n=n_folds, ci=ci))
    return (
        '<figure class="forest-figure">'
        f"<figcaption><strong>{html.escape(d['frag.forest.caption'])}</strong> "
        f'<span class="muted">{sub}</span></figcaption>'
        f'<ul class="forest">{"".join(items)}</ul>'
        f'<div class="forest-ticks" aria-hidden="true">{axis}</div>'
        "</figure>"
    )


def fairness_html(d: dict[str, str], ci: str) -> str:
    """Per-age-band table (value + 95% CI) from reports/p5_fairness.json."""
    f = _json("p5_fairness")
    head = "".join(f'<th scope="col">{html.escape(d[f"frag.fair.{k}"])}</th>' for k in FAIR_COLS)
    body = ""
    for g in f["groups"]:
        cells = "".join(
            f'<td>{100 * g[k]:.1f}%<span class="ci">[{100 * g[k + "_ci_low"]:.1f}, '
            f"{100 * g[k + '_ci_high']:.1f}]</span></td>"
            for k in FAIR_COLS
        )
        body += (
            f'<tr><th scope="row">{html.escape(g["group"])}</th><td>{g["n"]:,}</td>'
            f"<td>{g['n_fraud']}</td>{cells}</tr>"
        )
    cap = html.escape(tr(d["frag.fair.caption"], ci=ci))
    th = (html.escape(d[f"frag.fair.{k}"]) for k in ("band", "rows", "fraud"))
    return (
        '<div class="table-scroll"><table class="results fair">'
        f"<caption>{cap}</caption><thead><tr>"
        + "".join(f'<th scope="col">{x}</th>' for x in th)
        + f"{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def shap_html(d: dict[str, str]) -> str:
    gi = _json("p2_shap")["global_importance"]
    top = gi[:SHAP_TOP]
    vmax = top[0]["mean_abs_shap"]
    bars = "".join(
        f'<li><span class="name">{html.escape(x["feature"])}</span><span class="track">'
        f'<span class="bar" style="width:{90 * x["mean_abs_shap"] / vmax:.2f}%"></span>'
        f'<span class="val">{x["mean_abs_shap"]:.3f}</span></span></li>'
        for x in top
    )
    rows = "".join(
        f"<tr><td>{i}</td><td><code>{html.escape(x['feature'])}</code></td>"
        f"<td>{x['mean_abs_shap']:.3f}</td></tr>"
        for i, x in enumerate(gi, 1)
    )
    cap = html.escape(tr(d["frag.shap.caption"], n_top=len(top), n_all=len(gi)))
    summary = html.escape(tr(d["frag.shap.summary"], n_all=len(gi)))
    return (
        f'<figure class="bars-figure"><figcaption><strong>{cap}</strong> '
        f'<span class="muted">{html.escape(d["frag.shap.unit"])}</span></figcaption>'
        f'<ul class="bars">{bars}</ul></figure>'
        f"<details><summary>{summary}</summary>"
        f"<table><thead><tr><th>#</th><th>{html.escape(d['frag.shap.th_feature'])}</th>"
        f"<th>{html.escape(d['frag.shap.th_value'])}</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></details>"
    )


def localized_ctx(d: dict[str, str]) -> dict[str, str]:
    """Phrases computed from the results whose words depend on the language.

    The English output equals fraud_e2e.summary.build_context() (tested), so README, model card
    and the English page say the same thing.
    """
    f = _json("p5_fairness")
    ia = _json("p5_interactions")["variants"]
    dec = _json("p2_modeling")["decisions"]

    def verdict(v: str) -> str:
        return d["ctx.verdict." + v.replace(" ", "_")]

    def metric(m: str) -> str:
        return d[f"ctx.metric.{m}"]

    hint = [x for x in f["gaps"] if x["ci_low"] > 0 or x["ci_high"] < 0]
    widest = max(f["gaps"], key=lambda x: x["ci_adj_high"] - x["ci_adj_low"])
    n_real, n_cmp = f["n_gaps_real"], f["n_comparisons"]
    same_way = bool(hint) and all((x["diff"] > 0) == (x["group"] in ("45–59", "60+")) for x in hint)
    cal = [x for x in dec if x["step"].startswith("6.")]
    return {
        "cost_label": d["ctx.cost_label"],
        "calibration_verdict": d["ctx.calibration.none"]
        if all(x["verdict"] == "not distinguishable" for x in cal)
        else tr(d["ctx.calibration.changed"], nb="04"),
        "fair_verdict": verdict("not distinguishable")
        if n_real == 0
        else tr(d["ctx.fair_verdict.some"], n=n_real),
        "fair_headline": d["ctx.fair_headline.none"]
        if n_real == 0
        else tr(d["ctx.fair_headline.some"], n=n_real, m=n_cmp),
        "fair_direction": d["ctx.fair_direction.same" if same_way else "ctx.fair_direction.mixed"],
        "fair_hint": "; ".join(
            tr(
                d["ctx.fair_hint_item"],
                group=x["group"],
                metric=metric(x["metric"]),
                diff=f"{100 * x['diff']:+.2f}",
                ci=f"[{100 * x['ci_low']:+.2f}, {100 * x['ci_high']:+.2f}]",
            )
            for x in hint
        )
        or d["ctx.fair_hint_none"],
        "fair_widest": tr(
            d["ctx.fair_widest"],
            width=f"{100 * (widest['ci_adj_high'] - widest['ci_adj_low']):.1f}",
            group=widest["group"],
            metric=metric(widest["metric"]),
        ),
        "noage_verdict": verdict(f["no_age"]["test"]["verdict"]),
        "gam_verdict": verdict(ia["spline"]["test"]["verdict"]),
        "pairs_verdict": verdict(ia["flag_pairs"]["test"]["verdict"]),
        "age_shap_rank": tr(
            d["ctx.rank_of"], rank=f["customer_age_shap"]["rank"], of=f["customer_age_shap"]["of"]
        ),
    }


def text_constants() -> dict[str, str]:
    """Numbers quoted in the text that are settings, not results (still never typed in i18n)."""
    sens = load_sensitivity_factors()
    n_cv = {x["n_folds"] for x in _json("p2_modeling")["decisions"]}
    if len(n_cv) != 1:
        raise ValueError(f"decisions use different fold counts: {n_cv}")
    return {
        "ci_level": f"{100 * (1 - ALPHA):.0f}%",
        "per_n": "1,000",
        "axis_min": AXIS_MIN,
        "axis_max": AXIS_MAX,
        "n_cv_folds": str(n_cv.pop()),
        "n_folds_k": str(N_FOLDS),
        "n_repeats": str(N_REPEATS),
        "dot_every": str(DOT_EVERY),
        "min_lane_px": str(MIN_LANE_PX),
        "js_tol": JS_TOL,
        "band_young": AGE_LABELS[0],
        "band_old": AGE_LABELS[-1],
        "fn_frac_max": str(FN_FRAC_MAX),
        "sens_lo": f"{min(sens):g}",
        "sens_hi": f"{max(sens):g}",
        "ofl": OFL,
    }


def links_html(links: dict[str, str]) -> str:
    shown = [(k, u) for k, u in links.items() if u and k != "website"]
    if not shown:
        return "<!-- links: none set yet (configs/links.yaml) -->"
    items = "".join(
        f'<li><a href="{html.escape(u, quote=True)}" rel="noopener">{LINK_LABELS[k]}</a></li>'
        for k, u in shown
    )
    return f'<ul class="links" aria-label="Project links">{items}</ul>'


def canonical_html(links: dict[str, str]) -> str:
    u = links.get("website")
    if not u:
        return "<!-- canonical URL: set `website` in configs/links.yaml -->"
    u = html.escape(u, quote=True)
    return f'<link rel="canonical" href="{u}">\n<meta property="og:url" content="{u}">'


def csp_meta() -> str:
    """The CSP from vercel.json as a <meta> tag (GitHub Pages cannot set response headers).

    `frame-ancestors` is dropped: browsers ignore it in a meta tag.
    """
    cfg = json.loads((SRC / "vercel.json").read_text())
    policy = next(
        h["value"]
        for rule in cfg["headers"]
        for h in rule["headers"]
        if h["key"] == "Content-Security-Policy"
    )
    kept = [
        d.strip()
        for d in policy.split(";")
        if d.strip() and not d.strip().startswith("frame-ancestors")
    ]
    value = html.escape("; ".join(kept), quote=False).replace('"', "&quot;")
    return f'<meta http-equiv="Content-Security-Policy" content="{value}">'


def _data_json(data: dict) -> str:
    text = json.dumps(data, separators=(",", ":"), sort_keys=True, allow_nan=False)
    return text.replace("</", "<\\/")


# ----------------------------------------------------------------------------- build


def base_context() -> dict[str, str]:
    ctx = {k: _inline_md(v) for k, v in build_context().items()}
    stream = _json("p4_web_model")["stream"]
    ctx |= {
        "data_sha_short": _json("p2_threshold_web")["_meta"]["data_sha256"][:12],
        "stream_n": f"{stream['n_rows']:,}",
        "stream_fraud": f"{stream['n_fraud']}",
    }
    return ctx | {k: html.escape(v) for k, v in text_constants().items()}


def dictionaries() -> dict[str, dict[str, str]]:
    """Rendered dictionaries per language: {lang: {key: text with numbers filled in}}."""
    base = base_context()
    consts = text_constants()
    out = {}
    for lang in LANGS:
        d = load_dict(lang)
        ctx = base | {k: _inline_md(v) for k, v in localized_ctx(d).items()}
        ctx |= {
            "shap_html": shap_html(d),
            "forest_html": forest_html(forest_rows(d), d, consts["n_cv_folds"], consts["ci_level"]),
            "fairness_html": fairness_html(d, consts["ci_level"]),
        }
        out[lang] = {k: render(v, ctx) for k, v in d.items() if not k.startswith(("frag.", "ctx."))}
    return out


_MARK = re.compile(r"\[\[([\w.]+)\]\]")


def build_site() -> dict[str, bytes]:
    """All files of web/site/ as {relative path: bytes}. Pure; writes nothing."""
    links = load_links()
    dicts = dictionaries()
    ctx = base_context() | {
        "links_html": links_html(links),
        "canonical_html": canonical_html(links),
        "csp_meta": csp_meta(),
        "data_json": _data_json(page_data()),
    }
    page = render((SRC / TEMPLATE).read_text(), ctx)
    missing = sorted({k for k in _MARK.findall(page) if k not in dicts["en"]})
    if missing:
        raise KeyError(f"template uses keys missing from en.json: {missing}")
    page = _MARK.sub(lambda m: dicts["en"][m.group(1)], page)  # prerendered in English
    files = {"index.html": page.encode()}
    for lang, d in dicts.items():
        files[f"i18n/{lang}.json"] = (json.dumps(d, ensure_ascii=False, indent=0) + "\n").encode()
    for name in STATIC:
        files[name] = (SRC / name).read_bytes()
    return files


def stale_files(site: Path = SITE) -> list[str]:
    """Paths in web/site/ that differ from a fresh build (missing, changed or extra)."""
    want = build_site()
    have = {
        p.relative_to(site).as_posix()
        for p in site.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(site).parts)
    }  # hidden paths (e.g. .vercel/ written by the Vercel CLI) are not part of the site
    out = [n for n, b in want.items() if not (site / n).exists() or (site / n).read_bytes() != b]
    out += sorted(have - set(want))
    return out


def write_site(site: Path = SITE) -> list[str]:
    files = build_site()
    for name, content in files.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return sorted(files)
