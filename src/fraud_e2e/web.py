"""Build the static project page (web/site/) from web/src/ and committed reports.

Every number on the page comes from `reports/*.json|csv`: prose numbers via
`fraud_e2e.summary.build_context()`, charts and the threshold simulator via the
data block built here. web/site/ is entirely generated; edit web/src/.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

from fraud_e2e import config as C
from fraud_e2e.summary import LINK_LABELS, build_context, load_links, render

WEB = C.ROOT / "web"
SRC = WEB / "src"
SITE = WEB / "site"
STATIC = (
    "style.css",
    "theme.js",
    "app.js",
    "sim.js",
    "model.js",
    "flow.js",
    "viz.js",
    "favicon.svg",
    "vercel.json",
)
TEMPLATE = "index.html.tmpl"
SHAP_TOP = 10

# Pre-registered PR-AUC comparisons shown on the page: (decision step, label for "B vs A").
FOREST_ROWS = [
    ("2. imbalance: smote", "SMOTE vs no resampling (LightGBM)"),
    ("2. imbalance: class_weight", "Class weights vs no resampling (LightGBM)"),
    ("2. imbalance: undersample", "Undersampling vs no resampling (LightGBM)"),
    ("1. boosting library", "XGBoost vs LightGBM"),
    ("3. tuning", "Tuned vs default LightGBM"),
    ("4. boosting vs logistic", "Tuned LightGBM vs logistic regression"),
    ("5. ablation (reported only)", "Logistic with vs without the AI-scam flag"),
]
VERDICT = {"b better": "better", "a better": "worse", "not distinguishable": "no clear difference"}


def _json(name: str) -> dict:
    return json.loads((C.REPORTS_DIR / f"{name}.json").read_text())


def _inline_md(text: str) -> str:
    """HTML-escape a context value and turn `code` spans into <code>."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(text, quote=True))


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


def forest_rows() -> list[dict]:
    dec = {d["step"]: d for d in _json("p2_modeling")["decisions"]}
    rows = []
    for step, label in FOREST_ROWS:
        d = dec[step]
        if d["metric"] != "pr_auc":
            raise ValueError(f"{step}: expected a PR-AUC comparison")
        rows.append(
            {
                "label": label,
                "diff": d["mean_diff_b_minus_a"],
                "lo": d["ci_low"],
                "hi": d["ci_high"],
                "verdict": VERDICT[d["verdict"]],
            }
        )
    return rows


def _pos(x: float, lo: float, hi: float) -> str:
    return f"{100 * (x - lo) / (hi - lo):.2f}%"


def forest_html(rows: list[dict]) -> str:
    lo = min(r["lo"] for r in rows)
    hi = max(r["hi"] for r in rows)
    step = 0.05
    dlo = step * (int(lo / step) - (1 if lo < 0 else 0))
    dhi = step * (int(hi / step) + (1 if hi > 0 else 0))
    dlo, dhi = min(dlo, -step), max(dhi, step)
    items = []
    for r in rows:
        hollow = " dot--hollow" if r["verdict"] == "no clear difference" else ""
        vclass = {"worse": "worse", "better": "better"}.get(r["verdict"], "none")
        items.append(
            f'<li><div class="row-head"><span class="label">{html.escape(r["label"])}</span>'
            f'<span class="value">{r["diff"]:+.3f} [{r["lo"]:+.3f}, {r["hi"]:+.3f}]'
            f'<span class="verdict verdict--{vclass}">{r["verdict"]}</span></span></div>'
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
    return (
        '<figure class="forest-figure">'
        "<figcaption><strong>Difference in PR-AUC, first option minus second</strong> "
        '<span class="muted">(paired CV, 15 folds; bar = 95% CI; hollow dot = CI includes 0)</span>'
        "</figcaption>"
        f'<ul class="forest">{"".join(items)}</ul>'
        f'<div class="forest-ticks" aria-hidden="true">{axis}</div>'
        "</figure>"
    )


def shap_html() -> str:
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
    return (
        '<figure class="bars-figure"><figcaption><strong>Top '
        f"{len(top)} of {len(gi)} features by mean |SHAP|</strong> "
        '<span class="muted">(log-odds)</span></figcaption>'
        f'<ul class="bars">{bars}</ul></figure>'
        f"<details><summary>All {len(gi)} features as a table</summary>"
        "<table><thead><tr><th>#</th><th>Feature</th><th>Mean |SHAP| (log-odds)</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></details>"
    )


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


def page_data() -> dict:
    return {"sim": sim_data(), "model": model_data(), "stream": stream_data()}


def _data_json(data: dict) -> str:
    text = json.dumps(data, separators=(",", ":"), sort_keys=True, allow_nan=False)
    return text.replace("</", "<\\/")


def build_site() -> dict[str, bytes]:
    """All files of web/site/ as {relative path: bytes}. Pure; writes nothing."""
    ctx = {k: _inline_md(v) for k, v in build_context().items()}
    links = load_links()
    meta = _json("p2_threshold_web")["_meta"]
    stream = _json("p4_web_model")["stream"]
    ctx |= {
        "links_html": links_html(links),
        "canonical_html": canonical_html(links),
        "csp_meta": csp_meta(),
        "shap_html": shap_html(),
        "forest_html": forest_html(forest_rows()),
        "data_json": _data_json(page_data()),
        "data_sha_short": meta["data_sha256"][:12],
        "stream_n": f"{stream['n_rows']:,}",
        "stream_fraud": f"{stream['n_fraud']}",
    }
    files = {"index.html": render((SRC / TEMPLATE).read_text(), ctx).encode()}
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
    site.mkdir(parents=True, exist_ok=True)
    files = build_site()
    for name, content in files.items():
        (site / name).write_bytes(content)
    return sorted(files)
