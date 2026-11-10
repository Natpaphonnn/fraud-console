"""Content figures and short looping clips from the real model and committed results.

    python scripts/content_figures.py --quick              # previews -> reports/content/preview/
    python scripts/content_figures.py --full [--only NAME] # stills + clips -> reports/content/
                                                           # (GitHub Actions only: content.yml)
    python scripts/content_figures.py --annotated [--only NAME]  # annotated clips (Actions only)
    python scripts/content_figures.py --annotated-preview  # a few low-res annotated frames

Figures (every surface or point comes from the exported model or from reports/, nothing invented):
  landscape          P(fraud) over merchant_risk_score x velocity_score (linear 0-1: S-shaped cliff)
  landscape_logodds  the same slice in log-odds: a flat plane (still only)
  cost_valley        expected cost per 1,000 transactions over threshold x FN-loss multiplier
  galaxy             dev rows, PCA of their per-feature contributions to the model's log-odds

Each still comes in two variants, "full" (title + explanation + credit) and "social" (title +
credit); the full explanation is also written to <name>_caption.txt. Clips (~7 s, 30 fps, seamless
loop, no audio): mp4 1080x1080 and 1080x1920 plus a downsized gif (<= 8 MB).

Annotated clips (<name>_annotated_<size>.mp4 + <name>_annotated.gif, a separate set; the clean
clips are unchanged): short captions fade in one after another (each held ~2 s) with an arrow to
the point they talk about, in FraudMono (the website's monospace font, from web/src/fonts/), inside
the 9:16 safe area. Every number in a caption comes from reports/*.json.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # stay light on shared machines

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import animation, font_manager  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from fraud_e2e import config as C  # noqa: E402
from fraud_e2e.cost import COST_LABEL  # noqa: E402
from fraud_e2e.data import load_raw  # noqa: E402
from fraud_e2e.export import linear_contributions, linear_scores  # noqa: E402
from fraud_e2e.features import feature_frame  # noqa: E402
from fraud_e2e.splits import dev_only  # noqa: E402
from fraud_e2e.summary import load_links  # noqa: E402

OUT = C.REPORTS_DIR / "content"
BG, INK, MUTED, GRID, ACCENT = "#0a0c0f", "#e7eaee", "#8a93a0", "#2a313b", "#ffb020"
LEGIT = "#7fa6d6"
SOURCE = "Data: Credit Card Fraud Detection 2026 (Udit Jain, CC0), unspecified provenance."
STILL_SIZES = {"1080x1080": (1080, 1080), "1920x1080": (1920, 1080)}
CLIP_SIZES = {"1080x1080": (1080, 1080), "1080x1920": (1080, 1920)}
CLIP_SECONDS, CLIP_FPS = 7, 30
GIF_MAX_BYTES = 8 * 1024 * 1024
LANDSCAPE_FLAGS = 3  # red flags switched on: the 3 most common among dev fraud


def _json(name: str) -> dict:
    return json.loads((C.REPORTS_DIR / f"{name}.json").read_text())


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ----------------------------------------------------------------------------- data


def model() -> dict:
    return _json("p4_web_model")["final_model"]


def landscape_data(dev: pd.DataFrame, n: int = 80) -> dict:
    """Typical dev transaction (numeric medians, most common categories) with the red flags most
    common among dev fraud switched on, so the slice shows both plateaus and the cliff."""
    base = {c: float(dev[c].median()) for c in C.NUMERIC_COLS}
    base |= {c: dev[c].mode()[0] for c in C.CATEGORICAL_COLS + C.BOOLEAN_COLS}
    fraud = dev[dev[C.TARGET] == 1]
    share = fraud[C.BOOLEAN_COLS].mean().sort_values(ascending=False)
    flags_on = list(share.index[:LANDSCAPE_FLAGS])
    for f in C.BOOLEAN_COLS:
        base[f] = f in flags_on
    xs = np.linspace(dev["merchant_risk_score"].min(), dev["merchant_risk_score"].max(), n)
    ys = np.linspace(dev["velocity_score"].min(), dev["velocity_score"].max(), n)
    X, Y = np.meshgrid(xs, ys)
    grid = pd.DataFrame([base] * X.size)
    grid["merchant_risk_score"] = X.ravel()
    grid["velocity_score"] = Y.ravel()
    P = linear_scores(model(), feature_frame(grid)).reshape(X.shape)
    fixed = (
        f"amount {base['amount_usd']:.2f} USD, age {base['customer_age']:.0f}, "
        f"{base['merchant_category']}, {base['channel']}, {base['auth_method']},\n"
        f"other inputs at dev medians; red flags on: {', '.join(flags_on)} (others off)"
    )
    return {
        "X": X,
        "Y": Y,
        "P": P,
        "L": np.log(P / (1 - P)),
        "thr": _json("p2_threshold")["threshold"],
        "fixed": fixed,
        "flags_on": flags_on,
    }


def cost_data(n_mult: int = 60) -> dict:
    w = _json("p2_threshold_web")
    p = w["default_params"]
    t = np.asarray(w["threshold"])
    keep = t >= 3e-3  # below this almost everything is stepped up; the valley is the story
    tp, fp, fn, fa = (np.asarray(w[k])[keep] for k in ("tp", "fp", "fn", "fn_amount_usd"))
    t = t[keep]
    mult = np.unique(np.r_[np.linspace(0.25, 2.0, n_mult), 1.0])  # include x1 exactly
    per1000 = 1000 / w["n_rows"]
    base = (tp + fp) * p["review_cost_per_alert"] + fp * p["fp_friction_cost"]
    lost = [k * p["fn_loss_fraction"] * fa + fn * p["fn_fixed_cost"] for k in mult]
    Z = (base + np.stack(lost)) * per1000
    X, Y = np.meshgrid(np.log10(t), mult)
    best = Z.argmin(axis=1)
    i1 = int(np.flatnonzero(mult == 1.0)[0])
    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "valley": (np.log10(t[best]), mult, Z[np.arange(len(mult)), best]),
        "project": (np.log10(t[best[i1]]), mult[i1], Z[i1, best[i1]], t[best[i1]]),
        "curve1": (np.log10(t), Z[i1]),  # the x1 cost curve (the project's assumption)
        "t_range": (t.min(), t.max()),
        "params": p,
    }


def galaxy_data(dev: pd.DataFrame) -> dict:
    contrib = linear_contributions(model(), feature_frame(dev))
    pca = PCA(n_components=3, random_state=C.SEED)
    y = dev[C.TARGET].to_numpy()
    return {
        "P": pca.fit_transform(contrib.to_numpy()),
        "y": y,
        "var": float(pca.explained_variance_ratio_.sum()),
        "n_features": contrib.shape[1],
        "phase": np.random.default_rng(C.SEED).random(int(y.sum())),  # twinkle (animation only)
    }


# ----------------------------------------------------------------------------- text


def texts(name: str, d: dict) -> dict:
    """Title, full explanation and credit line (every number from the data)."""
    if name == "landscape":
        return {
            "title": "Fraud risk landscape",
            "body": "P(fraud) from the logistic model as merchant risk and velocity vary.\n"
            f"Fixed at a typical dev transaction: {d['fixed']}.\n"
            f"Amber contour: the step-up threshold, P = {d['thr']:.4f}. "
            f"P on this slice: {d['P'].min():.3f} to {d['P'].max():.3f}.",
            "credit": SOURCE,
        }
    if name == "landscape_logodds":
        return {
            "title": "Log-odds view: a flat plane",
            "body": "The same slice as the fraud risk landscape, vertical axis in log-odds.\n"
            "A logistic model is exactly a plane here: each input adds a fixed amount.\n"
            "That additive shape is what beat LightGBM on this dataset.",
            "credit": SOURCE,
        }
    if name == "cost_valley":
        p = d["params"]
        _, _, pz, pt = d["project"]
        return {
            "title": "Cost valley",
            "body": "Expected cost per 1,000 transactions by step-up threshold and by how much\n"
            "of a "
            "missed fraud's amount is lost (×0.25 to ×2 of the project's assumption).\n"
            f"Dev out-of-fold counts, mean of 3 CV repeats; thresholds {d['t_range'][0]:.3f} to "
            f"{d['t_range'][1]:.0f}. Glowing line: cheapest threshold per multiplier.\n"
            f"Dot: the project's setting (×1), threshold {pt:.4f}, {pz:,.0f} USD per 1,000.",
            "credit": f"{COST_LABEL}: step-up {p['review_cost_per_alert']:g} USD, good-customer "
            f"friction {p['fp_friction_cost']:g} USD,\nmissed fraud = amount + "
            f"{p['fn_fixed_cost']:g} USD. {SOURCE}",
        }
    y = d["y"]
    return {
        "title": "Transaction galaxy",
        "body": f"{len(y):,} dev transactions. Each point is one transaction's {d['n_features']} "
        "per-feature\ncontributions to the logistic model's log-odds, projected to 3-D with PCA\n"
        f"({100 * d['var']:.0f}% of the variance). Amber: fraud ({int(y.sum())}); "
        "blue: legitimate. Axes are unitless.",
        "credit": SOURCE,
    }


def caption_file(t: dict) -> str:
    link = load_links().get("github", "")
    return f"{t['title']}\n\n{t['body']}\n\n{t['credit']}\n{link}\n"


# ----------------------------------------------------------------------------- canvas


def canvas(w: int, h: int, variant: str, t: dict):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=BG)
    k = min(w, h) / 1080
    if h > w:  # vertical 9:16: the rotating plot gets the full width of the middle band
        rect = [0.0, 0.15, 1.0, 0.68]
    elif w > h:
        rect = [0.2, 0.08, 0.6, 0.76] if variant == "full" else [0.12, 0.1, 0.76, 0.8]
    else:
        rect = [0.02, 0.12, 0.96, 0.7] if variant == "full" else [0.02, 0.13, 0.96, 0.76]
    ax = fig.add_axes(rect, projection="3d", facecolor=BG)
    fs = 10 * k
    top = 0.955 if h <= w else 0.94
    fig.text(0.05, top, t["title"], color=INK, fontsize=2.8 * fs, fontweight="bold", va="top")
    if variant == "full":
        fig.text(0.05, top - 0.065 * 1080 / h, t["body"], color=MUTED, fontsize=1.15 * fs,
                 va="top", linespacing=1.5)  # fmt: skip
    fig.text(0.05, 0.03, t["credit"], color=MUTED, fontsize=fs, va="bottom", linespacing=1.5)
    link = load_links().get("github", "").replace("https://", "")
    if link and w >= h:  # top right, on the title line: never collides with the credit lines
        fig.text(0.95, top - 0.012, link, color=ACCENT, fontsize=1.1 * fs, ha="right", va="top")
    elif link:  # vertical: under the title
        fig.text(0.05, top - 0.04, link, color=ACCENT, fontsize=1.1 * fs, va="top")
    return fig, ax, k


def style(ax, k: float, hide_ticks: bool = False) -> None:
    ax.computed_zorder = False  # let glow lines/points sit on top of surfaces
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis._axinfo["grid"].update(color=GRID, linewidth=0.6)
        axis.line.set_color(GRID)
        axis.label.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9 * k + 1)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


GLOW = ((1100, 0.05), (450, 0.12), (160, 0.32), (40, 1.0))


def glow_line(ax, x, y, z, k=1.0):
    for lw, a in ((10, 0.07), (5.5, 0.14), (2.8, 0.4), (1.4, 1.0)):
        ax.plot(x, y, z, color=ACCENT, lw=lw * k, alpha=a, solid_capstyle="round", zorder=5)


def glow_points(ax, x, y, z, k=1.0, alpha=1.0, sizes=GLOW):
    for s, a in sizes:
        ax.scatter(x, y, z, s=s * k**2, color=ACCENT, alpha=float(np.clip(a * alpha, 0, 1)),
                   depthshade=False, lw=0, zorder=6)  # fmt: skip


# ----------------------------------------------------------------------------- scenes
# Each scene draws onto a cleared 3-D axis; phase in [0, 1) drives the loop (None = still).


def threshold_contour(d: dict):
    helper = plt.figure()
    cs = helper.add_subplot().contour(d["X"], d["Y"], d["P"], levels=[d["thr"]])
    plt.close(helper)
    return cs.allsegs[0]


def scene_landscape(ax, d, k, phase=None):
    # loop: rises from flat (0-25 %), true surface (25-88 %), settles back (88-100 %): seamless
    lift = 1.0 if phase is None else float(smooth(phase / 0.25) * smooth((1 - phase) / 0.12))
    ax.plot_surface(d["X"], d["Y"], d["P"] * lift, cmap="plasma", vmin=0, vmax=1, linewidth=0,
                    antialiased=True, rcount=80, ccount=80, zorder=1)  # fmt: skip
    if lift > 0.98:  # the threshold contour is drawn only on the true surface
        for seg in d["_contour"]:
            glow_line(ax, seg[:, 0], seg[:, 1], np.full(len(seg), d["thr"] + 0.01), k)
    ax.set_zlim(0, 1)
    ax.set_xlabel("merchant_risk_score", labelpad=12 * k)
    ax.set_ylabel("velocity_score", labelpad=12 * k)
    ax.set_zlabel("P(fraud)", labelpad=8 * k)


def scene_logodds(ax, d, k, phase=None):
    ax.plot_surface(d["X"], d["Y"], d["L"], cmap="plasma", linewidth=0, antialiased=True,
                    rcount=80, ccount=80, zorder=1)  # fmt: skip
    lt = np.log(d["thr"] / (1 - d["thr"]))
    for seg in d["_contour"]:
        glow_line(ax, seg[:, 0], seg[:, 1], np.full(len(seg), lt + 0.1), k)
    ax.set_xlabel("merchant_risk_score", labelpad=12 * k)
    ax.set_ylabel("velocity_score", labelpad=12 * k)
    ax.set_zlabel("log-odds of fraud", labelpad=8 * k)


def scene_cost(ax, d, k, phase=None):
    ax.plot_surface(d["X"], d["Y"], d["Z"], cmap="viridis", linewidth=0, antialiased=True,
                    alpha=0.97, zorder=1)  # fmt: skip
    lift = 0.012 * (d["Z"].max() - d["Z"].min())
    vx, vy, vz = d["valley"]
    glow_line(ax, vx, vy, vz + lift, k)
    px, py, pz, _ = d["project"]
    glow_points(ax, [px], [py], [pz + lift], k)
    if phase is not None:  # two comets slide down the x1 cost curve into its minimum, twice a loop
        cx, cz = d["curve1"]
        i_min = int(np.argmin(cz))
        local = (2 * phase) % 1.0
        fade = float(smooth(local / 0.15) * smooth((1 - local) / 0.2))
        for path in (np.arange(0, i_min + 1), np.arange(len(cx) - 1, i_min - 1, -1)):
            j = path[min(len(path) - 1, int(smooth(local) * (len(path) - 1)))]
            glow_points(ax, [cx[j]], [1.0], [cz[j] + lift], k, alpha=fade,
                        sizes=((600, 0.08), (220, 0.2), (60, 1.0)))  # fmt: skip
    ticks = [-2, -1, 0]
    ax.set_xticks(ticks, [f"{10.0**t:g}" for t in ticks])
    ax.set_xlabel("step-up threshold (log scale)", labelpad=12 * k)
    ax.set_ylabel("FN-loss multiplier", labelpad=12 * k)
    ax.set_zlabel("USD per 1,000 transactions", labelpad=14 * k)


def scene_galaxy(ax, d, k, phase=None):
    P, y = d["P"], d["y"]
    ax.scatter(
        *P[y == 0].T, s=1.2 * k**2, color=LEGIT, alpha=0.12, lw=0, depthshade=False, zorder=1
    )
    f = P[y == 1]
    tw = np.ones(len(f))
    if phase is not None:  # each fraud point twinkles twice per loop, own phase
        tw = 0.55 + 0.45 * np.sin(2 * np.pi * (2 * phase + d["phase"]))
    for s, a in ((260, 0.05), (110, 0.11), (38, 0.32), (10, 1.0)):
        rgba = np.tile(to_rgba(ACCENT), (len(f), 1))
        rgba[:, 3] = np.clip(a * tw, 0, 1)
        ax.scatter(*f.T, s=s * k**2, c=rgba, lw=0, depthshade=False, zorder=2)
    ax.set_xlabel("PC1", labelpad=-10)
    ax.set_ylabel("PC2", labelpad=-10)
    ax.set_zlabel("PC3", labelpad=-10)


FIGURES = {
    # name: (scene, data key, initial (elev, azim), hide axis numbers, has clip)
    "landscape": (scene_landscape, "landscape", (24, -122), False, True),
    "landscape_logodds": (scene_logodds, "landscape", (24, -122), False, False),
    "cost_valley": (scene_cost, "cost", (30, -125), False, True),
    "galaxy": (scene_galaxy, "galaxy", (18, -50), True, True),
}


def render(name: str, d: dict, w: int, h: int, variant: str, phase=None, fig_ax=None):
    scene, _, (elev, azim), hide, _ = FIGURES[name]
    if fig_ax is None:
        fig, ax, k = canvas(w, h, variant, texts(name, d))
    else:
        fig, ax, k = fig_ax
        ax.cla()
    style(ax, k, hide)
    scene(ax, d, k, phase)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):  # axis titles scale with the canvas like the ticks
        axis.label.set_fontsize(12 * k)
    if phase is None:
        ax.view_init(elev=elev, azim=azim)
    else:  # one full turn per loop, gentle bob
        ax.view_init(elev=elev + 5 * np.sin(2 * np.pi * phase), azim=azim + 360 * phase)
    return fig, ax, k


# ----------------------------------------------------------------------------- annotated clips

ANNOT_SECONDS = 8  # 3 captions, each held ~2 s
FADE = 0.25  # seconds: quick fade-in/out (<= 0.3 s), then fully opaque >= 1.5 s
# caption windows in seconds (start, end) incl. fades; the first one wraps around the loop, so the
# very first frame already shows text and the loop stays seamless
ANNOT_WINDOWS = ((-FADE, 2.5), (2.5, 5.25), (5.25, 8.0 - FADE))
CAPTION_PT, SMALL_PT = 34, 18  # at 1080 px; ~47 px caption, note >= half of it (phone-readable)
FONT_DIR = C.ROOT / "web" / "src" / "fonts"
MONO = "FraudMono"


def register_mono() -> set[int]:
    """Make the website's FraudMono (woff2 subsets) available to matplotlib; returns its cmap."""
    from tempfile import mkdtemp

    from fontTools.ttLib import TTFont

    tmp = Path(mkdtemp(prefix="fraudmono-"))
    cmap: set[int] = set()
    for weight in (400, 500, 600):
        font = TTFont(FONT_DIR / f"fraudmono-{weight}.woff2")
        font.flavor = None
        ttf = tmp / f"fraudmono-{weight}.ttf"
        font.save(ttf)
        font_manager.fontManager.addfont(str(ttf))
        cmap |= set(font.getBestCmap())
    return cmap


def safe_box(w: int, h: int) -> tuple[float, float, float, float]:
    """(left, right, top, bottom) in figure fractions where text may go. 9:16: clear of the top
    ~270 px, the bottom ~350 px and the right ~130 px that phone apps cover with their UI."""
    if h > w:
        return 0.07, 0.88, 0.86, 0.185
    return 0.06, 0.94, 0.95, 0.03


def annotations(name: str, d: dict) -> list[dict]:
    """Caption phrases (split at " · " into lines), optional 3-D target for an arrow, small note."""
    web = _json("p2_threshold_web")
    if name == "landscape":
        seg = max(d["_contour"], key=len)
        mid = seg[len(seg) // 2]
        return [
            {"text": "Fraud risk, as the model sees it"},
            {"text": "Low risk → flat · red flags → it spikes"},
            {"text": "Step-up threshold", "target": (mid[0], mid[1], d["thr"] + 0.01)},
        ]
    if name == "cost_valley":
        px, py, pz, _ = d["project"]
        lift = 0.012 * (d["Z"].max() - d["Z"].min())
        return [
            {"text": "What does each threshold cost?"},
            {"text": "Too strict: annoy customers · too loose: miss fraud"},
            {
                "text": f"Cheapest point ≈ ${pz:,.0f} · per 1,000 txns",
                "target": (px, py, pz + lift),
                "small": COST_LABEL,
            },
        ]
    y = d["y"]
    n, n_fraud = web["n_rows"], web["n_fraud"]
    if (len(y), int(y.sum())) != (n, n_fraud):  # the plotted rows are the dev split of the report
        raise SystemExit(f"galaxy rows {len(y)}/{int(y.sum())} != report {n}/{n_fraud}")
    centre = np.median(d["P"][y == 1], axis=0)
    return [
        {"text": f"{n:,} transactions"},
        {"text": f"Gold = fraud ({n_fraud:,})"},
        {"text": "Fraud sits apart — · the model learns this", "target": tuple(centre)},
    ]


def wrap(fig, phrase: str, size: float, width_px: float, weight=600, max_words=8) -> list[str]:
    """Break at " · " and newlines, then word-wrap so every line fits (measured); a phrase that
    needs two lines is split evenly (no lone last word)."""
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", fontsize=size, family=MONO, weight=weight)

    def width(text: str) -> float:
        probe.set_text(text)
        return probe.get_window_extent(renderer).width

    lines: list[str] = []
    for part in phrase.replace(" · ", "\n").split("\n"):
        words = part.split()
        greedy, line = [], ""
        for word in words:
            trial = f"{line} {word}".strip()
            if line and width(trial) > width_px:
                greedy.append(line)
                line = word
            else:
                line = trial
        greedy.append(line)
        if len(greedy) == 2:  # balanced: the cut with the narrowest wider line
            cuts = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
            greedy = list(min(cuts, key=lambda c: max(width(c[0]), width(c[1]))))
        lines.extend(greedy)
    probe.set_text("")
    for line in lines:
        probe.set_text(line)
        if probe.get_window_extent(renderer).width > width_px or len(line.split()) > max_words:
            raise SystemExit(f"caption line does not fit: {line!r}")
    probe.remove()
    return lines


def caption_alpha(t: float, start: float, end: float, total: float) -> float:
    """Fade in/out over FADE seconds inside [start, end); windows may wrap around the loop."""
    for shift in (0.0, total, -total):
        u = t + shift
        if start <= u < end:
            return float(min(1.0, (u - start) / FADE, (end - u) / FADE))
    return 0.0


def annotated_canvas(name: str, d: dict, w: int, h: int, cmap: set[int]):
    plt.rcParams["font.family"] = MONO  # axes, ticks and credit in the website's font too
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=BG)
    k = min(w, h) / 1080
    left, right, top, bottom = safe_box(w, h)
    rect = [0.0, 0.24, 1.0, 0.52] if h > w else [0.02, 0.09, 0.96, 0.66]
    ax = fig.add_axes(rect, projection="3d", facecolor=BG)
    t = texts(name, d)
    link = load_links().get("github", "").replace("https://", "")
    credit = t["credit"] + (f"\n{link}" if link else "")
    credit_lines = wrap(fig, credit, 9.5 * k, (right - left) * w, weight=400, max_words=99)
    fig.text(left, bottom, "\n".join(credit_lines), color=MUTED, fontsize=9.5 * k, va="bottom",
             linespacing=1.5)  # fmt: skip
    caps = []
    for spec in annotations(name, d):
        lines = wrap(fig, spec["text"], CAPTION_PT * k, (right - left) * w)
        caps.append({**spec, "lines": lines})
        if "small" in spec:
            caps[-1]["small_lines"] = wrap(fig, spec["small"], SMALL_PT * k, (right - left) * w,
                                           weight=500)  # fmt: skip
    used = "".join(c["text"] + c.get("small", "") for c in caps) + credit
    missing = sorted({ch for ch in used if ord(ch) not in cmap and not ch.isspace()})
    if missing:
        raise SystemExit(f"FraudMono lacks {missing}: rebuild it with `make fonts`")
    return fig, ax, k, caps


def draw_captions(fig, ax, k, caps, t: float, total: float, w: int, h: int) -> list:
    """Captions visible at time t (with arrow + ring when they point at something)."""
    left, _, top, _ = safe_box(w, h)
    line_h = CAPTION_PT * k * 1.3 / 72 * 100 / h  # figure fraction per caption line
    artists = []
    for cap, (start, end) in zip(caps, ANNOT_WINDOWS, strict=True):
        a = caption_alpha(t, start, end, total)
        if a <= 0:
            continue
        body = "\n".join(cap["lines"])
        size = CAPTION_PT * k
        artists.append(
            fig.text(left, top, body, color=INK, alpha=a, fontsize=size, weight=600, va="top",
                     linespacing=1.3)
        )  # fmt: skip
        y_end = top - len(cap["lines"]) * line_h
        if "small" in cap:
            small = "\n".join(cap["small_lines"])
            artists.append(fig.text(left, y_end - 0.006, small, color=MUTED, alpha=a,
                                    fontsize=SMALL_PT * k, weight=500, va="top",
                                    linespacing=1.3))  # fmt: skip
            y_end -= 0.006 + len(cap["small_lines"]) * SMALL_PT * k * 1.3 / 72 * 100 / h
        if "target" in cap:
            x2, y2, _ = proj3d.proj_transform(*cap["target"], ax.get_proj())
            ax.scatter(*([v] for v in cap["target"]), s=900 * k**2, facecolors="none",
                       edgecolors=INK, linewidths=3 * k, alpha=a, depthshade=False,
                       zorder=9)  # fmt: skip
            ax.annotate("", xy=(x2, y2), xycoords="data", xytext=(left + 0.05, y_end - 0.012),
                        textcoords="figure fraction", annotation_clip=False,
                        arrowprops={"arrowstyle": "-|>", "color": INK, "alpha": a, "lw": 3 * k,
                                    "mutation_scale": 26 * k, "shrinkA": 0, "shrinkB": 18 * k,
                                    "connectionstyle": "arc3,rad=-0.2"})  # fmt: skip
    return artists


def annotated_frame(name, d, fig_ax_caps, phase: float, w: int, h: int) -> list:
    fig, ax, k, caps = fig_ax_caps
    scene, _, (elev, azim), hide, _ = FIGURES[name]
    ax.cla()
    style(ax, k, hide)
    # landscape: the true surface from the first frame (no rise from flat); cost: one dot only
    scene(ax, d, k, phase if name == "galaxy" else None)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_fontsize(12 * k)
    # a gentle sway (not a full turn) so the arrows' targets stay readable; seamless loop
    ax.view_init(
        elev=elev + 4 * np.sin(2 * np.pi * phase), azim=azim + 18 * np.sin(2 * np.pi * phase)
    )
    return draw_captions(fig, ax, k, caps, phase * ANNOT_SECONDS, ANNOT_SECONDS, w, h)


def save_annotated_clip(name: str, d: dict, label: str, w: int, h: int, cmap: set[int]) -> Path:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required for clips")
    fig, ax, k, caps = annotated_canvas(name, d, w, h, cmap)
    path = OUT / f"{name}_annotated_{label}.mp4"
    args = [
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-preset",
        "slow",
        "-an",
        "-movflags",
        "+faststart",
    ]
    writer = animation.FFMpegWriter(fps=CLIP_FPS, codec="libx264", extra_args=args)
    n = ANNOT_SECONDS * CLIP_FPS
    with writer.saving(fig, str(path), dpi=100):
        for i in range(n):
            artists = annotated_frame(name, d, (fig, ax, k, caps), i / n, w, h)
            writer.grab_frame(facecolor=BG)
            for a in artists:
                a.remove()
    plt.close(fig)
    print("wrote", path.relative_to(C.ROOT))
    return path


def annotated_preview(name: str, d: dict, folder: Path, cmap: set[int]) -> None:
    """Low-res frames in the middle of each caption (layout check; no video)."""
    for label, (w, h) in {"square": (540, 540), "vertical": (540, 960)}.items():
        fig, ax, k, caps = annotated_canvas(name, d, w, h, cmap)
        for j, (start, end) in enumerate(ANNOT_WINDOWS):
            mid = ((start + end) / 2) % ANNOT_SECONDS
            artists = annotated_frame(name, d, (fig, ax, k, caps), mid / ANNOT_SECONDS, w, h)
            out = folder / f"{name}_annotated_{label}_{j + 1}.png"
            fig.savefig(out, facecolor=BG, dpi=100)
            for a in artists:
                a.remove()
        plt.close(fig)
        print("wrote", out.parent / f"{name}_annotated_{label}_*.png")


# ----------------------------------------------------------------------------- output


def save_still(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=BG, dpi=100)
    plt.close(fig)
    print("wrote", path.relative_to(C.ROOT))


def save_clip(name: str, d: dict, label: str, w: int, h: int) -> Path:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required for clips")
    n = CLIP_SECONDS * CLIP_FPS
    fig, ax, k = render(name, d, w, h, "social", phase=0.0)
    path = OUT / f"{name}_{label}.mp4"
    args = [
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-preset",
        "slow",
        "-an",
        "-movflags",
        "+faststart",
    ]
    writer = animation.FFMpegWriter(fps=CLIP_FPS, codec="libx264", extra_args=args)
    with writer.saving(fig, str(path), dpi=100):
        for i in range(n):  # phase i/n never reaches 1, so the last frame flows into the first
            render(name, d, w, h, "social", phase=i / n, fig_ax=(fig, ax, k))
            writer.grab_frame(facecolor=BG)
    plt.close(fig)
    print("wrote", path.relative_to(C.ROOT))
    return path


def make_gif(mp4: Path, name: str) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    gif = OUT / f"{name}.gif"
    palette = "split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=sierra2_4a"
    for width, fps in ((540, 15), (480, 15), (420, 12), (360, 12), (320, 10)):
        vf = f"fps={fps},scale={width}:-1:flags=lanczos,{palette}"
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-vf",
            vf,
            "-loop",
            "0",
            str(gif),
        ]
        subprocess.run(cmd, check=True)
        if gif.stat().st_size <= GIF_MAX_BYTES:
            break
    size = gif.stat().st_size
    if size > GIF_MAX_BYTES:
        raise SystemExit(f"{gif.name} is {size / 1e6:.1f} MB, above the 8 MB limit")
    print(f"wrote {gif.relative_to(C.ROOT)} ({width}px, {fps} fps, {size / 1e6:.1f} MB)")
    return gif


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="low-res previews (full + social)")
    mode.add_argument("--full", action="store_true", help="all stills and clips (GitHub Actions)")
    mode.add_argument("--annotated", action="store_true", help="annotated clips (GitHub Actions)")
    mode.add_argument("--annotated-preview", metavar="DIR", type=Path,
                      help="a few low-res annotated frames into DIR (layout check)")  # fmt: skip
    ap.add_argument("--only", choices=["landscape", "cost_valley", "galaxy"], help="one family")
    args = ap.parse_args()

    dev = dev_only(load_raw())  # dev rows only; the holdout is never used here
    land = landscape_data(dev)
    land["_contour"] = threshold_contour(land)
    data = {"landscape": land, "cost": cost_data(), "galaxy": galaxy_data(dev)}
    if args.annotated or args.annotated_preview:
        cmap = register_mono()
        for name in ("landscape", "cost_valley", "galaxy"):
            if args.only and name != args.only:
                continue
            d = data[FIGURES[name][1]]
            if args.annotated_preview:
                args.annotated_preview.mkdir(parents=True, exist_ok=True)
                annotated_preview(name, d, args.annotated_preview, cmap)
                continue
            OUT.mkdir(parents=True, exist_ok=True)
            clips = {
                lbl: save_annotated_clip(name, d, lbl, w, h, cmap)
                for lbl, (w, h) in CLIP_SIZES.items()
            }
            make_gif(clips["1080x1080"], f"{name}_annotated")
        return
    folder = OUT / "preview" if args.quick else OUT
    folder.mkdir(parents=True, exist_ok=True)
    for name, (_, key, _, _, has_clip) in FIGURES.items():
        if args.only and not name.startswith(args.only):
            continue
        d = data[key]
        (folder / f"{name}_caption.txt").write_text(caption_file(texts(name, d)))
        if args.quick:
            for variant in ("full", "social"):
                fig, _, _ = render(name, d, 540, 540, variant)
                save_still(fig, folder / f"{name}_{variant}_preview.png")
            continue
        for variant in ("full", "social"):
            for label, (w, h) in STILL_SIZES.items():
                fig, _, _ = render(name, d, w, h, variant)
                save_still(fig, OUT / f"{name}_{variant}_{label}.png")
        if has_clip:
            clips = {lbl: save_clip(name, d, lbl, w, h) for lbl, (w, h) in CLIP_SIZES.items()}
            make_gif(clips["1080x1080"], name)


if __name__ == "__main__":
    main()
