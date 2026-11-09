"""Content figures (stills + rotating clips) from the real model and committed results.

    python scripts/content_figures.py --quick   # low-res stills -> reports/content/preview/
    python scripts/content_figures.py --full    # 1080x1080 + 1920x1080 stills, ~8 s mp4 + gif
                                                # -> reports/content/ (GitHub Actions: content.yml)

1. Fraud risk landscape: P(fraud) from the exported final logistic model over merchant_risk_score x
   velocity_score, other inputs fixed at typical dev values (stated on the figure).
2. Cost valley: expected cost per 1,000 transactions over threshold x FN-loss multiplier, from the
   dev out-of-fold counts in reports/p2_threshold_web.json (illustrative assumed costs).
3. Transaction galaxy: dev rows projected to 3-D with PCA of their per-feature contributions to the
   model's log-odds; fraud highlighted.

Nothing is invented: every surface/point comes from reports/ or from scoring real dev rows.
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
from matplotlib import animation  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from fraud_e2e import config as C  # noqa: E402
from fraud_e2e.cost import COST_LABEL  # noqa: E402
from fraud_e2e.data import load_raw  # noqa: E402
from fraud_e2e.export import linear_contributions, linear_scores  # noqa: E402
from fraud_e2e.features import feature_frame  # noqa: E402
from fraud_e2e.splits import dev_only  # noqa: E402
from fraud_e2e.summary import load_links  # noqa: E402

OUT = C.REPORTS_DIR / "content"
BG, PANEL, INK, MUTED, GRID, ACCENT = (
    "#0a0c0f",
    "#0e1115",
    "#e7eaee",
    "#8a93a0",
    "#2a313b",
    "#ffb020",
)
SOURCE = (
    "Data: Credit Card Fraud Detection 2026 (Udit Jain, CC0), "
    "a public dataset of unspecified provenance."
)
SIZES = {"1080x1080": (1080, 1080), "1920x1080": (1920, 1080)}
CLIP_SECONDS, CLIP_FPS = 8, 30


def _json(name: str) -> dict:
    return json.loads((C.REPORTS_DIR / f"{name}.json").read_text())


# ----------------------------------------------------------------------------- data


def model() -> dict:
    return _json("p4_web_model")["final_model"]


def typical_row(dev: pd.DataFrame) -> dict:
    """Numeric medians; most common value for categories and flags (all flags are off)."""
    row = {c: float(dev[c].median()) for c in C.NUMERIC_COLS}
    row |= {c: dev[c].mode()[0] for c in C.CATEGORICAL_COLS + C.BOOLEAN_COLS}
    return row


def landscape_data(dev: pd.DataFrame, n: int = 90) -> dict:
    base = typical_row(dev)
    xs = np.linspace(dev["merchant_risk_score"].min(), dev["merchant_risk_score"].max(), n)
    ys = np.linspace(dev["velocity_score"].min(), dev["velocity_score"].max(), n)
    X, Y = np.meshgrid(xs, ys)
    grid = pd.DataFrame([base] * X.size)
    grid["merchant_risk_score"] = X.ravel()
    grid["velocity_score"] = Y.ravel()
    P = linear_scores(model(), feature_frame(grid)).reshape(X.shape)
    thr = _json("p2_threshold")["threshold"]
    flags_off = all(not base[c] for c in C.BOOLEAN_COLS)
    # z is plotted as log10 P(fraud): most of the slice is below 1%, a linear axis would hide it
    return {"X": X, "Y": Y, "Z": np.log10(P), "P": P, "thr": thr, "flags_off": flags_off}


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
        "t_range": (t.min(), t.max()),
        "params": p,
    }


def galaxy_data(dev: pd.DataFrame) -> dict:
    contrib = linear_contributions(model(), feature_frame(dev))
    pca = PCA(n_components=3, random_state=C.SEED)
    P = pca.fit_transform(contrib.to_numpy())
    return {
        "P": P,
        "y": dev[C.TARGET].to_numpy(),
        "var": float(pca.explained_variance_ratio_.sum()),
        "n_features": contrib.shape[1],
    }


# ----------------------------------------------------------------------------- drawing


def canvas(w: int, h: int, dpi: int = 100):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor=BG)
    rect = [0.2, 0.08, 0.6, 0.76] if w > h else [0.02, 0.13, 0.96, 0.7]
    ax = fig.add_axes(rect, projection="3d", facecolor=BG)
    ax.computed_zorder = False  # let glow lines/points sit on top of surfaces
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis._axinfo["grid"].update(color=GRID, linewidth=0.6)
        axis.line.set_color(GRID)
        axis.label.set_color(MUTED)
    k = min(w, h) / 1080
    ax.tick_params(colors=MUTED, labelsize=9 * k + 1)
    return fig, ax, k


def captions(fig, k: float, title: str, body: str, extra: str = "") -> None:
    fs = 10 * k
    fig.text(0.04, 0.955, title, color=INK, fontsize=2.6 * fs, fontweight="bold", va="top")
    fig.text(0.04, 0.89, body, color=MUTED, fontsize=1.15 * fs, va="top", linespacing=1.5)
    foot = SOURCE if not extra else f"{extra}\n{SOURCE}"
    fig.text(0.04, 0.03, foot, color=MUTED, fontsize=1.0 * fs, va="bottom", linespacing=1.5)
    link = load_links().get("github", "")
    if link:
        fig.text(0.96, 0.03, link.replace("https://", ""), color=ACCENT, fontsize=fs, ha="right")


def glow_line(ax, x, y, z, k=1.0):
    for lw, a in ((10, 0.07), (5.5, 0.14), (2.8, 0.4), (1.4, 1.0)):
        ax.plot(x, y, z, color=ACCENT, lw=lw * k, alpha=a, solid_capstyle="round", zorder=5)


def glow_point(ax, x, y, z, k=1.0):
    for s, a in ((1100, 0.05), (450, 0.12), (160, 0.32), (40, 1.0)):
        ax.scatter(
            [x], [y], [z], s=s * k**2, color=ACCENT, alpha=a, depthshade=False, lw=0, zorder=6
        )


def draw_landscape(d: dict, w: int, h: int):
    fig, ax, k = canvas(w, h)
    ax.plot_surface(d["X"], d["Y"], d["Z"], cmap="plasma", linewidth=0, antialiased=True,
                    rcount=90, ccount=90, zorder=1)  # fmt: skip
    lt = np.log10(d["thr"])
    helper = plt.figure()
    cs = helper.add_subplot().contour(d["X"], d["Y"], d["Z"], levels=[lt])  # 2-D helper
    plt.close(helper)
    for seg in cs.allsegs[0]:
        glow_line(ax, seg[:, 0], seg[:, 1], np.full(len(seg), lt + 0.04), k)
    zt = np.arange(np.ceil(d["Z"].min()), 1)
    ax.set_zticks(zt, [f"{10.0**v:g}" for v in zt])
    ax.set_xlabel("merchant_risk_score", labelpad=12 * k)
    ax.set_ylabel("velocity_score", labelpad=12 * k)
    ax.set_zlabel("P(fraud), log scale", labelpad=10 * k)
    ax.view_init(elev=26, azim=-58)
    fixed = "numeric inputs at dev medians, categories at their most common value"
    fixed += ", all six red flags off" if d["flags_off"] else ", flags at their most common value"
    captions(
        fig,
        k,
        "Fraud risk landscape",
        "P(fraud) from the project's logistic model as two inputs vary. The other 22 are fixed\n"
        f"({fixed}).\nAmber line: the step-up threshold, P = {d['thr']:.4f}. "
        f"On this slice P runs from {d['P'].min():.1e} to {d['P'].max():.2f}.\n"
        "Nearly a flat ramp: in log-odds a logistic model is exactly a plane (additive effects).",
    )
    return fig, ax


def draw_cost(d: dict, w: int, h: int):
    fig, ax, k = canvas(w, h)
    ax.plot_surface(d["X"], d["Y"], d["Z"], cmap="viridis", linewidth=0, antialiased=True,
                    alpha=0.97, zorder=1)  # fmt: skip
    lift = 0.012 * (d["Z"].max() - d["Z"].min())
    vx, vy, vz = d["valley"]
    glow_line(ax, vx, vy, vz + lift, k)
    px, py, pz, pt = d["project"]
    glow_point(ax, px, py, pz + lift, k)
    ticks = [-2, -1, 0]
    ax.set_xticks(ticks, [f"{10.0**t:g}" for t in ticks])
    ax.set_xlabel("step-up threshold (log scale)", labelpad=12 * k)
    ax.set_ylabel("FN-loss multiplier", labelpad=12 * k)
    ax.set_zlabel("USD per 1,000 transactions", labelpad=14 * k)
    ax.view_init(elev=30, azim=-125)
    p = d["params"]
    captions(
        fig,
        k,
        "Cost valley",
        "Expected cost per 1,000 transactions by step-up threshold and by how much of a\n"
        "missed fraud's amount is lost (×0.25 to ×2 of the project's assumption).\n"
        "Dev out-of-fold counts, "
        f"mean of 3 CV repeats; thresholds {d['t_range'][0]:.3f} to {d['t_range'][1]:.0f}. "
        "Glowing line: cheapest threshold per multiplier.\n"
        f"Dot: the project's setting (×1), threshold {pt:.4f}, {pz:,.0f} USD per 1,000.",
        extra=f"{COST_LABEL}: step-up {p['review_cost_per_alert']:g} USD, good-customer friction "
        f"{p['fp_friction_cost']:g} USD, missed fraud = amount + {p['fn_fixed_cost']:g} USD.",
    )
    return fig, ax


def draw_galaxy(d: dict, w: int, h: int):
    fig, ax, k = canvas(w, h)
    P, y = d["P"], d["y"]
    legit, fraud = P[y == 0], P[y == 1]
    ax.scatter(
        *legit.T, s=2.4 * k**2, color="#7fa6d6", alpha=0.35, lw=0, depthshade=False, zorder=1
    )
    for s, a in ((150, 0.04), (60, 0.1), (20, 0.35), (6, 1.0)):
        ax.scatter(*fraud.T, s=s * k**2, color=ACCENT, alpha=a, lw=0, depthshade=False, zorder=2)
    for set_labels in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
        set_labels([])
    for set_label, name in ((ax.set_xlabel, "PC1"), (ax.set_ylabel, "PC2"), (ax.set_zlabel, "PC3")):
        set_label(name, labelpad=-6)
    ax.view_init(elev=18, azim=-50)
    captions(
        fig,
        k,
        "Transaction galaxy",
        f"{len(y):,} dev transactions. Each point is one transaction's "
        f"{d['n_features']} per-feature\n"
        "contributions to the logistic model's log-odds, projected to 3-D with PCA\n"
        f"({100 * d['var']:.0f}% of the variance). "
        f"Amber: fraud ({int(y.sum())}); blue: legitimate.",
    )
    return fig, ax


# ----------------------------------------------------------------------------- output


def save_still(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=BG, dpi=fig.dpi)
    plt.close(fig)
    print("wrote", path.relative_to(C.ROOT))


def save_clip(draw, data: dict, name: str, size=(1080, 1080)) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required for clips")
    fig, ax = draw(data, *size)
    elev, azim0 = ax.elev, ax.azim
    n = CLIP_SECONDS * CLIP_FPS

    def turn(i):
        ax.view_init(elev=elev + 6 * np.sin(2 * np.pi * i / n), azim=azim0 + 360 * i / n)
        return ()

    mp4 = OUT / f"{name}_spin.mp4"
    writer = animation.FFMpegWriter(
        fps=CLIP_FPS,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow"],
    )
    animation.FuncAnimation(fig, turn, frames=n, blit=False).save(mp4, writer=writer, dpi=fig.dpi)
    plt.close(fig)
    palette = "split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=sierra2_4a"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
         f"fps=15,scale=540:-1:flags=lanczos,{palette}", str(OUT / f"{name}_spin.gif")],
        check=True,
    )  # fmt: skip
    print("wrote", mp4.relative_to(C.ROOT), "+ gif")


FIGURES = {
    "landscape": (draw_landscape, "landscape"),
    "cost_valley": (draw_cost, "cost"),
    "galaxy": (draw_galaxy, "galaxy"),
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="one low-res still per figure (preview)")
    mode.add_argument("--full", action="store_true", help="all sizes and clips (GitHub Actions)")
    ap.add_argument("--only", choices=list(FIGURES), help="render one figure")
    args = ap.parse_args()

    dev = dev_only(load_raw())  # dev rows only; the holdout is never used here
    data = {"landscape": landscape_data(dev), "cost": cost_data(), "galaxy": galaxy_data(dev)}
    for name, (draw, key) in FIGURES.items():
        if args.only and name != args.only:
            continue
        if args.quick:
            fig, _ = draw(data[key], 540, 540)
            save_still(fig, OUT / "preview" / f"{name}_preview.png")
            continue
        for label, (w, h) in SIZES.items():
            fig, _ = draw(data[key], w, h)
            save_still(fig, OUT / f"{name}_{label}.png")
        save_clip(draw, data[key], name)


if __name__ == "__main__":
    main()
