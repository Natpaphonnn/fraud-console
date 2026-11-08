"""Matplotlib helpers with one consistent, readable style.

Colour roles (fixed, never cycled): legit = blue, fraud = orange, reference
lines = neutral grey. Every plot sets axis labels with units.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud_e2e import config as C

LEGIT = "#2a78d6"
FRAUD = "#eb6834"
NEUTRAL = "#8a8985"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def save(fig, name: str, folder: Path = C.FIGURES_DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path


def rate_bars(table: pd.DataFrame, base_rate: float, ax=None, title: str | None = None):
    """Horizontal bars of fraud rate per level with 95% Wilson CI whiskers."""
    t = table.sort_values("fraud_rate")
    ax = ax or plt.subplots(figsize=(6, 0.35 * len(t) + 1.2))[1]
    y = np.arange(len(t))
    ax.barh(y, t["fraud_rate"] * 100, color=FRAUD, height=0.6, zorder=2)
    err = [(t["fraud_rate"] - t["ci_low"]) * 100, (t["ci_high"] - t["fraud_rate"]) * 100]
    ax.errorbar(
        t["fraud_rate"] * 100,
        y,
        xerr=err,
        fmt="none",
        ecolor=INK_2,
        elinewidth=1,
        capsize=2,
        zorder=3,
    )
    ax.axvline(base_rate * 100, color=NEUTRAL, ls="--", lw=1, zorder=1)
    ax.set_yticks(y, [f"{lvl}  (n={n:,})" for lvl, n in zip(t["level"], t["n"])])
    ax.set_xlabel("Fraud rate (%) — bars: 95% Wilson CI, dashed: overall rate")
    ax.grid(axis="y", visible=False)
    ax.set_title(title or t["feature"].iloc[0])
    return ax


def binned_rate_line(
    table: pd.DataFrame, base_rate: float, xlabel: str, ax=None, logx: bool = False
):
    """Fraud rate across quantile bins of a numeric feature, CI as a band."""
    ax = ax or plt.subplots(figsize=(5, 3.2))[1]
    x = table["bin_mid"].astype(float)
    ax.fill_between(x, table["ci_low"] * 100, table["ci_high"] * 100, color=FRAUD, alpha=0.18, lw=0)
    ax.plot(x, table["fraud_rate"] * 100, color=FRAUD, lw=2, marker="o", ms=4)
    ax.axhline(base_rate * 100, color=NEUTRAL, ls="--", lw=1)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraud rate (%)")
    ax.set_ylim(bottom=0)
    return ax


def class_hist(
    df: pd.DataFrame,
    col: str,
    xlabel: str,
    ax=None,
    bins: int = 40,
    logx: bool = False,
    target: str = C.TARGET,
):
    """Overlaid histograms for legit vs fraud, each class normalised to 100%."""
    ax = ax or plt.subplots(figsize=(5, 3.2))[1]
    x = df[col].astype(float)
    if logx:
        edges = np.geomspace(max(x.min(), 1e-2), x.max(), bins + 1)
    else:
        edges = np.histogram_bin_edges(x, bins=bins)
    for label, colour, name in [(0, LEGIT, "Legit"), (1, FRAUD, "Fraud")]:
        xs = x[df[target] == label]
        ax.hist(
            xs,
            bins=edges,
            weights=np.full(len(xs), 100 / len(xs)),
            histtype="step",
            lw=2,
            color=colour,
            label=f"{name} (n={len(xs):,})",
        )
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of class in bin (%)")
    ax.legend()
    return ax


def auc_forest(table: pd.DataFrame, ax=None):
    """Univariate ROC AUC per feature with bootstrap CI; 0.5 = no signal."""
    t = table.sort_values("separation")
    ax = ax or plt.subplots(figsize=(6, 0.33 * len(t) + 1.2))[1]
    y = np.arange(len(t))
    colours = [FRAUD if a > 0.5 else LEGIT for a in t["auc"]]
    ax.hlines(y, t["auc_ci_low"], t["auc_ci_high"], color=INK_2, lw=1.5)
    ax.scatter(t["auc"], y, color=colours, s=40, zorder=3, edgecolor="#fcfcfb", lw=1.5)
    ax.axvline(0.5, color=NEUTRAL, ls="--", lw=1)
    ax.set_yticks(y, t["feature"])
    ax.set_xlabel("Univariate ROC AUC (95% bootstrap CI); >0.5 orange = higher value → more fraud")
    ax.grid(axis="y", visible=False)
    return ax
