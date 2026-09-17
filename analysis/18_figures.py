#!/usr/bin/env python3
"""Clean, presentation-quality figures.

Regenerates the committed figures (docs/figures/) from the saved
tables and processed dataset with a consistent, minimal style: descriptive
titles, axis labels, legends, subtle grid. No annotations or callouts — the
figures are meant to be read off their axes like standard scientific plots.

Run from the project root (after the analysis pipeline has produced the tables):
    python analysis/18_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mlrans import dataset, paths  # noqa: E402

T = paths.TABLES_DIR
FIG = paths.FIGURES_DIR
DOCS = paths.ROOT / "docs" / "figures"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    "legend.frameon": True, "legend.framealpha": 0.9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.dpi": 120,
})
NAVY, RED, ORANGE, GREEN, GREY = "#1f3b73", "#c0392b", "#e08e0b", "#2e8b57", "#7f8c8d"


def _triangle(ax):
    p = np.array([[1, 0], [0, 0], [0.5, np.sqrt(3) / 2], [1, 0]])
    ax.plot(p[:, 0], p[:, 1], "k-", lw=1.3)
    ax.text(1.0, -0.06, "one-component", ha="center", fontsize=9)
    ax.text(0.0, -0.06, "two-component", ha="center", fontsize=9)
    ax.text(0.5, np.sqrt(3) / 2 + 0.03, "isotropic", ha="center", fontsize=9)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.12, 0.95)


def _save(fig, name):
    for d in (FIG, DOCS):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png")
    plt.close(fig)
    print(f"  [fig] {name}.png")


def fig_reattachment():
    clo = pd.read_csv(T / "closure_reattachment.csv").set_index("closure")
    apr = pd.read_csv(T / "aposteriori_reattachment.csv").set_index("field")
    dns = float(clo.loc["DNS", "x_reattachment"])
    vals = {"DNS": dns,
            "k-ω SST": float(clo.loc["k-omega SST", "x_reattachment"]),
            "k-ε": float(clo.loc["k-epsilon", "x_reattachment"]),
            "ML-corrected\n(unseen geometry)":
                float(apr.loc["corrected (pred beta, unseen)", "x_reattachment"])}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(list(vals), list(vals.values()), color=NAVY)
    ax.axhline(dns, color="k", ls=":", lw=1.1)
    ax.set_ylabel("reattachment location  $x/H$")
    ax.set_title("Reattachment location: baselines and ML correction vs DNS")
    fig.tight_layout()
    _save(fig, "reattachment_summary")


def fig_barycentric():
    df = dataset.load_dataset()
    c = df[df["case"] == "case_1p0"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for a, (xk, yk, ttl) in zip(ax, [
            ("bary_x_rans", "bary_y_rans", "RANS  (k-ω SST)"),
            ("bary_x_dns", "bary_y_dns", "DNS")]):
        _triangle(a)
        a.hexbin(c[xk], c[yk], gridsize=45, cmap="Blues", mincnt=1,
                 extent=(-0.05, 1.05, -0.02, 0.9), linewidths=0.2)
        a.set_title(ttl)
    fig.suptitle("Reynolds-stress anisotropy in the barycentric map "
                 "(periodic hill)", fontsize=12.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "anisotropy_barycentric")


def fig_generalization():
    cf = pd.read_csv(T / "crossflow_generalization.csv")
    mf = pd.read_csv(T / "multiflow_generalization.csv")
    sub = mf[mf["held_out_flow"] != "PHLL"]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"wspace": 0.22})

    names = cf["test_flow"].str.replace(" (in-distribution)", "", regex=False)
    ax[0].bar(names, cf["bary_r2"], color=[GREY] + [NAVY] * (len(cf) - 1))
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_ylim(-2.6, 1.05)
    ax[0].set_ylabel("anisotropy prediction  $R^2$")
    ax[0].set_title("Trained on periodic hills, tested on each flow")
    ax[0].tick_params(axis="x", rotation=15)

    x = np.arange(len(sub))
    ax[1].bar(x - 0.2, sub["single_flow_PHLL"], 0.4, color=NAVY,
              label="trained on 1 flow (hills)")
    ax[1].bar(x + 0.2, sub["multi_flow"], 0.4, color=ORANGE,
              label="trained on 3 flows")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_ylim(-2.6, 1.05)
    ax[1].set_xticks(x); ax[1].set_xticklabels(sub["held_out_flow"])
    ax[1].set_ylabel("anisotropy prediction  $R^2$")
    ax[1].set_title("Single-flow vs multi-flow training")
    ax[1].legend(loc="lower right")
    fig.suptitle("Generalisation across flow type", fontsize=13.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "generalization_story")


def fig_geometry_generalization():
    m = pd.read_csv(T / "correction_cv_metrics.csv")
    m = m[(m["model"] == "random_forest") & (m["features"] == "invariant")]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    labels = [s.replace("case_", "α=").replace("p", ".") for s in m["held_out"]]
    ax.bar(labels, m["r2"], color=GREEN)
    ax.axhline(m["r2"].mean(), color="k", ls="--", lw=1.1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("$R^2$ on the unseen hill shape")
    ax.set_xlabel("hill shape held out of training")
    ax.set_title("Generalisation to an unseen hill shape "
                 "(leave-one-geometry-out)")
    fig.tight_layout()
    _save(fig, "geometry_generalization")


def _summary():
    clo = pd.read_csv(T / "closure_reattachment.csv").set_index("closure")
    apr = pd.read_csv(T / "aposteriori_reattachment.csv").set_index("field")
    cf = pd.read_csv(T / "crossflow_generalization.csv")
    mf = pd.read_csv(T / "multiflow_generalization.csv")
    df = dataset.load_dataset(); c = df[df["case"] == "case_1p0"]
    dns = float(clo.loc["DNS", "x_reattachment"])

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    a = ax[0, 0]
    a.bar(["DNS", "RANS\nbaseline", "ML-corrected\n(unseen geom.)"],
          [dns, float(clo.loc["k-omega SST", "x_reattachment"]),
           float(apr.loc["corrected (pred beta, unseen)", "x_reattachment"])],
          color=["k", RED, ORANGE])
    a.axhline(dns, color="k", ls=":", lw=1)
    a.set_ylabel("reattachment  $x/H$")
    a.set_title("A   Reattachment: baseline vs correction vs DNS")

    b = ax[0, 1]; _triangle(b)
    b.hexbin(c["bary_x_dns"], c["bary_y_dns"], gridsize=40, cmap="Blues",
             mincnt=1, extent=(-0.05, 1.05, -0.02, 0.9), linewidths=0.2)
    b.plot(c["bary_x_rans"], c["bary_y_rans"], ".", ms=1.2, color=RED, alpha=0.5)
    b.set_title("B   Anisotropy: DNS (blue) vs RANS (red), barycentric map")

    cc = ax[1, 0]
    names = cf["test_flow"].str.replace(" (in-distribution)", "", regex=False)
    cc.bar(names, cf["bary_r2"], color=[GREY] + [NAVY] * (len(cf) - 1))
    cc.axhline(0, color="k", lw=1); cc.set_ylim(-2.6, 1.05)
    cc.set_ylabel("anisotropy  $R^2$")
    cc.set_title("C   Cross-flow: trained on hills, tested on each flow")
    cc.tick_params(axis="x", rotation=15)

    sub = mf[mf["held_out_flow"] != "PHLL"]; x = np.arange(len(sub))
    d = ax[1, 1]
    d.bar(x - 0.2, sub["single_flow_PHLL"], 0.4, color=NAVY, label="1 flow")
    d.bar(x + 0.2, sub["multi_flow"], 0.4, color=ORANGE, label="3 flows")
    d.axhline(0, color="k", lw=1); d.set_ylim(-2.6, 1.05)
    d.set_xticks(x); d.set_xticklabels(sub["held_out_flow"])
    d.set_ylabel("anisotropy  $R^2$"); d.legend(loc="lower right")
    d.set_title("D   Multi-flow training (single vs three flows)")

    fig.suptitle("ML-augmented RANS of separated flow  (periodic hill, "
                 r"$Re_H=5600$, validated against DNS)", fontsize=15, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "summary")


def main() -> int:
    paths.ensure_output_dirs()
    print("Rendering figures...")
    fig_reattachment()
    fig_barycentric()
    fig_generalization()
    fig_geometry_generalization()
    _summary()
    print("Done. Figures written to results/figures/ and docs/figures/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
