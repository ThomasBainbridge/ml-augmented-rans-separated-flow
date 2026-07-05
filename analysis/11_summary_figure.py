#!/usr/bin/env python3
"""Summary figure: the project's full arc on one canvas.

  A  It works: the coupled correction fixes reattachment (unseen geometry).
  B  Why linear RANS fails: anisotropy states collapse vs DNS fill the map.
  C  The challenge: a correction trained on one flow fails on different flows.
  D  The lever: training on several flows rescues cross-flow generalisation.

Reads only saved tables / the processed dataset, so it never recomputes CFD.

Run from the project root (after 03,05,09,10,16,17):
    python analysis/11_summary_figure.py
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


def _triangle(ax):
    p = np.array([[1, 0], [0, 0], [0.5, np.sqrt(3) / 2], [1, 0]])
    ax.plot(p[:, 0], p[:, 1], "k-", lw=1)
    ax.set_aspect("equal"); ax.axis("off")


def main() -> int:
    paths.ensure_output_dirs()
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # A) it works: reattachment baseline -> corrected vs DNS -------------
    clo = pd.read_csv(T / "closure_reattachment.csv").set_index("closure")
    dns = float(clo.loc["DNS", "x_reattachment"])
    base = float(clo.loc["k-omega SST", "x_reattachment"])
    apr = pd.read_csv(T / "aposteriori_reattachment.csv").set_index("field")
    ss = float(apr.loc["corrected (pred beta, unseen)", "x_reattachment"])
    a = ax[0, 0]
    a.bar(["DNS", "k-ω SST\nbaseline", "ML-corrected\n(unseen geom.)"],
          [dns, base, ss], color=["k", "tab:red", "tab:orange"])
    a.axhline(dns, color="k", ls=":", lw=1)
    a.set_ylabel(r"reattachment $x/H$")
    a.set_title("A  The correction works (a-posteriori, unseen geometry)")

    # B) why linear RANS fails: anisotropy states (case_1p0) -------------
    df = dataset.load_dataset()
    c = df[df["case"] == "case_1p0"]
    b = ax[0, 1]
    _triangle(b)
    b.scatter(c["bary_x_rans"], c["bary_y_rans"], s=2, alpha=0.10,
              color="tab:red", label="RANS (linear)")
    b.scatter(c["bary_x_dns"], c["bary_y_dns"], s=2, alpha=0.10,
              color="tab:blue", label="DNS")
    lg = b.legend(loc="upper right", markerscale=4, framealpha=0.9)
    for h in (lg.legend_handles if hasattr(lg, "legend_handles") else lg.legendHandles):
        h.set_alpha(1)
    b.set_title("B  Why linear RANS fails: anisotropy collapses vs DNS fills")

    # C) the challenge: cross-flow generalisation (MVR-8) ---------------
    cf = pd.read_csv(T / "crossflow_generalization.csv")
    cc = ax[1, 0]
    cols = ["0.5"] + ["tab:blue"] * (len(cf) - 1)
    cc.bar(cf["test_flow"].str.replace(" (in-distribution)", "", regex=False),
           cf["bary_r2"], color=cols)
    cc.axhline(0, color="k", lw=0.8)
    cc.set_ylabel(r"anisotropy $R^2$ on test flow")
    cc.set_ylim(min(-2.5, cf["bary_r2"].min() - 0.2), 1.0)
    cc.set_title("C  Trained on periodic hills, tested on UNSEEN flows (fails)")
    cc.tick_params(axis="x", rotation=15)

    # D) the lever: multi-flow rescue (MVR-9) ---------------------------
    mf = pd.read_csv(T / "multiflow_generalization.csv")
    sub = mf[mf["held_out_flow"] != "PHLL"]
    x = np.arange(len(sub))
    d = ax[1, 1]
    d.bar(x - 0.2, sub["single_flow_PHLL"], 0.4, color="tab:blue",
          label="trained on 1 flow")
    d.bar(x + 0.2, sub["multi_flow"], 0.4, color="tab:orange",
          label="trained on 3 flows")
    d.axhline(0, color="k", lw=0.8)
    d.set_xticks(x); d.set_xticklabels(sub["held_out_flow"])
    d.set_ylabel(r"anisotropy $R^2$ on held-out flow")
    d.set_ylim(min(-2.5, sub[["single_flow_PHLL", "multi_flow"]].min().min() - 0.2), 1.0)
    d.set_title("D  Training diversity rescues generalisation")
    d.legend(loc="lower right", fontsize=9)

    fig.suptitle("ML-augmented RANS of separated flow (periodic hill, "
                 r"$Re_H=5600$) — validated against DNS", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = paths.FIGURES_DIR / "summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] {out.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
