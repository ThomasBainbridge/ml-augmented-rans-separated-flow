#!/usr/bin/env python3
"""Multi-flow training: does training diversity rescue generalisation? (MVR-9)

MVR-8 showed a correction trained on ONE flow (periodic hills) fails on other
separated flows. The natural question: does training on SEVERAL flows transfer
to an unseen one? Here we do leave-one-FLOW-out cross-validation across the four
McConkey separated-flow families (periodic hills, curved backward-facing step,
parametric bumps, converging-diverging channel): train on three, test on the
held-out fourth, and compare to the single-flow (train-on-periodic-hills) result
from MVR-8.

Target: DNS anisotropy in barycentric coordinates. Features: geometry-agnostic
invariant + nonlocal set.

Run from the project root (after extracting the McConkey dataset):
    python analysis/17_multiflow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from mlrans import dataset, mcconkey, paths, plots  # noqa: E402

FEATS = dataset.feature_columns("invariant_nonlocal")
MAX_TRAIN = 150_000          # subsample training for speed (RF on ~600k is slow)
RNG = np.random.default_rng(0)


def _bary_r2(yt, yp):
    return 0.5 * (r2_score(yt[:, 0], yp[:, 0]) + r2_score(yt[:, 1], yp[:, 1]))


def _fit_predict(Xtr, Ytr, Xte):
    lo, hi = np.nanpercentile(Xtr, [1, 99], axis=0)
    Xtr = np.clip(Xtr, lo, hi); Xte = np.clip(Xte, lo, hi)
    if len(Xtr) > MAX_TRAIN:
        idx = RNG.choice(len(Xtr), MAX_TRAIN, replace=False)
        Xtr, Ytr = Xtr[idx], Ytr[idx]
    m = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
    m.fit(Xtr, Ytr)
    return m.predict(Xte)


def main() -> int:
    paths.ensure_output_dirs()
    X, Y, fam, case, coords = mcconkey.assemble(FEATS)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    X, Y, fam = X[ok], Y[ok], fam[ok]
    families = ["PHLL", "CBFS", "BUMP", "CNDV"]
    print("points per flow:", {f: int((fam == f).sum()) for f in families})

    ph = fam == "PHLL"
    rows = []
    for held in families:
        te = fam == held
        # multi-flow: train on all OTHER flows
        multi = _bary_r2(Y[te], _fit_predict(X[~te], Y[~te], X[te]))
        # single-flow reference: train on periodic hills only (skip if held==PHLL)
        if held == "PHLL":
            single = np.nan
        else:
            single = _bary_r2(Y[te], _fit_predict(X[ph], Y[ph], X[te]))
        rows.append({"held_out_flow": held, "single_flow_PHLL": single,
                     "multi_flow": multi})
        print(f"  {held:6s}  single(PHLL)={single:7.3f}   multi(3 flows)={multi:7.3f}")

    tab = pd.DataFrame(rows)
    tab.to_csv(paths.TABLES_DIR / "multiflow_generalization.csv", index=False)

    # figure: single vs multi-flow, per held-out flow (the non-PHLL ones)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = tab[tab["held_out_flow"] != "PHLL"]
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - 0.2, sub["single_flow_PHLL"], 0.4, label="trained on 1 flow (periodic hills)")
    ax.bar(x + 0.2, sub["multi_flow"], 0.4, label="trained on 3 other flows")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(sub["held_out_flow"])
    ax.set_ylabel(r"anisotropy barycentric $R^2$ on held-out flow")
    ax.set_xlabel("held-out (unseen) flow")
    ax.set_title("Does training diversity rescue cross-flow generalisation?")
    ax.set_ylim(min(-1.0, sub[["single_flow_PHLL", "multi_flow"]].min().min() - 0.2), 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / "multiflow_generalization.png", dpi=150)
    plt.close(fig)
    print("\n[fig] multiflow_generalization.png  [csv] multiflow_generalization.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
