#!/usr/bin/env python3
"""Cross-flow generalisation of the anisotropy correction (MVR-8).

The ultimate test of whether the correction learned *physics* rather than the
periodic-hill geometry: train on periodic hills only, then predict the DNS
Reynolds-stress anisotropy on completely DIFFERENT separated flows — a curved
backward-facing step (CBFS), parametric bumps (BUMP) and a converging-diverging
channel (CNDV) — using the McConkey et al. (2021) dataset.

Only the geometry-agnostic (invariant + nonlocal) features are used, since they
carry no absolute position and are therefore comparable across geometries. The
target is the DNS anisotropy in barycentric-map coordinates.

Reference (within-flow) baselines are reported alongside, so the drop from
in-distribution to out-of-distribution is explicit and honest.

Run from the project root (after extracting the McConkey dataset):
    python analysis/16_crossflow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402

from mlrans import anisotropy, dataset, mcconkey, paths, plots  # noqa: E402

FEATS = dataset.feature_columns("invariant_nonlocal")


def _prep():
    """Load every case; return features, barycentric target, family, case."""
    rows_X, rows_y, fam, case_id, coords = [], [], [], [], []
    avail = mcconkey.available()
    for family, cases in avail.items():
        for c in cases:
            feat, b, cx, cy = mcconkey.features_target(c)
            x, y = anisotropy.barycentric_from_b(b)
            rows_X.append(feat[FEATS].to_numpy())
            rows_y.append(np.column_stack([x, y]))
            fam += [family] * len(feat)
            case_id += [c] * len(feat)
            coords.append(np.column_stack([cx, cy]))
    X = np.concatenate(rows_X); Y = np.concatenate(rows_y)
    fam = np.array(fam); case_id = np.array(case_id)
    coords = np.concatenate(coords)
    # clip heavy-tailed nonlocal features on the training (PHLL) distribution
    ph = fam == "PHLL"
    lo, hi = np.nanpercentile(X[ph], [1, 99], axis=0)
    X = np.clip(X, lo, hi)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    return X[ok], Y[ok], fam[ok], case_id[ok], coords[ok], avail


def _bary_r2(yt, yp):
    return 0.5 * (r2_score(yt[:, 0], yp[:, 0]) + r2_score(yt[:, 1], yp[:, 1]))


def main() -> int:
    paths.ensure_output_dirs()
    X, Y, fam, case_id, coords, avail = _prep()
    print("Cases:", {k: len(v) for k, v in avail.items()})
    print(f"Total points: {len(X)}  |  features: {len(FEATS)}")

    ph = fam == "PHLL"

    # Reference: within-periodic-hill leave-one-geometry-out.
    logo = LeaveOneGroupOut()
    within = []
    for tr, te in logo.split(X[ph], Y[ph], case_id[ph]):
        m = RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=0)
        m.fit(X[ph][tr], Y[ph][tr])
        within.append(_bary_r2(Y[ph][te], m.predict(X[ph][te])))
    print(f"\n[reference] within-PHLL leave-one-geometry-out barycentric R2 = "
          f"{np.mean(within):.3f}")

    # Cross-flow: train on ALL periodic hills, test on each other flow.
    model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
    model.fit(X[ph], Y[ph])
    print("\nCross-flow (train on periodic hills, predict UNSEEN flow):")
    rows = [{"test_flow": "PHLL (in-distribution)", "bary_r2": float(np.mean(within))}]
    for family in ("CBFS", "BUMP", "CNDV"):
        m = fam == family
        if not m.any():
            continue
        r2 = _bary_r2(Y[m], model.predict(X[m]))
        rows.append({"test_flow": family, "bary_r2": r2})
        print(f"  {family:6s}  barycentric R2 = {r2:6.3f}   ({m.sum()} points)")
    tab = pd.DataFrame(rows)
    tab.to_csv(paths.TABLES_DIR / "crossflow_generalization.csv", index=False)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["0.5"] + ["tab:blue"] * (len(tab) - 1)
    ax.bar(tab["test_flow"], tab["bary_r2"], color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(r"anisotropy barycentric $R^2$")
    ax.set_title("Cross-flow generalisation: trained on periodic hills, "
                 "tested on unseen separated flows")
    ax.set_ylim(min(-0.2, tab["bary_r2"].min() - 0.1), 1.0)
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / "crossflow_generalization.png", dpi=150)
    plt.close(fig)

    # predicted-vs-DNS anisotropy map on the curved backward-facing step
    if (fam == "CBFS").any():
        m = fam == "CBFS"
        pred = model.predict(X[m])
        err = np.hypot(pred[:, 0] - Y[m, 0], pred[:, 1] - Y[m, 1])
        plots.plot_scatter_field(
            coords[m, 0], coords[m, 1], err,
            paths.FIGURES_DIR / "crossflow_CBFS_error.png",
            title="Anisotropy prediction error on the curved backward-facing "
                  "step (trained on periodic hills only)",
            cbar_label="|bary error|", cmap="magma", vmin=0, vmax=0.3)
    print("\n[fig] crossflow_generalization.png, crossflow_CBFS_error.png  "
          "[csv] crossflow_generalization.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
