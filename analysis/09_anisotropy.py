#!/usr/bin/env python3
"""Reynolds-stress anisotropy correction (MVR-6): a more expressive target.

beta_nut corrects only the magnitude of the turbulent stress and is undefined
where the flow is counter-gradient (~20% of cells). This script instead learns
the **anisotropy** discrepancy between DNS and RANS in the barycentric map
(Delta x_bary, Delta y_bary) — the state-of-the-art target (Emory & Iaccarino;
Wu, Wang & Xiao) — which is defined everywhere and captures the *shape* of the
turbulence that a linear eddy-viscosity model gets wrong.

Outputs:
  * leave-one-geometry-out R^2 for each anisotropy-discrepancy component;
  * the barycentric map showing RANS states collapsing vs DNS filling it;
  * true vs predicted anisotropy-discrepancy magnitude maps (unseen geometry).

Run from the project root (after 03 with anisotropy columns):
    python analysis/09_anisotropy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlrans import anisotropy, correction, dataset, paths, plots  # noqa: E402

MAP_CASE = "case_1p0"


def _triangle(ax):
    import numpy as np
    pts = np.array([[1, 0], [0, 0], [0.5, np.sqrt(3) / 2], [1, 0]])
    ax.plot(pts[:, 0], pts[:, 1], "k-", lw=1)
    ax.text(1.02, -0.03, "1-comp", fontsize=8)
    ax.text(-0.12, -0.03, "2-comp", fontsize=8)
    ax.text(0.42, 0.90, "3-comp (iso)", fontsize=8)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths.ensure_output_dirs()
    df = dataset.load_dataset()
    if "dbary_x" not in df.columns:
        print("Anisotropy columns missing — rebuild: python analysis/03_build_dataset.py")
        return 1

    feats = dataset.feature_columns("invariant")
    ok = (df["bary_valid"] & np.isfinite(df[feats]).all(axis=1)
          & np.isfinite(df["dbary_x"]) & np.isfinite(df["dbary_y"]))
    d = df[ok].reset_index(drop=True)
    X, g = d[feats].to_numpy(), d["case"].to_numpy()
    print(f"Anisotropy dataset: {len(d)} valid cells "
          f"({100*len(d)/len(df):.0f}% of all — vs 80% for beta_nut).")

    for comp in ("dbary_x", "dbary_y"):
        res = correction.geometry_wise_cv(X, d[comp].to_numpy(), g, "random_forest")
        print(f"[{comp}] mean out-of-geometry R2 = {res['mean_r2']:.3f}  "
              f"(folds: {[round(r,2) for r in res['fold_r2']]})")

    # --- barycentric map: RANS states vs DNS states (one geometry) -------
    c = d[d["case"] == MAP_CASE]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    for ax, tag, cx, cy in [(axes[0], "RANS (k-omega SST)", "bary_x_rans", "bary_y_rans"),
                            (axes[1], "DNS", "bary_x_dns", "bary_y_dns")]:
        _triangle(ax)
        ax.scatter(c[cx], c[cy], s=3, alpha=0.15, edgecolors="none")
        ax.set_title(tag)
        ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle(f"Reynolds-stress anisotropy states ({MAP_CASE}): "
                 "linear RANS collapses, DNS fills the map")
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / "anisotropy_barycentric.png", dpi=150)
    plt.close(fig)

    # --- predicted vs true discrepancy magnitude (unseen geometry) -------
    mag = np.sqrt(d["dbary_x"].to_numpy()**2 + d["dbary_y"].to_numpy()**2)
    oofx = correction.geometry_wise_oof(X, d["dbary_x"].to_numpy(), g, "random_forest")
    oofy = correction.geometry_wise_oof(X, d["dbary_y"].to_numpy(), g, "random_forest")
    magp = np.sqrt(oofx**2 + oofy**2)
    cm = d["case"].to_numpy() == MAP_CASE
    vmax = float(np.nanpercentile(mag[cm], 97))
    for tag, vals in [("true", mag[cm]), ("pred", magp[cm])]:
        plots.plot_scatter_field(
            d.loc[cm, "x"], d.loc[cm, "y"], vals,
            paths.FIGURES_DIR / f"anisotropy_discrepancy_{tag}_{MAP_CASE}.png",
            title=rf"Anisotropy discrepancy |$\Delta$bary| {tag} — unseen ({MAP_CASE})",
            cbar_label=r"|$\Delta$ barycentric|", cmap="viridis", vmin=0, vmax=vmax)

    print("[fig] anisotropy_barycentric.png, "
          f"anisotropy_discrepancy_true/pred_{MAP_CASE}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
