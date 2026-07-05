#!/usr/bin/env python3
"""Robustness / rigor figures (MVR-3 supplement).

Three self-contained checks, all on the alpha=1.0 hill:
  1. Prediction uncertainty: random-forest per-tree spread of log(beta_nut) on
     the held-out geometry -> where the correction should (not) be trusted.
  2. Wall shear: streamwise wall shear along the hill for baseline vs corrected
     RANS, with the DNS reattachment marked (Cf sign change = reattachment).
  3. Closure sensitivity: reattachment for k-omega SST vs k-epsilon vs DNS,
     showing the model-form error is closure-specific, not a bug.

Run from the project root (after 03/04/06 and the kEpsilon run):
    python analysis/05_robustness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlrans import correction, dataset, metrics, paths, plots  # noqa: E402

TARGET = "case_1p0"
RUN = dataset.ML_RUNS


def _internal(case):
    import pyvista as pv
    return pv.read(sorted((RUN / case).glob("VTK/**/internal.vtu"))[-1])


def _reattach(case, field="U"):
    mesh, centres, dbot = dataset.load_case_mesh(case)
    U = np.asarray(mesh.cell_data[field])[:, 0]
    return metrics.reattachment_nearwall(centres[:, 0], centres[:, 1], U, dbot)


# DNS reattachment computed once (from the co-located UDNS on case_1p0), with
# the same wall-faithful method used for every RANS field.
def _dns_reattach():
    return _reattach(TARGET, field="UDNS")


def uncertainty_map(df):
    feats = dataset.feature_columns("invariant")
    tr = (df["case"] != TARGET) & df["wellposed"] & np.isfinite(df[feats]).all(axis=1)
    rf = correction.make_model("random_forest")
    rf.fit(df.loc[tr, feats].to_numpy(), df.loc[tr, "log_beta_nut"].to_numpy())

    tgt = df[df["case"] == TARGET].reset_index(drop=True)
    X = tgt[feats].to_numpy()
    # epistemic spread = std of per-tree predictions
    per_tree = np.stack([t.predict(X) for t in rf.estimators_], axis=0)
    std = per_tree.std(axis=0)
    plots.plot_scatter_field(
        tgt["x"], tgt["y"], std, paths.FIGURES_DIR / "correction_uncertainty.png",
        title=rf"Prediction uncertainty (RF per-tree std of $\log\beta_{{\nu_t}}$, unseen {TARGET})",
        cbar_label=r"std $\log\beta_{\nu_t}$", cmap="magma",
        vmin=0, vmax=float(np.percentile(std, 98)))
    print(f"[fig] correction_uncertainty.png  (median std={np.median(std):.3f})")


def wall_shear_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyvista as pv

    def wss(case):
        vtp = sorted((RUN / case).glob("VTK/**/boundary/bottomWall.vtp"))[-1]
        b = pv.read(vtp)
        x = b.cell_centers().points[:, 0]
        tau = np.asarray(b.cell_data["wallShearStress"])[:, 0]
        o = np.argsort(x)
        return x[o], tau[o]

    dns_x = _dns_reattach()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for case, col, lab in [(TARGET, "tab:red", "k-omega SST baseline"),
                           (f"prop_pred_{TARGET}", "tab:blue",
                            "corrected (pred, unseen)")]:
        x, tau = wss(case)
        ax.plot(x, tau / dataset.UB**2, color=col, lw=1.4, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(dns_x, color="0.4", ls=":", lw=1.2, label="DNS reattachment")
    ax.set_xlabel(r"$x/H$")
    ax.set_ylabel(r"$\tau_{w,x}/U_b^2$")
    ax.set_title("Hill-surface wall shear: baseline vs corrected "
                 "(zero crossing = separation/reattachment)")
    ax.set_xlim(0, 9)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / "wall_shear_comparison.png", dpi=150)
    plt.close(fig)
    print("[fig] wall_shear_comparison.png")


def closure_comparison():
    dns_x = _dns_reattach()
    rows = [{"closure": "DNS", "x_reattachment": dns_x}]
    rows.append({"closure": "k-omega SST", "x_reattachment": _reattach(TARGET)})
    keps = RUN / f"kEpsilon_{TARGET}"
    if (keps / "VTK").exists():
        rows.append({"closure": "k-epsilon", "x_reattachment": _reattach(f"kEpsilon_{TARGET}")})
    else:
        print("  (k-epsilon run not found; skipping that bar)")
    tab = pd.DataFrame(rows)
    tab.to_csv(paths.TABLES_DIR / "closure_reattachment.csv", index=False)
    print("\nReattachment by closure (x/H):")
    print(tab.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = {"DNS": "k", "k-omega SST": "tab:red", "k-epsilon": "tab:green"}
    ax.bar(tab["closure"], tab["x_reattachment"],
           color=[colors.get(c, "tab:blue") for c in tab["closure"]])
    ax.axhline(dns_x, color="k", ls=":", lw=1)
    ax.set_ylabel(r"reattachment $x/H$")
    ax.set_title("Closure sensitivity of reattachment (periodic hill, Re=5600)")
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / "closure_comparison.png", dpi=150)
    plt.close(fig)
    print("[fig] closure_comparison.png")


def main() -> int:
    paths.ensure_output_dirs()
    df = dataset.load_dataset()
    uncertainty_map(df)
    wall_shear_figure()
    closure_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
