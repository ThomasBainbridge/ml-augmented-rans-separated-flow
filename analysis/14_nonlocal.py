#!/usr/bin/env python3
"""Nonlocal / transport-informed features (MVR-7).

The TBNN diagnostic (analysis/13) showed the anisotropy's leading coefficient g1
is unpredictable from the LOCAL strain/rotation invariants (R^2 ~ 0). This tests
the direct hypothesis: do NONLOCAL / non-equilibrium features (production/
dissipation imbalance, TKE convection, curvature, adverse pressure gradient,
gradient misalignment) recover that predictability — and improve the
correction's out-of-geometry generalisation?

Two experiments, leave-one-geometry-out:
  A. Predictability of the basis coefficients g_n: local invariants vs
     invariants + nonlocal features.
  B. Generalisation of the correction targets (log beta_nut, anisotropy
     discrepancy): invariant vs invariant + nonlocal feature sets.

Run from the project root (after 03 with nonlocal columns):
    python analysis/14_nonlocal.py
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

from mlrans import anisotropy, correction, dataset, paths, plots, tbnn  # noqa: E402

CASES = list(dataset.paths.DNS_CASE_NAMES)


def optimal_g(case):
    """Per-cell least-squares basis coefficients g for the DNS anisotropy."""
    import pyvista as pv
    m = pv.read(sorted((dataset.ML_RUNS / case).glob("VTK/**/internal.vtu"))[-1])
    S, R = tbnn.normalised_S_R(dataset._cell_gradient(m, "U"),
                               np.asarray(m.cell_data["omega"]))
    T = tbnn.tensor_basis(S, R)
    M = anisotropy._to_matrix(np.asarray(m.cell_data["TauDNS"]))
    k = 0.5 * (M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2])
    b = M / (2.0 * np.maximum(k, 1e-12)[:, None, None])
    for i in range(3):
        b[:, i, i] -= 1.0 / 3.0
    Tf = T.reshape(T.shape[0], T.shape[1], 9); bf = b.reshape(b.shape[0], 9)
    A = np.einsum("nkq,nlq->nkl", Tf, Tf); rhs = np.einsum("nkq,nq->nk", Tf, bf)
    return np.linalg.solve(A + 1e-8 * np.eye(T.shape[1])[None], rhs)


def _logo_r2(X, y, groups):
    logo = LeaveOneGroupOut()
    r2 = []
    for tr, te in logo.split(X, y, groups):
        m = RandomForestRegressor(n_estimators=120, n_jobs=-1, random_state=0)
        m.fit(X[tr], y[tr])
        r2.append(r2_score(y[te], m.predict(X[te])))
    return float(np.mean(r2))


def main() -> int:
    paths.ensure_output_dirs()
    df = dataset.load_dataset()
    inv_f = dataset.feature_columns("invariant")
    nl_f = dataset.feature_columns("nonlocal")
    both_f = dataset.feature_columns("invariant_nonlocal")

    # clip heavy-tailed nonlocal features
    d = df.copy()
    for c in nl_f:
        lo, hi = np.nanpercentile(d[c], [1, 99])
        d[c] = d[c].clip(lo, hi)
    g = np.concatenate([optimal_g(c) for c in CASES])   # aligned: build_all order
    groups = d["case"].to_numpy()
    fin = np.isfinite(d[both_f]).all(axis=1).to_numpy() & np.all(np.isfinite(g), axis=1)

    # === A: predictability of basis coefficients ===
    print("A. Basis-coefficient predictability (leave-one-geometry-out R^2):")
    print(f"{'coeff':6s} {'local (invariants)':>20s} {'+ nonlocal':>14s}")
    rows_a = []
    for n in range(g.shape[1]):
        yg = g[:, n]
        loc = _logo_r2(d.loc[fin, inv_f].to_numpy(), yg[fin], groups[fin])
        aug = _logo_r2(d.loc[fin, both_f].to_numpy(), yg[fin], groups[fin])
        tag = " (eddy-visc)" if n == 0 else ""
        print(f"g{n+1:<5d} {loc:20.3f} {aug:14.3f}{tag}")
        rows_a.append({"coeff": f"g{n+1}", "local": loc, "nonlocal": aug})
    pd.DataFrame(rows_a).to_csv(paths.TABLES_DIR / "nonlocal_g_predictability.csv",
                                index=False)

    # === B: generalisation of the correction targets ===
    print("\nB. Correction-target generalisation (leave-one-geometry-out R^2):")
    for tgt, mask_col in [("log_beta_nut", "wellposed"),
                          ("dbary_x", "bary_valid"), ("dbary_y", "bary_valid")]:
        m = d[mask_col].to_numpy() & np.isfinite(d[both_f]).all(axis=1).to_numpy() \
            & np.isfinite(d[tgt]).to_numpy()
        loc = _logo_r2(d.loc[m, inv_f].to_numpy(), d.loc[m, tgt].to_numpy(), groups[m])
        aug = _logo_r2(d.loc[m, both_f].to_numpy(), d.loc[m, tgt].to_numpy(), groups[m])
        print(f"  {tgt:14s} local={loc:6.3f}   +nonlocal={aug:6.3f}")

    # figure: g_n predictability, local vs +nonlocal
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = pd.DataFrame(rows_a)
    x = np.arange(len(A))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, A["local"], 0.4, label="local invariants")
    ax.bar(x + 0.2, A["nonlocal"], 0.4, label="+ nonlocal features")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(A["coeff"])
    ax.set_ylabel(r"leave-one-geometry-out $R^2$")
    ax.set_title("Does nonlocality recover the anisotropy closure? "
                 "(g1 = eddy-viscosity term)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / "nonlocal_g_predictability.png", dpi=150)
    plt.close(fig)
    print("[fig] nonlocal_g_predictability.png  [csv] nonlocal_g_predictability.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
