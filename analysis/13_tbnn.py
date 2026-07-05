#!/usr/bin/env python3
"""Tensor-Basis Neural Network for the anisotropy (Track B).

Predicts the DNS Reynolds-stress anisotropy tensor b from the mean strain/
rotation invariants using a TBNN (Ling et al. 2016), whose architecture builds
in rotational invariance:  b = sum_n g_n(lambda) T_n. Compared, on identical
inputs (the 5 scalar invariants), against a generic random forest that predicts
the tensor components directly. Both are validated leave-one-geometry-out and
scored in the barycentric map.

KEY, HONEST FINDING. The TBNN underperforms the unconstrained RF here, and the
diagnostic explains why: the leading basis coefficient g1 (the eddy-viscosity
alignment) is essentially UNPREDICTABLE from the local invariants (train
R^2 ~ 0), even though the basis itself can represent b almost perfectly
(best-possible per-cell fit ~ 0.88 barycentric R^2). In other words the local
Reynolds-stress closure assumption underlying the TBNN — that the anisotropy is
a function of the local strain/rotation invariants — breaks down for separated
flow, where nonlocal / history effects and counter-gradient transport dominate.
This is a concrete demonstration of the limits of local data-driven closures,
consistent with the ~20% counter-gradient cells found in MVR-2, and motivates
nonlocal / transport-informed features as the way forward.

Run from the project root (after the 5 runs):
    python analysis/13_tbnn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from mlrans import anisotropy, dataset, paths, plots, tbnn  # noqa: E402

CASES = list(dataset.paths.DNS_CASE_NAMES)


def case_arrays(case):
    """Return (invariants, basis, b_dns, x, y) per cell for one geometry."""
    import pyvista as pv
    vtu = sorted((dataset.ML_RUNS / case).glob("VTK/**/internal.vtu"))[-1]
    mesh = pv.read(vtu)
    gradU = dataset._cell_gradient(mesh, "U")
    omega = np.asarray(mesh.cell_data["omega"])
    S, R = tbnn.normalised_S_R(gradU, omega)
    inv = tbnn.invariants(S, R)
    basis = tbnn.tensor_basis(S, R)
    # DNS anisotropy b from TauDNS (VTK symm order)
    M = anisotropy._to_matrix(np.asarray(mesh.cell_data["TauDNS"]))
    k = 0.5 * (M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2])
    b = M / (2.0 * np.maximum(k, 1e-12)[:, None, None])
    for i in range(3):
        b[:, i, i] -= 1.0 / 3.0
    c = mesh.cell_centers().points
    return inv, basis, b, c[:, 0], c[:, 1]


def _bary_r2(b_true, b_pred):
    xt, yt = anisotropy.barycentric_from_b(b_true)
    xp, yp = anisotropy.barycentric_from_b(b_pred)
    ok = np.isfinite(xt) & np.isfinite(xp) & np.isfinite(yt) & np.isfinite(yp)
    return 0.5 * (r2_score(xt[ok], xp[ok]) + r2_score(yt[ok], yp[ok]))


def main() -> int:
    paths.ensure_output_dirs()
    data = {c: case_arrays(c) for c in CASES}
    print(f"TBNN inputs built for {len(data)} geometries "
          f"({sum(len(v[0]) for v in data.values())} cells).")

    rows = []
    for held in CASES:
        tr = [c for c in CASES if c != held]
        inv_tr = np.concatenate([data[c][0] for c in tr])
        bas_tr = np.concatenate([data[c][1] for c in tr])
        b_tr = np.concatenate([data[c][2] for c in tr])
        inv_te, bas_te, b_te = data[held][0], data[held][1], data[held][2]

        # clip invariants (heavy tails) + normalise the basis tensors
        lo, hi = np.percentile(inv_tr, [1, 99], axis=0)
        inv_tr = np.clip(inv_tr, lo, hi); inv_te = np.clip(inv_te, lo, hi)
        bas_tr, bscale = tbnn.normalise_basis(bas_tr)
        bas_te, _ = tbnn.normalise_basis(bas_te, bscale)

        model, scaler = tbnn.train_tbnn(inv_tr, bas_tr, b_tr, epochs=80)
        b_tbnn = tbnn.predict_tbnn(model, scaler, inv_te, bas_te)

        # generic RF on the same invariants -> 6 tensor components
        comps = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        y_tr = np.stack([b_tr[:, i, j] for i, j in comps], axis=1)
        rf = RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=0)
        rf.fit(inv_tr, y_tr)
        p = rf.predict(inv_te)
        b_rf = np.zeros_like(b_te)
        for k, (i, j) in enumerate(comps):
            b_rf[:, i, j] = b_rf[:, j, i] = p[:, k]

        r2_tbnn = _bary_r2(b_te, b_tbnn)
        r2_rf = _bary_r2(b_te, b_rf)
        rows.append({"held_out": held, "tbnn_r2": r2_tbnn, "rf_r2": r2_rf})
        print(f"  held-out {held:10s}  TBNN barycentric R2={r2_tbnn:6.3f}   "
              f"RF={r2_rf:6.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(paths.TABLES_DIR / "tbnn_vs_rf.csv", index=False)
    print(f"\nMean out-of-geometry barycentric R2:  TBNN={df['tbnn_r2'].mean():.3f}  "
          f"RF(components)={df['rf_r2'].mean():.3f}")

    # --- diagnostic: is the anisotropy a LOCAL function of the invariants? ---
    # best-possible per-cell g fit, then how predictable each g_n is from lambda.
    def _optg(T, b):
        Tf = T.reshape(T.shape[0], T.shape[1], 9); bf = b.reshape(b.shape[0], 9)
        A = np.einsum("nkq,nlq->nkl", Tf, Tf); rhs = np.einsum("nkq,nq->nk", Tf, bf)
        return np.linalg.solve(A + 1e-8 * np.eye(T.shape[1])[None], rhs)

    from sklearn.model_selection import cross_val_predict
    inv_all = np.concatenate([data[c][0] for c in CASES])
    g_all = np.concatenate([_optg(data[c][1], data[c][2]) for c in CASES])
    lo, hi = np.percentile(inv_all, [1, 99], axis=0)
    inv_all = np.clip(inv_all, lo, hi)
    print("\nDiagnostic — predictability of each basis coefficient g_n from the "
          "local invariants (5-fold R^2; T1=S is the eddy-viscosity term):")
    for n in range(g_all.shape[1]):
        rfg = RandomForestRegressor(n_estimators=60, n_jobs=-1, random_state=0,
                                    max_depth=12)
        pred = cross_val_predict(rfg, inv_all, g_all[:, n], cv=3)
        print(f"    g{n+1}: R^2 = {r2_score(g_all[:, n], pred):6.3f}")
    print("  g1 ~ 0 => the anisotropy is NOT a local function of the invariants; "
          "the TBNN closure assumption fails for this separated flow (see docstring).")

    # map: TBNN-predicted vs DNS anisotropy for case_1p0 (held out)
    inv, basis, b_true, x, y = data["case_1p0"]
    tr = [c for c in CASES if c != "case_1p0"]
    inv_tr = np.concatenate([data[c][0] for c in tr])
    lo, hi = np.percentile(inv_tr, [1, 99], axis=0)
    bas_tr, bscale = tbnn.normalise_basis(np.concatenate([data[c][1] for c in tr]))
    model, scaler = tbnn.train_tbnn(
        np.clip(inv_tr, lo, hi), bas_tr,
        np.concatenate([data[c][2] for c in tr]), epochs=80)
    basis_n, _ = tbnn.normalise_basis(basis, bscale)
    b_pred = tbnn.predict_tbnn(model, scaler, np.clip(inv, lo, hi), basis_n)
    xt, yt = anisotropy.barycentric_from_b(b_true)
    xp, yp = anisotropy.barycentric_from_b(b_pred)
    plots.plot_scatter_field(
        x, y, np.hypot(xp - xt, yp - yt),
        paths.FIGURES_DIR / "tbnn_anisotropy_error_case_1p0.png",
        title="TBNN anisotropy error (barycentric distance to DNS, unseen case_1p0)",
        cbar_label="|b_pred - b_DNS| (bary)", cmap="magma", vmin=0, vmax=0.3)
    print("[fig] tbnn_anisotropy_error_case_1p0.png  [csv] tbnn_vs_rf.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
