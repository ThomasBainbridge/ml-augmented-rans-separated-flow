#!/usr/bin/env python3
"""Coupled Reynolds-stress ANISOTROPY correction with a custom solver (Track A+).

Predicts the anisotropy discrepancy tensor  db = b_DNS - b_RANS  (6 components,
random forest on invariant + nonlocal features), forms the deviatoric stress
correction  aDelta = 2 k_RANS * db, and injects div(aDelta) into the momentum
equation via the compiled solver simpleFoamAniso — with turbulence co-adapting.
Unlike betaNut (magnitude only), this corrects the STRESS SHAPE, so it can act
even where the eddy-viscosity ansatz is counter-gradient.

Requires:  (cd src/solvers/simpleFoamAniso && openfoam2312 wmake)
Run:       python analysis/15_aniso_coupled.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import RandomForestRegressor  # noqa: E402

from mlrans import anisotropy, dataset, grid, metrics, paths  # noqa: E402

TARGET = "case_1p0"
RUN = dataset.ML_RUNS
CASES = list(dataset.paths.DNS_CASE_NAMES)
# OpenFOAM symmTensor component order:
OF = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]   # xx,xy,xz,yy,yz,zz

ADELTA_HEADER = """FoamFile
{{ version 2.0; format ascii; class volSymmTensorField; object aDelta; }}
dimensions      [0 2 -2 0 0 0 0];
internalField   nonuniform List<symmTensor>
{n}
(
{vals}
)
;
boundaryField
{{
    "(inlet|outlet)" {{ type cyclic; }}
    defaultFaces     {{ type empty; }}
    "(bottomWall|topWall)" {{ type fixedValue; value uniform (0 0 0 0 0 0); }}
}}
"""


def _foam(cmd, cwd):
    r = "openfoam2312 bash -c" if shutil.which("openfoam2312") else "bash -c"
    return subprocess.run(r.split() + [f"cd '{cwd}' && {cmd}"],
                          capture_output=True, text=True)


def _b_tensors(mesh, field):
    """Anisotropy tensor b (N,3,3) from a Reynolds-stress field (VTK symm)."""
    M = anisotropy._to_matrix(np.asarray(mesh.cell_data[field]))
    k = 0.5 * (M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2])
    b = M / (2.0 * np.maximum(k, 1e-12)[:, None, None])
    for i in range(3):
        b[:, i, i] -= 1.0 / 3.0
    return b, k


def _case_db(case):
    """db = b_DNS - b_RANS (N,6 OF order), plus RANS k, per cell."""
    import pyvista as pv
    m = pv.read(sorted((RUN / case).glob("VTK/**/internal.vtu"))[-1])
    b_r, k_r = _b_tensors(m, "turbulenceProperties:R")
    b_d, _ = _b_tensors(m, "TauDNS")
    db = b_d - b_r
    return np.stack([db[:, i, j] for i, j in OF], axis=1), k_r


def main() -> int:
    paths.ensure_output_dirs()
    df = dataset.load_dataset()
    feats = dataset.feature_columns("invariant_nonlocal")
    for c in dataset.NONLOCAL_FEATURES:            # clip heavy tails
        lo, hi = np.nanpercentile(df[c], [1, 99]); df[c] = df[c].clip(lo, hi)

    db = {c: _case_db(c) for c in CASES}
    X = {c: df[df["case"] == c][feats].to_numpy() for c in CASES}

    tr = [c for c in CASES if c != TARGET]
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
    rf.fit(np.concatenate([X[c] for c in tr]),
           np.concatenate([db[c][0] for c in tr]))
    k_r = db[TARGET][1]
    db_pred = rf.predict(X[TARGET])                # predicted (unseen geometry)
    db_true = db[TARGET][0]                        # exact DNS discrepancy (ceiling)

    base_case = RUN / TARGET
    base_time = max((p for p in base_case.iterdir()
                     if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))

    def run(tag, dbf):
        aDelta = 2.0 * k_r[:, None] * dbf
        dst = RUN / f"aniso_{tag}_{TARGET}"
        if dst.exists():
            shutil.rmtree(dst)
        (dst / "0").mkdir(parents=True)
        shutil.copytree(base_case / "constant", dst / "constant")
        shutil.copytree(base_case / "system", dst / "system")
        for f in ("U", "p", "k", "omega", "nut"):
            shutil.copy(base_time / f, dst / "0" / f)
        _foam("foamDictionary -entry application -set simpleFoamAniso "
              "system/controlDict", dst)
        fvs = dst / "system" / "fvSchemes"
        fvs.write_text(fvs.read_text().replace(
            "div((nuEff*dev2(T(grad(U))))) Gauss linear;",
            "div((nuEff*dev2(T(grad(U))))) Gauss linear;\n"
            "    div(aDelta) Gauss linear;"))
        (dst / "0" / "aDelta").write_text(ADELTA_HEADER.format(
            n=len(aDelta),
            vals="\n".join("(" + " ".join(f"{v:.8g}" for v in row) + ")"
                           for row in aDelta)))
        _foam("simpleFoamAniso > log.simpleFoamAniso 2>&1", dst)
        conv = "converged" if "SIMPLE solution converged" in (
            dst / "log.simpleFoamAniso").read_text() else "endTime"
        _foam("foamToVTK -latestTime -ascii > log.foamToVTK 2>&1", dst)
        print(f"  [{tag}] |db| med={np.median(np.abs(dbf)):.4f} -> {conv}")
        return dst

    dst_pred = run("pred", db_pred)
    dst_true = run("true", db_true)

    import pyvista as pv
    _, centres, dbot = dataset.load_case_mesh(TARGET)
    mb = pv.read(sorted(base_case.glob("VTK/**/internal.vtu"))[-1])
    U_dns = np.asarray(mb.cell_data["UDNS"])[:, 0]
    U_base = np.asarray(mb.cell_data["U"])[:, 0]
    U_an = np.asarray(pv.read(sorted(dst_pred.glob("VTK/**/internal.vtu"))[-1])
                      .cell_data["U"])[:, 0]
    U_true = np.asarray(pv.read(sorted(dst_true.glob("VTK/**/internal.vtu"))[-1])
                        .cell_data["U"])[:, 0]
    xy = np.column_stack([centres[:, 0], centres[:, 1]])

    def rmse(U):
        e = [metrics.rmse(grid.vertical_profile(xy, U, xs)[1],
                          grid.vertical_profile(xy, U_dns, xs)[1])
             for xs in grid.PROFILE_STATIONS]
        return float(np.nanmean(e)) / dataset.UB

    def reatt(U):
        return metrics.reattachment_nearwall(centres[:, 0], centres[:, 1], U, dbot)

    rows = [
        {"method": "DNS", "x_reattachment": reatt(U_dns), "U_profile_rmse": 0.0},
        {"method": "baseline", "x_reattachment": reatt(U_base), "U_profile_rmse": rmse(U_base)},
        {"method": "ANISO-coupled (pred, unseen)", "x_reattachment": reatt(U_an),
         "U_profile_rmse": rmse(U_an)},
        {"method": "ANISO-coupled (exact db, ceiling)", "x_reattachment": reatt(U_true),
         "U_profile_rmse": rmse(U_true)},
    ]
    tab = pd.DataFrame(rows)
    tab.to_csv(paths.TABLES_DIR / "aniso_coupled.csv", index=False)
    print("\n" + tab.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
