#!/usr/bin/env python3
"""Fully COUPLED a-posteriori correction with the custom solver (Track A).

Unlike the frozen propagation (analysis/06, 10), this uses simpleFoamBeta — a
forked simpleFoam that solves momentum with nuEff = nu + betaNut*nut while the
turbulence model keeps updating every iteration. So k, omega, nut and the mean
flow all co-adapt to the ML correction field; only betaNut is fixed.

betaNut is predicted by a random forest trained on the OTHER four geometries
(invariant features) and applied to the held-out case_1p0.

Requires the compiled solver:
    (cd src/solvers/simpleFoamBeta && wmake)
Run from the project root (after 03 and the 5 runs):
    python analysis/12_coupled.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlrans import correction, dataset, foamfield, metrics, paths  # noqa: E402

TARGET = "case_1p0"
RUN = dataset.ML_RUNS
BETA_CLIP = (0.1, 5.0)          # keep the coupled solve well-posed

BETANUT_HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
| betaNut: data-driven eddy-viscosity correction (dimensionless)             |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      betaNut;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField   nonuniform List<scalar>
{n}
(
{vals}
)
;

boundaryField
{{
    "(inlet|outlet)" {{ type cyclic; }}
    defaultFaces     {{ type empty; }}
    "(bottomWall|topWall)" {{ type zeroGradient; }}
}}

// ************************************************************************* //
"""


def _foam(cmd, cwd):
    r = "openfoam2312 bash -c" if shutil.which("openfoam2312") else "bash -c"
    return subprocess.run(r.split() + [f"cd '{cwd}' && {cmd}"],
                          capture_output=True, text=True)


def _latest_vtu(case_dir):
    return sorted(case_dir.glob("VTK/**/internal.vtu"))[-1]


def main() -> int:
    if not shutil.which("simpleFoamBeta") and not (
            Path.home() / "OpenFOAM").exists():
        print("simpleFoamBeta not found — build it: "
              "(cd src/solvers/simpleFoamBeta && openfoam2312 wmake)")
        return 1

    paths.ensure_output_dirs()
    df = dataset.load_dataset()
    feats = dataset.feature_columns("invariant")
    tr = (df["case"] != TARGET) & df["wellposed"] & np.isfinite(df[feats]).all(axis=1)
    model = correction.make_model("random_forest")
    model.fit(df.loc[tr, feats].to_numpy(), df.loc[tr, "log_beta_nut"].to_numpy())

    tgt = df[df["case"] == TARGET].reset_index(drop=True)
    beta = np.exp(model.predict(tgt[feats].to_numpy()))
    beta = np.clip(np.nan_to_num(beta, nan=1.0), *BETA_CLIP)
    print(f"{TARGET}: betaNut med={np.median(beta):.3f}  "
          f"[{beta.min():.2f}, {beta.max():.2f}]")

    base_case = RUN / TARGET
    base_time = max((p for p in base_case.iterdir()
                     if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))

    dst = RUN / f"coupled_{TARGET}"
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "0").mkdir(parents=True)
    shutil.copytree(base_case / "constant", dst / "constant")
    shutil.copytree(base_case / "system", dst / "system")
    for f in ("U", "p", "k", "omega", "nut"):
        shutil.copy(base_time / f, dst / "0" / f)
    # turbulence stays ON (co-adapts); solver switched to simpleFoamBeta.
    _foam("foamDictionary -entry application -set simpleFoamBeta system/controlDict", dst)
    # the hand-rolled stress term needs a matching divScheme for nuEffb.
    fvs = dst / "system" / "fvSchemes"
    txt = fvs.read_text()
    assert "div((nuEff*dev2(T(grad(U))))) Gauss linear;" in txt
    fvs.write_text(txt.replace(
        "div((nuEff*dev2(T(grad(U))))) Gauss linear;",
        "div((nuEff*dev2(T(grad(U))))) Gauss linear;\n"
        "    div((nuEffb*dev2(T(grad(U))))) Gauss linear;"))
    # write the betaNut field (mesh cell order matches the copied 0/ fields)
    (dst / "0" / "betaNut").write_text(BETANUT_HEADER.format(
        n=beta.size, vals="\n".join(f"{b:.8g}" for b in beta)))

    print("Running coupled solver (simpleFoamBeta)...")
    _foam("simpleFoamBeta > log.simpleFoamBeta 2>&1", dst)
    log = (dst / "log.simpleFoamBeta").read_text()
    conv = "converged" if "SIMPLE solution converged" in log else "endTime"
    print(f"  solve finished ({conv})")
    _foam("foamToVTK -latestTime -ascii > log.foamToVTK 2>&1", dst)

    # reattachment, consistent wall-faithful method
    _, centres, dbot = dataset.load_case_mesh(TARGET)
    import pyvista as pv
    mb = pv.read(_latest_vtu(base_case))
    U_dns = np.asarray(mb.cell_data["UDNS"])[:, 0]
    U_cpl = np.asarray(pv.read(_latest_vtu(dst)).cell_data["U"])[:, 0]
    xc, yc = centres[:, 0], centres[:, 1]

    from mlrans import grid
    U_base = np.asarray(mb.cell_data["U"])[:, 0]
    xy = np.column_stack([xc, yc])

    def reatt(U):
        return metrics.reattachment_nearwall(xc, yc, U, dbot)

    def prof_rmse(U):
        e = [metrics.rmse(grid.vertical_profile(xy, U, xs)[1],
                          grid.vertical_profile(xy, U_dns, xs)[1])
             for xs in grid.PROFILE_STATIONS]
        return float(np.nanmean(e)) / dataset.UB

    # eddy-viscosity self-regulation: how much did the SST nut change?
    nut_ratio = (np.asarray(pv.read(_latest_vtu(dst)).cell_data["nut"]).mean()
                 / np.asarray(mb.cell_data["nut"]).mean())

    rows = [
        {"method": "DNS", "x_reattachment": reatt(U_dns), "U_profile_rmse": 0.0},
        {"method": "k-omega SST baseline", "x_reattachment": reatt(U_base),
         "U_profile_rmse": prof_rmse(U_base)},
        {"method": "COUPLED (simpleFoamBeta)", "x_reattachment": reatt(U_cpl),
         "U_profile_rmse": prof_rmse(U_cpl)},
    ]
    tab = pd.DataFrame(rows)
    tab.to_csv(paths.TABLES_DIR / "coupled_reattachment.csv", index=False)
    print("\nReattachment (x/H) and velocity-profile RMSE vs DNS:")
    print(tab.to_string(index=False))
    print(f"\nSST eddy-viscosity self-regulation: mean nut(coupled)/nut(baseline) "
          f"= {nut_ratio:.2f}  (< 1 => the model partly cancels the correction)")
    print("For reference, the frozen propagations gave reattachment 3.85 "
          "(single-shot) / 5.47 (self-consistent) — frozen OVERSTATES the "
          "reattachment shift by holding nut fixed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
