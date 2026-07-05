#!/usr/bin/env python3
"""A-posteriori test of the eddy-viscosity correction (MVR-4).

Does injecting the correction actually improve the CFD? We propagate a corrected
eddy viscosity (nut_corr = beta * nut_baseline) through a FROZEN-turbulence
momentum solve and compare the resulting reattachment / velocity profiles
against the baseline and DNS, on a held-out geometry (case_1p0).

Two corrections are propagated:
  * 'true' : beta from the DNS Reynolds stress  -> the ceiling of what an
             eddy-viscosity correction can achieve (ansatz limit).
  * 'pred' : beta from a random forest trained on the OTHER four geometries
             (invariant features) -> the honest, unseen-geometry result.

This is a-posteriori and therefore the real credibility test: a good a-priori
fit does not guarantee the corrected solve is stable or more accurate.

Run from the project root (after 03/04 and scripts/run_ml_cases.sh):
    python analysis/06_aposteriori.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlrans import correction, dataset, foamfield, grid, metrics, paths, plots  # noqa: E402

TARGET = "case_1p0"
RUN = dataset.ML_RUNS


def _foam(cmd: str, cwd: Path):
    full = f"cd '{cwd}' && {cmd}"
    if shutil.which("openfoam2312"):
        return subprocess.run(["openfoam2312", "bash", "-c", full],
                              capture_output=True, text=True)
    return subprocess.run(["bash", "-c", full], capture_output=True, text=True)


def _latest_time_dir(case: Path) -> Path:
    times = [p for p in case.iterdir()
             if p.is_dir() and p.name.replace(".", "", 1).isdigit()
             and p.name != "0.orig"]
    return max(times, key=lambda p: float(p.name))


def _build_and_run(tag: str, k_corr: np.ndarray, base_case: Path,
                   base_time: Path) -> Path:
    """Create a frozen-turbulence propagation case with corrected eddy viscosity.

    kOmegaSST recomputes nut from k/omega at start-up (it ignores an injected
    nut field). Since nut is linear in k in both SST branches, we instead scale
    the frozen turbulent kinetic energy k -> beta*k; with `turbulence off` the
    model then produces nut ~ beta*nut_baseline and holds it fixed while the
    momentum/pressure equations re-solve.
    """
    dst = RUN / f"prop_{tag}_{TARGET}"
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "0").mkdir(parents=True)
    shutil.copytree(base_case / "constant", dst / "constant")
    shutil.copytree(base_case / "system", dst / "system")
    # Start from the converged baseline fields; freeze omega, scale k by beta.
    for f in ("U", "p", "omega", "nut"):
        shutil.copy(base_time / f, dst / "0" / f)
    foamfield.write_scalar_internal(base_time / "k", dst / "0" / "k", k_corr)
    # turbulence off => nut computed once from (beta*k, omega), then frozen.
    _foam("foamDictionary -entry RAS/turbulence -set off constant/turbulenceProperties",
          dst)
    r = _foam("simpleFoam > log.simpleFoam 2>&1", dst)
    conv = "converged" if (dst / "log.simpleFoam").read_text().count(
        "SIMPLE solution converged") else "endTime"
    print(f"  [{tag}] solve finished ({conv})")
    _foam("foamToVTK -latestTime -ascii > log.foamToVTK 2>&1", dst)
    return dst


def _Ux(case: str):
    import pyvista as pv
    vtu = sorted((RUN / case).glob("VTK/**/internal.vtu"))[-1]
    return np.asarray(pv.read(vtu).cell_data["U"])[:, 0]


def main() -> int:
    paths.ensure_output_dirs()
    df = dataset.load_dataset()
    feats = dataset.feature_columns("invariant")

    # Train RF on the four OTHER geometries (well-posed cells only).
    tr = (df["case"] != TARGET) & df["wellposed"] & np.isfinite(df[feats]).all(axis=1)
    model = correction.make_model("random_forest")
    model.fit(df.loc[tr, feats].to_numpy(), df.loc[tr, "log_beta_nut"].to_numpy())

    # Target geometry: full per-cell frame in mesh order.
    tgt = df[df["case"] == TARGET].reset_index(drop=True)
    beta_pred = np.exp(model.predict(tgt[feats].to_numpy()))
    # In cells where the DNS-inferred nut was masked (low shear), beta_nut is
    # NaN; there we have no information, so keep the baseline (beta = 1).
    beta_true = np.nan_to_num(tgt["beta_nut"].to_numpy(), nan=1.0,
                              posinf=10.0, neginf=0.0)
    beta_pred = np.nan_to_num(beta_pred, nan=1.0, posinf=10.0, neginf=0.0)
    print(f"{TARGET}: {len(tgt)} cells | beta_pred med={np.median(beta_pred):.2f} "
          f"| beta_true med={np.median(beta_true):.2f}")

    base_case = RUN / TARGET
    base_time = _latest_time_dir(base_case)

    # Corrected turbulent kinetic energy k -> beta*k (drives nut ~ beta*nut).
    import pyvista as pv
    kb = np.asarray(pv.read(sorted(base_case.glob("VTK/**/internal.vtu"))[-1])
                    .cell_data["k"])
    k_true = np.nan_to_num(np.clip(beta_true, 0.0, None) * kb)
    k_pred = np.nan_to_num(np.clip(beta_pred, 0.0, None) * kb)
    prop_true = _build_and_run("true", k_true, base_case, base_time)
    prop_pred = _build_and_run("pred", k_pred, base_case, base_time)

    # --- reattachment: wall-faithful near-bottom-wall method, all on the
    #     same mesh so distance-to-wall is computed once and shared -----------
    mb, centres, dbot = dataset.load_case_mesh(TARGET)
    xc, yc = centres[:, 0], centres[:, 1]
    U_base = np.asarray(mb.cell_data["U"])[:, 0]
    U_dns = np.asarray(mb.cell_data["UDNS"])[:, 0]
    U_t, U_p = _Ux(f"prop_true_{TARGET}"), _Ux(f"prop_pred_{TARGET}")

    def reatt(U):
        return metrics.reattachment_nearwall(xc, yc, U, dbot)

    rows = []
    for name, U in [("DNS", U_dns), ("RANS baseline", U_base),
                    ("corrected (true beta)", U_t),
                    ("corrected (pred beta, unseen)", U_p)]:
        rows.append({"field": name, "x_reattachment": reatt(U)})
    xy_dns = np.column_stack([xc, yc]); xy_base = xy_dns; xy_p = xy_dns
    tab = pd.DataFrame(rows)
    tab.to_csv(paths.TABLES_DIR / "aposteriori_reattachment.csv", index=False)
    print("\nReattachment location (x/H):")
    print(tab.to_string(index=False))

    # --- velocity-profile overlay: baseline / corrected / DNS -----------
    _profiles_figure(xy_dns, U_dns, xy_base, U_base, xy_p, U_p,
                     paths.FIGURES_DIR / "aposteriori_profiles.png")
    print(f"\n[fig] aposteriori_profiles.png")
    print(f"[csv] aposteriori_reattachment.csv")
    return 0


def _profiles_figure(xy_d, Ud, xy_b, Ub, xy_p, Up, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, xs in enumerate(grid.PROFILE_STATIONS):
        for xy, U, col, ls, lab in [
            (xy_d, Ud, "k", "-", "DNS"),
            (xy_b, Ub, "tab:red", "--", "RANS baseline"),
            (xy_p, Up, "tab:blue", "-.", "corrected (unseen)")]:
            y, u = grid.vertical_profile(xy, U, xs)
            ax.plot(xs + u / dataset.UB, y, color=col, ls=ls, lw=1.2,
                    label=lab if i == 0 else None)
        ax.axvline(xs, color="0.9", lw=0.6, zorder=0)
    xs = np.linspace(0, 9, 400)
    ax.fill_between(xs, 0, grid.hill_surface(xs), color="0.3", zorder=5)
    ax.set_xlabel(r"$x/H + U/U_b$")
    ax.set_ylabel(r"$y/H$")
    ax.set_title("A-posteriori correction (frozen-nut propagation), unseen geometry")
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(0, grid.Y_TOP)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
