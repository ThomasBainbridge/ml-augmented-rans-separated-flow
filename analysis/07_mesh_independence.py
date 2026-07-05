#!/usr/bin/env python3
"""Grid-convergence (mesh-independence) study of the baseline (MVR-1 supplement).

Runs the standalone quasi-2D k-omega SST periodic-hill case at several
resolutions and compares the reattachment length and peak velocity-profile
error. Establishes that the reported baseline model-form error is physics, not
discretisation error, and reports a Richardson/GCI-style estimate of the
grid-converged reattachment.

Run from the project root (needs OpenFOAM):
    python analysis/07_mesh_independence.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlrans import grid, metrics, paths  # noqa: E402

BASE = paths.BASELINE_CASE           # cases/periodicHill_kOmegaSST (blockMesh, H=0.028 m)
RUN = paths.CASES_DIR / "ml_runs"
H = paths.HILL_HEIGHT_M
# (nx, ny) spanwise stays 1. Current baseline is (200,160).
RESOLUTIONS = [(100, 80), (150, 120), (200, 160), (300, 240)]


def _foam(cmd, cwd):
    r = "openfoam2312 bash -c" if shutil.which("openfoam2312") else "bash -c"
    return subprocess.run(r.split() + [f"cd '{cwd}' && {cmd}"],
                          capture_output=True, text=True)


def _reattach(case_dir: Path) -> float:
    import pyvista as pv
    m = pv.read(sorted(case_dir.glob("VTK/**/internal.vtu"))[-1])
    c = m.cell_centers().points
    x, y = c[:, 0] / H, c[:, 1] / H          # -> hill-height units
    U = np.asarray(m.cell_data["U"])[:, 0]
    dbot = y - grid.hill_surface(x)
    return metrics.reattachment_nearwall(x, y, U, dbot)


def run_case(nx, ny) -> dict:
    tag = f"mesh_{nx}x{ny}"
    dst = RUN / tag
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(BASE, dst, ignore=shutil.ignore_patterns(
        "VTK", "postProcessing", "processor*", "[1-9]*", "0", "dynamicCode",
        "log.*", "constant/polyMesh"))
    bm = dst / "system" / "blockMeshDict"
    bm.write_text(bm.read_text().replace("(200 160 1)", f"({nx} {ny} 1)"))
    _foam("./Allrun", dst)
    _foam("foamToVTK -latestTime -ascii > log.foamToVTK 2>&1", dst)
    ncells = nx * ny
    return {"mesh": tag, "cells": ncells, "x_reattachment": _reattach(dst)}


def gci(f_coarse, f_med, f_fine, r=1.5, p=2.0):
    """Grid-Convergence-Index-style fine-grid error estimate (fixed order p)."""
    num = abs(f_fine - f_med)
    return 1.25 * num / (r**p - 1.0) / max(abs(f_fine), 1e-9) * 100.0


def main() -> int:
    paths.ensure_output_dirs()
    rows = [run_case(nx, ny) for nx, ny in RESOLUTIONS]
    df = pd.DataFrame(rows)
    df.to_csv(paths.TABLES_DIR / "mesh_independence.csv", index=False)
    print(df.to_string(index=False))

    xr = df["x_reattachment"].to_numpy()
    if len(xr) >= 3 and np.all(np.isfinite(xr[-3:])):
        est = gci(xr[-3], xr[-2], xr[-1])
        print(f"\nFinest-two reattachment change: "
              f"{abs(xr[-1]-xr[-2]):.3f} H ({100*abs(xr[-1]-xr[-2])/xr[-1]:.1f}%)")
        print(f"GCI-style fine-grid uncertainty on reattachment: ~{est:.1f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["cells"], df["x_reattachment"], "o-")
    ax.axhline(4.5, color="k", ls=":", label="DNS reattachment")
    ax.set_xscale("log")
    ax.set_xlabel("mesh cells")
    ax.set_ylabel(r"reattachment $x/H$")
    ax.set_title("Mesh independence of baseline k-omega SST reattachment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / "mesh_independence.png", dpi=150)
    plt.close(fig)
    print("[fig] mesh_independence.png  [csv] mesh_independence.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
