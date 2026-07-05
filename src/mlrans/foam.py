"""Load OpenFOAM RANS fields from the foamToVTK export.

The baseline case is exported with ``scripts/export_latest_fields.sh`` which
runs ``foamToVTK -latestTime``. This module reads that output with pyvista and
returns the internal-mesh fields on cell centres, with coordinates normalised
by the hill height so they share the DNS (x/H, y/H) frame.

pyvista is imported lazily so that scripts which only touch the DNS side
(e.g. analysis/01_inspect_dns.py) do not require it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import paths


@dataclass
class RANSField:
    """Baseline RANS solution sampled at cell centres (physical mean flow).

    Coordinates x, y are in hill-height units (matching the DNS frame).
    Velocities are in physical units (Ub = 1, so effectively Ub-normalised).
    """

    x: np.ndarray                       # x/H
    y: np.ndarray                       # y/H
    U: np.ndarray                       # Ux
    V: np.ndarray                       # Uy
    k: np.ndarray                       # turbulent kinetic energy
    omega: np.ndarray                   # specific dissipation rate
    nut: np.ndarray                     # turbulent (eddy) viscosity
    p: np.ndarray                       # kinematic pressure
    extras: dict = field(default_factory=dict)   # e.g. Reynolds stress R

    @property
    def n_points(self) -> int:
        return self.x.size

    @property
    def xy(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])


def _find_internal_vtk(case_dir: Path) -> Path:
    """Locate the latest-time internal mesh file inside <case>/VTK."""
    vtk_dir = case_dir / "VTK"
    if not vtk_dir.is_dir():
        raise FileNotFoundError(
            f"No VTK export found at {vtk_dir}.\n"
            "Export the solved fields first:  scripts/export_latest_fields.sh"
        )
    # foamToVTK (v2312) writes internal.vtu inside per-time subdirectories, and
    # older/alternate layouts write <case>_<time>.vtk directly. Support both.
    candidates = sorted(vtk_dir.rglob("internal.vtu")) + sorted(vtk_dir.glob("*.vtk"))
    if not candidates:
        raise FileNotFoundError(
            f"No internal mesh (internal.vtu / *.vtk) under {vtk_dir}."
        )
    # Latest time = last when sorted by name (foamToVTK zero-pads time indices).
    return candidates[-1]


def load_baseline(case_dir: Path | None = None,
                  hill_height_m: float = paths.HILL_HEIGHT_M) -> RANSField:
    """Load the baseline RANS internal field at cell centres.

    Parameters
    ----------
    case_dir : the OpenFOAM case directory. Defaults to the project baseline.
    hill_height_m : H used to normalise coordinates into DNS (x/H, y/H) units.
    """
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "pyvista is required to read OpenFOAM VTK output. "
            "Install the project requirements:  pip install -r requirements.txt"
        ) from exc

    case_dir = Path(case_dir) if case_dir is not None else paths.BASELINE_CASE
    mesh_path = _find_internal_vtk(case_dir)
    mesh = pv.read(mesh_path)

    # Work on cell-centre values: OpenFOAM stores fields as cell data. If the
    # reader delivered point data, sample it back onto cell centres.
    cell_mesh = mesh.point_data_to_cell_data() if mesh.n_cells and \
        len(mesh.cell_data) == 0 else mesh
    centres = cell_mesh.cell_centers().points  # (Ncells, 3) in metres

    def get(name: str) -> np.ndarray:
        if name in cell_mesh.cell_data:
            return np.asarray(cell_mesh.cell_data[name])
        if name in cell_mesh.point_data:
            # fall back: interpolate point field to cell centres
            return np.asarray(cell_mesh.point_data_to_cell_data().cell_data[name])
        raise KeyError(f"Field '{name}' not present in {mesh_path.name}. "
                       f"Available: {list(cell_mesh.cell_data.keys())}")

    U = get("U")
    H = hill_height_m
    extras: dict = {}
    for opt in ("turbulenceProperties:R", "R", "wallShearStress"):
        if opt in cell_mesh.cell_data:
            extras[opt] = np.asarray(cell_mesh.cell_data[opt])

    return RANSField(
        x=centres[:, 0] / H,
        y=centres[:, 1] / H,
        U=U[:, 0],
        V=U[:, 1],
        k=get("k"),
        omega=get("omega"),
        nut=get("nut"),
        p=get("p"),
        extras=extras,
    )
