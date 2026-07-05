"""Load and inspect the periodic-hill DNS reference data.

Data source
-----------
Xiao, Wu, Laizet, Duan (2020), "Flows over periodic hills of parameterized
geometries", Computers & Fluids 200, 104431. Public database:
https://github.com/xiaoh/para-database-for-PIML  (pehill-5-cases-DNS)

Fetch it with ``scripts/fetch_dns_data.sh``. Each case directory ``dns-data``
contains four whitespace-delimited ASCII files on the DNS mesh. Coordinates are
already normalised by the hill height H, and velocities by the bulk velocity Ub.

Column layout (established by inspecting the files, see analysis/01_inspect_dns.py)
---------------------------------------------------------------------------------
mean_files.dat  (N, 6):  CONFIRMED
    0: x/H     1: y/H     2: U/Ub    3: V/Ub    4: W/Ub (~0)   5: p/(rho Ub^2)

rms_files*.dat  (N, 6 or 5):  PROVISIONAL - see note below
    All share columns 0,1 = x/H, y/H. The remaining columns hold second-order
    velocity statistics (Reynolds-stress components / r.m.s. fluctuations). The
    exact per-column semantics are NOT fully documented in the repo; the one
    robust, physically identifiable quantity is the Reynolds shear stress
    <u'v'>, which is the only strongly negative O(0.1) column:
        rms_files1.dat column 5  ->  <u'v'>/Ub^2   (negative on windward side)

    Treat the individual normal-stress columns as provisional until confirmed
    against the CAF-2020 paper. The pipeline only *requires* <u'v'> (for the
    later eddy-viscosity correction target), and that column is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths

# Column indices (0-based) within each file.
MEAN_COLS = {"x": 0, "y": 1, "U": 2, "V": 3, "W": 4, "p": 5}

# Reynolds shear stress <u'v'> lives in rms_files1, last column. This is the
# only second-order column we rely on downstream, and it is physically
# unambiguous (sign + magnitude match the known periodic-hill result).
UV_FILE = "rms_files1.dat"
UV_COL = 5

DNS_FILES = ("mean_files.dat", "rms_files.dat", "rms_files1.dat", "rms_files2.dat")


@dataclass
class DNSCase:
    """Mean DNS fields for one periodic-hill geometry, on the raw DNS mesh.

    All arrays are 1-D of equal length (scattered points, not a structured
    grid). Coordinates are in hill-height units; velocities in Ub units.
    """

    name: str
    x: np.ndarray            # x/H
    y: np.ndarray            # y/H
    U: np.ndarray            # U/Ub
    V: np.ndarray            # V/Ub
    W: np.ndarray            # W/Ub (~0, spanwise-homogeneous mean flow)
    p: np.ndarray            # p/(rho Ub^2)
    uv: np.ndarray | None    # <u'v'>/Ub^2  (None if rms file absent)

    @property
    def n_points(self) -> int:
        return self.x.size

    @property
    def xy(self) -> np.ndarray:
        """(N, 2) array of (x/H, y/H) coordinates."""
        return np.column_stack([self.x, self.y])


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"DNS file not found: {path}\n"
            "Download the reference data first:\n"
            "    scripts/fetch_dns_data.sh all"
        )
    return path


def load_mean(case_name: str) -> np.ndarray:
    """Load the raw ``mean_files.dat`` array for a case (N, 6)."""
    path = _require(paths.dns_case_dir(case_name) / "mean_files.dat")
    return np.loadtxt(path)


def load_reynolds_shear(case_name: str) -> np.ndarray | None:
    """Load the Reynolds shear stress <u'v'>/Ub^2 (N,) if available, else None."""
    path = paths.dns_case_dir(case_name) / UV_FILE
    if not path.exists():
        return None
    return np.loadtxt(path)[:, UV_COL]


def load_dns_case(case_name: str) -> DNSCase:
    """Load mean fields (+ Reynolds shear stress) for one DNS case.

    Parameters
    ----------
    case_name : one of paths.DNS_CASE_NAMES, e.g. "case_1p0".
    """
    if case_name not in paths.DNS_CASE_NAMES:
        raise ValueError(
            f"Unknown DNS case {case_name!r}. Expected one of {paths.DNS_CASE_NAMES}."
        )
    mean = load_mean(case_name)
    if mean.shape[1] < 6:
        raise ValueError(
            f"{case_name}/mean_files.dat has {mean.shape[1]} columns, expected >= 6."
        )
    uv = load_reynolds_shear(case_name)
    return DNSCase(
        name=case_name,
        x=mean[:, MEAN_COLS["x"]],
        y=mean[:, MEAN_COLS["y"]],
        U=mean[:, MEAN_COLS["U"]],
        V=mean[:, MEAN_COLS["V"]],
        W=mean[:, MEAN_COLS["W"]],
        p=mean[:, MEAN_COLS["p"]],
        uv=uv,
    )


def summarise_file(path: Path) -> dict:
    """Return a shape/range summary of one DNS ``.dat`` file for inspection."""
    a = np.loadtxt(path)
    a = np.atleast_2d(a)
    return {
        "file": path.name,
        "shape": a.shape,
        "col_min": np.round(a.min(axis=0), 5).tolist(),
        "col_max": np.round(a.max(axis=0), 5).tolist(),
    }


def available_cases() -> list[str]:
    """Which DNS cases are actually present on disk (downloaded)."""
    return [
        c for c in paths.DNS_CASE_NAMES
        if (paths.dns_case_dir(c) / "mean_files.dat").exists()
    ]
