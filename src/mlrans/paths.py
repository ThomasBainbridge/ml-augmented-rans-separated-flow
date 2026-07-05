"""Project paths and shared physical constants.

Centralising these here keeps every script runnable from the project root and
avoids scattering magic numbers (hill height, bulk velocity, Reynolds number)
across the codebase.
"""

from __future__ import annotations

from pathlib import Path

# --- Directory layout -------------------------------------------------------
# This file is <root>/src/mlrans/paths.py, so the root is three parents up.
ROOT = Path(__file__).resolve().parents[2]

CASES_DIR = ROOT / "cases"
DATA_DIR = ROOT / "data"
DATA_EXTERNAL = DATA_DIR / "external"
DATA_PROCESSED = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Baseline RANS case and the DNS reference set.
BASELINE_CASE = CASES_DIR / "periodicHill_kOmegaSST"
DNS_5CASES_DIR = DATA_EXTERNAL / "pehill-5-cases-DNS"

# Parameterised-geometry cases available in the DNS database (hill-slope alpha).
DNS_CASE_NAMES = ("case_0p5", "case_0p8", "case_1p0", "case_1p2", "case_1p5")


# --- Physical constants (periodic hill, Xiao et al. 2020) -------------------
# The DNS database and the baseline RANS case are both defined at Re_H = 5600.
HILL_HEIGHT_M = 0.028   # H, hill height in metres (blockMesh: 28 mm, scale 1e-3)
BULK_VELOCITY = 1.0     # Ub, bulk velocity [m/s] set by meanVelocityForce
NU = 5.0e-6             # kinematic viscosity [m^2/s]  (=> Re_H = Ub*H/nu = 5600)
REYNOLDS_H = BULK_VELOCITY * HILL_HEIGHT_M / NU   # 5600

# Domain extent in hill-height units (standard parameterised periodic hill).
DOMAIN_LX_H = 9.0       # streamwise length / H
DOMAIN_LY_H = 3.036     # channel height / H


def ensure_output_dirs() -> None:
    """Create results directories if they do not yet exist (idempotent)."""
    for d in (FIGURES_DIR, TABLES_DIR, DATA_PROCESSED):
        d.mkdir(parents=True, exist_ok=True)


def dns_case_dir(case_name: str) -> Path:
    """Return the dns-data directory for a named DNS case (e.g. 'case_1p0')."""
    return DNS_5CASES_DIR / case_name / "dns-data"
