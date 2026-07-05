"""Unit tests for the pure-computation core (no OpenFOAM / pyvista needed).

Run:  python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlrans import anisotropy, features, grid, metrics  # noqa: E402


# --- geometry ---------------------------------------------------------------
def test_hill_surface_crest_and_floor():
    # crest at x=0 is one hill height; mid-domain floor is zero.
    assert grid.hill_surface(0.0) == pytest.approx(1.0, abs=1e-6)
    assert grid.hill_surface(4.5) == pytest.approx(0.0, abs=1e-6)
    # symmetric about mid-domain
    assert grid.hill_surface(0.3) == pytest.approx(grid.hill_surface(9 - 0.3), abs=1e-6)


def test_hill_surface_bounds():
    x = np.linspace(0, 9, 500)
    h = grid.hill_surface(x)
    assert np.all(h >= -1e-9) and np.all(h <= 1.0 + 1e-9)


# --- beta_nut target --------------------------------------------------------
def test_effective_eddy_viscosity_sign_and_mask():
    # <u'v'> negative, positive shear -> positive nut
    uv = np.array([-0.01, -0.01])
    nut = features.effective_eddy_viscosity(uv, np.array([1.0, 0.0]),
                                            np.array([0.0, 0.0]), shear_floor=1e-3)
    assert nut[0] > 0                 # well-posed
    assert np.isnan(nut[1])           # zero shear -> masked


def test_beta_nut_clip():
    b = features.beta_nut(np.array([5e-5, -1e-5]), np.array([1e-5, 1e-5]),
                          clip=(0.0, 10.0))
    assert b[0] == pytest.approx(5.0)
    assert b[1] == 0.0                 # negative -> clipped to floor


# --- invariant features -----------------------------------------------------
def test_invariant_features_bounded():
    n = 50
    rng = np.random.default_rng(0)
    U = rng.normal(size=(n, 3)) * 0.02
    gradU = rng.normal(size=(n, 9))
    gradp = rng.normal(size=(n, 3))
    k = np.abs(rng.normal(size=n)) * 1e-6
    omega = np.abs(rng.normal(size=n)) + 0.01
    nut = np.abs(rng.normal(size=n)) * 1e-4
    wd = np.abs(rng.normal(size=n)) + 0.01
    f = features.invariant_features(U, gradU, gradp, k, omega, nut, wd, nu=5e-6)
    assert np.all(np.abs(f["q1_qcrit"]) <= 1 + 1e-9)
    assert np.all((f["q2_tke_intensity"] >= 0) & (f["q2_tke_intensity"] <= 1))
    assert np.all((f["q3_wall_Re"] >= 0) & (f["q3_wall_Re"] <= 2))
    assert np.all(np.abs(f["q4_pgrad_stream"]) <= 0.5 + 1e-9)


# --- anisotropy barycentric map --------------------------------------------
def test_barycentric_isotropic_is_top_corner():
    # isotropic stress R = (2/3)k I  -> b = 0 -> three-component corner
    k = 1.0
    R = np.array([[2/3*k, 2/3*k, 2/3*k, 0.0, 0.0, 0.0]])  # VTK order xx,yy,zz,xy,yz,xz
    x, y, valid = anisotropy.barycentric(R)
    assert valid[0]
    assert x[0] == pytest.approx(0.5, abs=1e-6)
    assert y[0] == pytest.approx(np.sqrt(3) / 2, abs=1e-6)


def test_barycentric_one_component_corner():
    # single non-zero normal stress -> one-component state (1,0)
    R = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    x, y, _ = anisotropy.barycentric(R)
    assert x[0] == pytest.approx(1.0, abs=1e-6)
    assert y[0] == pytest.approx(0.0, abs=1e-6)


def test_barycentric_inside_triangle():
    rng = np.random.default_rng(1)
    # random SPD-ish stresses
    A = rng.normal(size=(30, 3, 3))
    R = np.einsum("nij,nkj->nik", A, A)  # SPD
    R6 = np.stack([R[:, 0, 0], R[:, 1, 1], R[:, 2, 2],
                   R[:, 0, 1], R[:, 1, 2], R[:, 0, 2]], axis=1)
    x, y, valid = anisotropy.barycentric(R6)
    # inside triangle with corners (1,0),(0,0),(0.5,h)
    h = np.sqrt(3) / 2
    assert np.all(y[valid] >= -1e-6) and np.all(y[valid] <= h + 1e-6)


# --- metrics ----------------------------------------------------------------
def test_reattachment_nearwall_synthetic():
    # near-wall U reversed (negative) up to x=4, positive after -> reattach ~4
    x = np.linspace(0.5, 7.8, 400)
    y = np.zeros_like(x)
    d = np.full_like(x, 0.01)          # all within band
    U = np.where(x < 4.0, -0.1, 0.1)
    xr = metrics.reattachment_nearwall(x, y, U, d, band=0.04)
    assert xr == pytest.approx(4.0, abs=0.15)


def test_rmse_ignores_nan():
    a = np.array([1.0, 2.0, np.nan])
    b = np.array([1.0, 4.0, 5.0])
    assert metrics.rmse(a, b) == pytest.approx(np.sqrt((0 + 4) / 2))
