"""Reynolds-stress anisotropy in the barycentric map.

The eddy-viscosity correction (beta_nut) can only fix the *magnitude* of the
turbulent stress; it cannot fix its *shape* (anisotropy), and it is undefined
where the flow is counter-gradient. The more expressive, state-of-the-art target
(Emory & Iaccarino; Wu, Wang & Xiao) is the Reynolds-stress **anisotropy**,
represented by its position in the barycentric map — a bounded triangle whose
corners are the one-, two- and three-component (isotropic) limiting states.

Given a Reynolds-stress tensor R_ij:
    k      = 1/2 R_kk
    b_ij   = R_ij/(2k) - 1/3 delta_ij           (anisotropy tensor)
    lambda1>=lambda2>=lambda3                    (eigenvalues of b)
    C1c=lambda1-lambda2, C2c=2(lambda2-lambda3), C3c=3 lambda3+1
    (x,y) = C1c*(1,0) + C2c*(0,0) + C3c*(1/2, sqrt(3)/2)

A linear eddy-viscosity RANS model (k-omega SST) produces an anisotropy that is
tied to the mean strain, so its states collapse toward the plane-strain edge;
DNS fills the interior. The learning target is that discrepancy.
"""

from __future__ import annotations

import numpy as np

# Barycentric-triangle corners.
_C1 = np.array([1.0, 0.0])                    # one-component
_C2 = np.array([0.0, 0.0])                    # two-component
_C3 = np.array([0.5, np.sqrt(3.0) / 2.0])     # three-component (isotropic)


def _to_matrix(R6_vtk: np.ndarray) -> np.ndarray:
    """(N,6) VTK symmetric-tensor order (xx,yy,zz,xy,yz,xz) -> (N,3,3)."""
    c = np.asarray(R6_vtk)
    xx, yy, zz, xy, yz, xz = (c[:, i] for i in range(6))
    M = np.empty((c.shape[0], 3, 3))
    M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = xx, yy, zz
    M[:, 0, 1] = M[:, 1, 0] = xy
    M[:, 1, 2] = M[:, 2, 1] = yz
    M[:, 0, 2] = M[:, 2, 0] = xz
    return M


def barycentric_from_b(b: np.ndarray):
    """Barycentric-map coordinates (x, y) directly from anisotropy tensors b.

    ``b`` is (N, 3, 3), symmetric and (ideally) traceless. Used to score TBNN
    predictions, which output b rather than the full stress.
    """
    b = np.asarray(b)
    w = np.linalg.eigvalsh(0.5 * (b + np.transpose(b, (0, 2, 1))))
    lam1, lam2, lam3 = w[:, 2], w[:, 1], w[:, 0]
    C1c, C2c, C3c = lam1 - lam2, 2.0 * (lam2 - lam3), 3.0 * lam3 + 1.0
    x = C1c * _C1[0] + C2c * _C2[0] + C3c * _C3[0]
    y = C1c * _C1[1] + C2c * _C2[1] + C3c * _C3[1]
    return x, y


def barycentric(R6_vtk: np.ndarray, k_floor: float = 1e-12):
    """Barycentric-map coordinates (x, y) and validity mask for a stress field.

    Parameters
    ----------
    R6_vtk : (N, 6) Reynolds stress in VTK symmetric order (xx,yy,zz,xy,yz,xz).

    Returns (x, y, valid): x,y are the barycentric coordinates (NaN where k<=0),
    valid marks cells with positive turbulent energy.
    """
    M = _to_matrix(R6_vtk)
    k = 0.5 * (M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2])
    valid = k > k_floor
    b = M / (2.0 * np.where(valid, k, 1.0)[:, None, None])
    b[:, 0, 0] -= 1.0 / 3.0
    b[:, 1, 1] -= 1.0 / 3.0
    b[:, 2, 2] -= 1.0 / 3.0
    # symmetric -> real eigenvalues, sorted descending
    w = np.linalg.eigvalsh(b)                 # ascending
    lam1, lam2, lam3 = w[:, 2], w[:, 1], w[:, 0]
    C1c = lam1 - lam2
    C2c = 2.0 * (lam2 - lam3)
    C3c = 3.0 * lam3 + 1.0
    x = C1c * _C1[0] + C2c * _C2[0] + C3c * _C3[0]
    y = C1c * _C1[1] + C2c * _C2[1] + C3c * _C3[1]
    x = np.where(valid, x, np.nan)
    y = np.where(valid, y, np.nan)
    return x, y, valid
