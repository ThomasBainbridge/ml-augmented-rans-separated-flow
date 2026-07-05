"""Common-frame interpolation and vertical-profile extraction.

RANS (cell centres) and DNS (its own mesh) live on different point clouds. To
compare them we interpolate both onto a shared set of query points — either
vertical profile lines at fixed x/H stations (the classic periodic-hill
comparison) or a regular grid for error maps.

All coordinates are in hill-height units (x/H, y/H).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

# Classic periodic-hill comparison stations (x/H). Profiles of U(y) at these
# streamwise locations are the standard way turbulence models are assessed.
PROFILE_STATIONS = (0.05, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

# Domain top (channel height in H units).
Y_TOP = 3.036


def hill_surface(x_over_H: np.ndarray | float) -> np.ndarray:
    """Lower-wall (hill) height y/H as a function of x/H for the alpha=1 hill.

    Reproduces the standard parameterised periodic-hill profile (H = 28 mm,
    Xiao et al.). Used to mask the solid region and to anchor profile lines.
    The polynomial is defined on x in [0, 54] mm and mirrored about mid-domain.
    """
    x = np.atleast_1d(np.asarray(x_over_H, dtype=float)).copy()
    xmm = x * 28.0                      # back to mm, where the polynomial lives
    # Domain length is 9H = 252 mm; the hill at the downstream end mirrors the
    # upstream one about x = 126 mm.
    xmm = np.where(xmm > 126.0, 252.0 - xmm, xmm)
    h = np.zeros_like(xmm)

    def seg(mask, coeffs):
        xx = xmm[mask]
        h[mask] = np.polyval(coeffs[::-1], xx)  # coeffs given low->high order

    seg((xmm >= 0) & (xmm < 9),
        [28.0, 0.0, 6.775070969851e-03, -2.124527775800e-03])
    seg((xmm >= 9) & (xmm < 14),
        [2.507355893131e1, 9.754803562315e-01, -1.016116352781e-01, 1.889794677828e-03])
    seg((xmm >= 14) & (xmm < 20),
        [2.579601052357e1, 8.206693007457e-01, -9.055370274339e-02, 1.626510569859e-03])
    seg((xmm >= 20) & (xmm < 30),
        [4.046435022819e1, -1.379581654948e0, 1.945884504128e-02, -2.070318932190e-04])
    seg((xmm >= 30) & (xmm < 40),
        [1.792461334664e1, 8.743920332081e-01, -5.567361123058e-02, 6.277731764683e-04])
    seg((xmm >= 40) & (xmm <= 54),
        [5.639011190988e1, -2.010520359035e0, 1.644919857549e-02, 2.674976141766e-05])
    # Clip to physical bounds: crest height is H (=28 mm) and floor is 0.
    h = np.clip(h, 0.0, 28.0)
    h = h / 28.0                        # back to H units
    return h if h.size > 1 else float(h[0])


def interpolate(src_xy: np.ndarray, src_val: np.ndarray, dst_xy: np.ndarray,
                method: str = "linear") -> np.ndarray:
    """Interpolate a scattered field onto destination points.

    Points outside the source convex hull return NaN (linear/cubic); callers
    should mask NaNs rather than silently trusting extrapolation.
    """
    return griddata(src_xy, src_val, dst_xy, method=method)


def vertical_profile(xy: np.ndarray, val: np.ndarray, x_station: float,
                     n: int = 200, y_top: float = Y_TOP,
                     method: str = "linear") -> tuple[np.ndarray, np.ndarray]:
    """Extract val(y) along a vertical line at a fixed x/H station.

    Returns (y, val_on_line) with y spanning from the local hill surface to the
    channel top, so profiles start exactly at the wall.
    """
    y0 = float(np.atleast_1d(hill_surface(x_station))[0])
    y = np.linspace(y0, y_top, n)
    q = np.column_stack([np.full(n, x_station), y])
    v = interpolate(xy, val, q, method=method)
    return y, v


def regular_grid(nx: int = 180, ny: int = 120, x_range=(0.0, 9.0),
                 y_top: float = Y_TOP):
    """Build a regular (x/H, y/H) grid masked to the fluid region above the hill.

    Returns (X, Y, mask) where mask is True inside the fluid domain. Use the
    mask to blank the solid hill region in error maps.
    """
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(0.0, y_top, ny)
    X, Y = np.meshgrid(xs, ys)
    mask = Y >= hill_surface(X.ravel()).reshape(X.shape)
    return X, Y, mask
