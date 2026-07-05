"""Quantitative RANS-vs-DNS error metrics for separated flow.

Focuses on the quantities that expose RANS model-form error on the periodic
hill: velocity-profile error, and the separation/reattachment locations that
bound the recirculation bubble.
"""

from __future__ import annotations

import numpy as np

from . import grid


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square difference, ignoring NaNs (outside-hull query points)."""
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d**2))) if d.size else float("nan")


def mae(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    return float(np.mean(np.abs(d))) if d.size else float("nan")


def profile_velocity_error(rans_xy, rans_U, dns_xy, dns_U,
                           stations=grid.PROFILE_STATIONS) -> dict:
    """Per-station and aggregate RMSE of the streamwise velocity profile U(y).

    Both fields are interpolated onto the same vertical lines, so the metric is
    independent of the differing RANS/DNS meshes.
    """
    per_station = {}
    all_r, all_d = [], []
    for xs in stations:
        y, ur = grid.vertical_profile(rans_xy, rans_U, xs)
        _, ud = grid.vertical_profile(dns_xy, dns_U, xs)
        per_station[xs] = rmse(ur, ud)
        all_r.append(ur)
        all_d.append(ud)
    return {
        "per_station_rmse": per_station,
        "overall_rmse": rmse(np.concatenate(all_r), np.concatenate(all_d)),
    }


def near_wall_streamwise(xy, U, offset: float = 0.05,
                         x_range=(0.05, 8.5), n: int = 400):
    """Sample U a small distance ``offset`` (in H) above the lower (hill) wall.

    The sign of this near-wall velocity marks recirculation (U<0) versus
    attached flow (U>0), which is how separation/reattachment are located.
    """
    xs = np.linspace(x_range[0], x_range[1], n)
    ys = grid.hill_surface(xs) + offset
    q = np.column_stack([xs, ys])
    u = grid.interpolate(xy, U, q)
    return xs, u


def _sign_crossings(xs, u, rising: bool):
    """x-locations where u crosses zero (rising: -->+, or falling: +-->-)."""
    u = np.asarray(u)
    good = np.isfinite(u)
    xs, u = xs[good], u[good]
    s = np.sign(u)
    idx = np.where(np.diff(s) != 0)[0]
    out = []
    for i in idx:
        if rising and u[i] < 0 <= u[i + 1]:
            pass
        elif (not rising) and u[i] > 0 >= u[i + 1]:
            pass
        else:
            continue
        # linear interpolation of the zero crossing
        x0, x1, u0, u1 = xs[i], xs[i + 1], u[i], u[i + 1]
        out.append(x0 - u0 * (x1 - x0) / (u1 - u0))
    return out


def reattachment_location(xy, U, offset: float = 0.05):
    """Estimate leeward reattachment point x/H (last U<0 -> U>0 crossing).

    Returns np.nan if no reverse-flow region is detected near the wall.
    """
    xs, u = near_wall_streamwise(xy, U, offset=offset)
    crossings = _sign_crossings(xs, u, rising=True)
    return float(crossings[-1]) if crossings else float("nan")


def separation_location(xy, U, offset: float = 0.05):
    """Estimate separation point x/H (first attached -> reversed crossing)."""
    xs, u = near_wall_streamwise(xy, U, offset=offset)
    crossings = _sign_crossings(xs, u, rising=False)
    return float(crossings[0]) if crossings else float("nan")


def reattachment_nearwall(x, y, U, dist_bottom, band: float = 0.04,
                          x_range=(0.5, 7.8), nbins: int = 80):
    """Leeward reattachment x/H from the sign of the near-bottom-wall velocity.

    More faithful and geometry-robust than sampling an analytic hill surface:
    it uses the actual cells adjacent to the bottom wall (``dist_bottom`` = each
    cell's distance to the bottomWall patch), bins them in x, and finds where
    the mean near-wall streamwise velocity recovers from reversed (U<0) to
    attached (U>0). Applied identically to RANS and DNS (same mesh), so the
    comparison is consistent.

    Returns the leeward-most negative->positive crossing (end of the main
    recirculation), or np.nan if no reversed-flow band is found.
    """
    x = np.asarray(x); U = np.asarray(U); d = np.asarray(dist_bottom)
    sel = d < band
    if sel.sum() < nbins:
        return float("nan")
    xs, us = x[sel], U[sel]
    edges = np.linspace(*x_range, nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(xs, edges) - 1
    prof = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.any():
            prof[b] = np.mean(us[m])
    good = np.isfinite(prof)
    cx, cp = centres[good], prof[good]
    # last negative->positive crossing = downstream end of the bubble
    cross = [cx[i] - cp[i] * (cx[i + 1] - cx[i]) / (cp[i + 1] - cp[i])
             for i in range(len(cp) - 1) if cp[i] < 0 <= cp[i + 1]]
    return float(cross[-1]) if cross else float("nan")


def bubble_summary(xy, U, offset: float = 0.05) -> dict:
    """Separation/reattachment/length summary of the recirculation bubble."""
    xsep = separation_location(xy, U, offset=offset)
    xreatt = reattachment_location(xy, U, offset=offset)
    length = (xreatt - xsep) if np.isfinite(xsep) and np.isfinite(xreatt) else float("nan")
    return {"x_separation": xsep, "x_reattachment": xreatt, "bubble_length": length}
