#!/usr/bin/env python3
"""Flow-field visualisations (velocity contours + streamlines).

The most intuitive pictures in the project — the mean flow itself:

  flowfield_rans_vs_dns        baseline RANS vs DNS recirculation bubble
  flowfield_correction         baseline -> ML-corrected -> DNS (bubble shrinks)
  flowfield_other_flows        the other separated flows used for cross-flow
  flowfield_geometry_sweep     the five hill shapes (alpha = 0.5 .. 1.5)
  correction_field_map         where the correction acts, on the hill

Produced with matplotlib/pyvista from the co-located VTK fields and the McConkey
dataset (scriptable and reproducible, unlike a ParaView screenshot).

Run from the project root (after scripts/run_ml_cases.sh, 06 and 03):
    python analysis/19_flowviz.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mlrans import dataset, grid, paths  # noqa: E402

TARGET = "case_1p0"
RUN = dataset.ML_RUNS
MCK = paths.DATA_EXTERNAL / "mcconkey-dataset" / "komegasst"
DOCS = paths.ROOT / "docs" / "figures"
UB = dataset.UB


def _save(fig, name):
    for d in (paths.FIGURES_DIR, DOCS):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}.png")


def _cell_U(case):
    import pyvista as pv
    m = pv.read(sorted((RUN / case).glob("VTK/**/internal.vtu"))[-1])
    c = m.cell_centers().points
    return np.column_stack([c[:, 0], c[:, 1]]), m


# ---- grid + streamline panels (for the alpha=1 hill, has hill_surface) ------
def _grid_fields(xy, U, V, nx=380, ny=150):
    xs = np.linspace(0, 9, nx); ys = np.linspace(0, grid.Y_TOP, ny)
    X, Y = np.meshgrid(xs, ys); q = np.column_stack([X.ravel(), Y.ravel()])
    Ug = grid.interpolate(xy, U, q).reshape(X.shape)
    Vg = grid.interpolate(xy, V, q).reshape(X.shape)
    below = Y < grid.hill_surface(X.ravel()).reshape(X.shape)
    Ug[below] = np.nan; Vg[below] = np.nan
    return xs, ys, X, Y, Ug, Vg


def _panel(ax, xs, ys, X, Y, Ug, Vg, title):
    cf = ax.contourf(X, Y, Ug / UB, levels=np.linspace(-0.4, 1.2, 30),
                     cmap="RdBu_r", extend="both")
    ax.contour(X, Y, Ug, levels=[0.0], colors="k", linewidths=1.2)
    ax.streamplot(xs, ys, np.nan_to_num(Ug), np.nan_to_num(Vg), density=1.3,
                  color="k", linewidth=0.5, arrowsize=0.6)
    xf = np.linspace(0, 9, 400)
    ax.fill_between(xf, 0, grid.hill_surface(xf), color="0.45", zorder=5)
    ax.plot(xf, grid.hill_surface(xf), color="k", lw=1, zorder=6)
    ax.set_xlim(0, 9); ax.set_ylim(0, grid.Y_TOP); ax.set_aspect("equal")
    ax.set_ylabel("$y/H$"); ax.set_title(title, fontsize=11)
    return cf


# ---- robust rendering for arbitrary geometries ------------------------------
# Interpolate a scattered field onto a regular grid and blank grid cells that
# are far from any data point (a concave-hull / alpha-shape style mask). This
# reconstructs the true fluid domain for any mesh — no triangulation holes and
# no degenerate triangles on strongly-graded or near-coincident meshes.
def _grid_mask(x, y, vals, nx=460):
    from scipy.spatial import cKDTree
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    ny = max(50, int(nx * (ymax - ymin) / (xmax - xmin)))
    xs = np.linspace(xmin, xmax, nx); ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([x, y]); gpts = np.column_stack([X.ravel(), Y.ravel()])
    Vg = grid.interpolate(pts, vals, gpts).reshape(X.shape)
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    dist, _ = cKDTree(pts).query(gpts)
    Vg[(dist.reshape(X.shape) > 2.5 * np.hypot(dx, dy))] = np.nan   # blank solid
    return xs, ys, X, Y, Vg


def _field_panel(ax, x, y, U, title, uref, xlim, ylim):
    xs, ys, X, Y, Ug = _grid_mask(x, y, U)
    cf = ax.contourf(X, Y, Ug / uref, levels=np.linspace(-0.4, 1.2, 25),
                     cmap="RdBu_r", extend="both")
    ax.contour(X, Y, Ug, levels=[0.0], colors="k", linewidths=1.0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.set_anchor("W")      # equal scale, left-aligned
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    return cf


# ============================ figures =======================================
def fig_rans_vs_dns():
    xy, m = _cell_U(TARGET)
    U = np.asarray(m.cell_data["U"]); Ud = np.asarray(m.cell_data["UDNS"])
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    _panel(ax[0], *_grid_fields(xy, U[:, 0], U[:, 1]), "RANS  (k-ω SST)")
    cf = _panel(ax[1], *_grid_fields(xy, Ud[:, 0], Ud[:, 1]), "DNS  (reference)")
    ax[1].set_xlabel("$x/H$")
    fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02,
                 label=r"streamwise velocity  $U/U_b$")
    fig.suptitle("Mean flow over the periodic hill "
                 "(black line = edge of the reversed-flow region)", y=0.98)
    _save(fig, "flowfield_rans_vs_dns")


def fig_correction():
    xy, m = _cell_U(TARGET)
    U = np.asarray(m.cell_data["U"]); Ud = np.asarray(m.cell_data["UDNS"])
    _, mc = _cell_U(f"prop_pred_{TARGET}")
    Uc = np.asarray(mc.cell_data["U"])
    fig, ax = plt.subplots(3, 1, figsize=(9, 8.4), sharex=True)
    _panel(ax[0], *_grid_fields(xy, U[:, 0], U[:, 1]), "RANS baseline (k-ω SST)")
    _panel(ax[1], *_grid_fields(xy, Uc[:, 0], Uc[:, 1]),
           "ML-corrected (a-priori propagation, unseen geometry)")
    cf = _panel(ax[2], *_grid_fields(xy, Ud[:, 0], Ud[:, 1]), "DNS (reference)")
    ax[2].set_xlabel("$x/H$")
    fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02,
                 label=r"streamwise velocity  $U/U_b$")
    fig.suptitle("The ML correction shortens the recirculation bubble toward DNS",
                 y=0.99)
    _save(fig, "flowfield_correction")


def _mck(case, field):
    return np.load(MCK / f"komegasst_{case}_{field}.npy")


def fig_other_flows():
    flows = [("PHLL_case_1p0", "periodic hill"),
             ("CBFS_13700", "curved backward-facing step"),
             ("CNDV_12600", "converging-diverging channel"),
             ("BUMP_h31", "bump")]
    # normalise each flow's coordinates by its own height so shapes are
    # comparable despite very different physical scales.
    data = []
    for c, ttl in flows:
        x, y, U = _mck(c, "Cx"), _mck(c, "Cy"), _mck(c, "Ux")
        s = y.max() - y.min()
        data.append((ttl, (x - x.min()) / s, (y - y.min()) / s, U,
                     np.nanpercentile(np.abs(U), 95)))
    maxw = max(xn.max() for _, xn, _, _, _ in data)
    fig, ax = plt.subplots(len(flows), 1, figsize=(9, 8.5))
    for a, (ttl, xn, yn, U, uref) in zip(ax, data):
        cf = _field_panel(a, xn, yn, U, ttl, uref, (0, maxw), (0, 1.05))
    fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.02,
                 label="streamwise velocity (per-flow normalised)")
    fig.suptitle("The separated flows used for the cross-flow test are genuinely "
                 "different geometries", y=0.98)
    _save(fig, "flowfield_other_flows")


def fig_geometry_sweep():
    cases = ["case_0p5", "case_0p8", "case_1p0", "case_1p2", "case_1p5"]
    data = []
    for c in cases:
        xy, m = _cell_U(c)
        data.append((c, xy[:, 0], xy[:, 1], np.asarray(m.cell_data["UDNS"])[:, 0]))
    maxlen = max(x.max() for _, x, _, _ in data) + 0.15
    fig, ax = plt.subplots(len(cases), 1, figsize=(9, 9))
    for a, (c, x, y, U) in zip(ax, data):
        cf = _field_panel(a, x, y, U, f"α = {c.split('_')[1].replace('p', '.')}",
                          UB, (0, maxlen), (0, grid.Y_TOP + 0.05))
    fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.02,
                 label=r"DNS streamwise velocity  $U/U_b$")
    fig.suptitle("Separation over the five hill shapes at the same scale "
                 "(steeper hills are longer)", y=0.98)
    _save(fig, "flowfield_geometry_sweep")


def fig_correction_map():
    df = dataset.load_dataset()
    c = df[df["case"] == TARGET]
    fin = np.isfinite(c["beta_nut"].to_numpy())        # drop masked low-shear cells
    x, y = c["x"].to_numpy()[fin], c["y"].to_numpy()[fin]
    beta = np.clip(c["beta_nut"].to_numpy()[fin], 0, 3)
    _, _, X, Y, Bg = _grid_mask(x, y, beta)
    # light NaN-aware smoothing: beta_nut is a noisy point-wise ratio, so show
    # the spatial pattern (not per-cell spikes). The solid region stays blank.
    from scipy.ndimage import gaussian_filter
    mask = np.isfinite(Bg)
    num = gaussian_filter(np.where(mask, Bg, 0.0), 1.6)
    den = gaussian_filter(mask.astype(float), 1.6)
    Bs = np.where(mask, num / np.maximum(den, 1e-6), np.nan)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    cf = ax.contourf(X, Y, Bs, levels=np.linspace(0.5, 2.0, 16),
                     cmap="RdBu_r", extend="both")
    xf = np.linspace(0, 9, 400)
    ax.fill_between(xf, 0, grid.hill_surface(xf), color="0.45", zorder=5)
    ax.plot(xf, grid.hill_surface(xf), color="k", lw=1, zorder=6)
    ax.set_xlim(0, 9); ax.set_ylim(0, grid.Y_TOP); ax.set_aspect("equal")
    ax.set_xlabel("$x/H$"); ax.set_ylabel("$y/H$")
    fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.02,
                 label=r"correction  $\beta_{\nu_t}=\nu_{t,DNS}/\nu_{t,RANS}$")
    ax.set_title("Where the model needs correcting: β > 1 (red) = RANS under-mixes,\n"
                 "concentrated in the separated shear layer", fontsize=11)
    _save(fig, "correction_field_map")


def main() -> int:
    paths.ensure_output_dirs()
    print("Rendering flow-field figures...")
    fig_rans_vs_dns()
    fig_correction()
    fig_other_flows()
    fig_geometry_sweep()
    fig_correction_map()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
