"""Plotting: baseline RANS-vs-DNS velocity profiles and error maps.

Matplotlib only (no seaborn). Every function takes an explicit output path and
saves a figure; nothing is fabricated — callers pass real interpolated data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / script-safe backend
import matplotlib.pyplot as plt
import numpy as np

from . import grid


def _draw_hill(ax, x_range=(0.0, 9.0), n=400, color="0.3"):
    """Shade the solid hill region under the lower-wall profile."""
    xs = np.linspace(*x_range, n)
    ys = grid.hill_surface(xs)
    ax.fill_between(xs, 0, ys, color=color, zorder=5, linewidth=0)
    ax.plot(xs, ys, color="k", lw=1.0, zorder=6)


def plot_velocity_profiles(rans_xy, rans_U, dns_xy, dns_U, out_path: Path,
                           stations=grid.PROFILE_STATIONS, u_scale: float = 1.0):
    """Overlay U(y) profiles at each x/H station: RANS vs DNS.

    Each profile is drawn at its station, offset horizontally by U/Ub * u_scale,
    the conventional periodic-hill presentation.
    """
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, xs in enumerate(stations):
        y, ur = grid.vertical_profile(rans_xy, rans_U, xs)
        _, ud = grid.vertical_profile(dns_xy, dns_U, xs)
        lbl_r = "RANS (k-omega SST)" if i == 0 else None
        lbl_d = "DNS" if i == 0 else None
        ax.plot(xs + u_scale * ud, y, color="k", lw=1.4, label=lbl_d, zorder=4)
        ax.plot(xs + u_scale * ur, y, color="tab:red", lw=1.2, ls="--",
                label=lbl_r, zorder=4)
        ax.axvline(xs, color="0.85", lw=0.6, zorder=1)

    _draw_hill(ax)
    ax.set_xlabel(r"$x/H \;+\; U/U_b$")
    ax.set_ylabel(r"$y/H$")
    ax.set_title("Streamwise velocity profiles: baseline RANS vs DNS "
                 r"(periodic hill, $Re_H=5600$)")
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(0, grid.Y_TOP)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_scatter_field(x, y, val, out_path: Path, title="", cbar_label="",
                       cmap="viridis", vmin=None, vmax=None, symmetric=False,
                       s=6):
    """Scatter cell-centre values directly (geometry-agnostic, no hill mask).

    Works for any hill slope because cells only exist in the fluid region.
    """
    v = np.asarray(val, dtype=float)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    if symmetric:
        m = np.nanmax(np.abs(v))
        vmin, vmax = -m, m
    sc = ax.scatter(x, y, c=v, cmap=cmap, vmin=vmin, vmax=vmax, s=s,
                    edgecolors="none")
    fig.colorbar(sc, ax=ax, label=cbar_label, pad=0.01)
    ax.set_xlabel(r"$x/H$")
    ax.set_ylabel(r"$y/H$")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_feature_importance(importances: dict, out_path: Path,
                            title="Feature importance"):
    """Horizontal bar chart of feature importances / |coefficients|."""
    names = list(importances.keys())[::-1]
    vals = [importances[n] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(names) + 1))
    ax.barh(names, vals, color="tab:blue")
    ax.set_xlabel("relative importance")
    ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pred_vs_true(y_true, y_pred, out_path: Path, title="", label="value"):
    """Scatter of predicted vs true, with the y=x reference line."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.scatter(yt, yp, s=4, alpha=0.25, edgecolors="none")
    lo, hi = np.percentile(np.concatenate([yt, yp]), [1, 99])
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"true {label}")
    ax.set_ylabel(f"predicted {label}")
    ax.set_title(title)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_generalization_bars(metrics_df, out_path: Path,
                             title="Out-of-geometry generalisation (leave-one-hill-out)"):
    """Grouped bar chart of held-out R^2 per model per geometry.

    ``metrics_df`` has columns: model, held_out, r2. This is the headline
    honesty figure: how well each model predicts a hill slope it never saw.
    """
    models = list(dict.fromkeys(metrics_df["model"]))
    cases = sorted(metrics_df["held_out"].unique())
    x = np.arange(len(cases))
    w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, m in enumerate(models):
        sub = metrics_df[metrics_df["model"] == m].set_index("held_out")
        vals = [sub.loc[c, "r2"] if c in sub.index else np.nan for c in cases]
        ax.bar(x + i * w, vals, w, label=m)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(cases)
    ax.set_ylabel(r"held-out $R^2$ on $\log\beta_{\nu_t}$")
    ax.set_xlabel("geometry held out of training (unseen hill slope)")
    ax.set_title(title)
    ax.set_ylim(min(-0.5, float(np.nanmin(metrics_df["r2"])) - 0.1), 1.0)
    ax.legend(loc="lower center", ncol=len(models), framealpha=0.9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_error_map(X, Y, error, out_path: Path, mask=None,
                   title="RANS error map", cbar_label=r"$U_{RANS}-U_{DNS}$ [$U_b$]",
                   symmetric=True, cmap="RdBu_r"):
    """Filled contour of a (signed) error field on the regular grid.

    ``mask`` (True = fluid) blanks the solid hill region. A symmetric colour
    scale centres zero, so red/blue directly read as RANS over/under-prediction.
    """
    E = np.array(error, dtype=float)
    if mask is not None:
        E = np.where(mask, E, np.nan)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    if symmetric:
        vmax = np.nanmax(np.abs(E))
        vmin = -vmax
    else:
        vmin, vmax = np.nanmin(E), np.nanmax(E)
    pc = ax.pcolormesh(X, Y, E, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    _draw_hill(ax)
    fig.colorbar(pc, ax=ax, label=cbar_label, pad=0.01)
    ax.set_xlabel(r"$x/H$")
    ax.set_ylabel(r"$y/H$")
    ax.set_title(title)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, grid.Y_TOP)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
