#!/usr/bin/env python3
"""Baseline RANS vs DNS comparison -> the MVR-1 "failure map".

Produces, from REAL data only:
  * velocity-profile overlay (RANS vs DNS) at the standard x/H stations;
  * a streamwise-velocity error map (RANS - DNS) with the hill masked;
  * a separation/reattachment/bubble-length table (RANS vs DNS).

The script refuses to run if either side is missing, printing the exact command
to produce it — it never invents results.

Run from the project root:
    python analysis/02_compare_rans_dns.py [--dns-case case_1p0]

Prerequisites:
    scripts/fetch_dns_data.sh case_1p0        # DNS reference
    scripts/run_baseline_sst.sh               # RANS solve
    scripts/export_latest_fields.sh           # foamToVTK export
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlrans import dns, foam, grid, metrics, paths, plots  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dns-case", default="case_1p0",
                    help="DNS geometry to compare against (default: case_1p0).")
    args = ap.parse_args()

    paths.ensure_output_dirs()

    # --- DNS side --------------------------------------------------------
    if args.dns_case not in dns.available_cases():
        print(f"DNS case '{args.dns_case}' not found on disk.")
        print(f"  Download it:  scripts/fetch_dns_data.sh {args.dns_case}")
        return 1
    d = dns.load_dns_case(args.dns_case)
    print(f"DNS   : {d.name}  ({d.n_points} points)")

    # --- RANS side -------------------------------------------------------
    try:
        r = foam.load_baseline()
    except FileNotFoundError as exc:
        print("Baseline RANS export not found.")
        print(f"  {exc}")
        print("  Produce it:  scripts/run_baseline_sst.sh && "
              "scripts/export_latest_fields.sh")
        return 1
    print(f"RANS  : k-omega SST  ({r.n_points} cells)")

    # --- 1) velocity-profile overlay ------------------------------------
    fig1 = paths.FIGURES_DIR / f"velocity_profiles_{d.name}.png"
    plots.plot_velocity_profiles(r.xy, r.U, d.xy, d.U, fig1)
    print(f"[fig] {fig1.relative_to(paths.ROOT)}")

    # --- 2) streamwise-velocity error map -------------------------------
    X, Y, mask = grid.regular_grid()
    q = np.column_stack([X.ravel(), Y.ravel()])
    Ur = grid.interpolate(r.xy, r.U, q).reshape(X.shape)
    Ud = grid.interpolate(d.xy, d.U, q).reshape(X.shape)
    err = Ur - Ud
    fig2 = paths.FIGURES_DIR / f"U_error_map_{d.name}.png"
    plots.plot_error_map(
        X, Y, err, fig2, mask=mask,
        title=r"Baseline RANS model-form error: $U_{RANS}-U_{DNS}$ "
              f"(periodic hill {d.name}, $Re_H=5600$)")
    print(f"[fig] {fig2.relative_to(paths.ROOT)}")

    # --- 3) reattachment table (wall-faithful near-wall method) ---------
    # Distance to the lower wall from the analytic hill surface (both fields
    # are compared with one consistent definition).
    def reatt(xy, U):
        d_bot = xy[:, 1] - grid.hill_surface(xy[:, 0])
        return metrics.reattachment_nearwall(xy[:, 0], xy[:, 1], U, d_bot)

    prof = metrics.profile_velocity_error(r.xy, r.U, d.xy, d.U)
    table = pd.DataFrame({
        "quantity": ["x_reattachment [x/H]", "U-profile overall RMSE [Ub]"],
        "RANS": [reatt(r.xy, r.U), prof["overall_rmse"]],
        "DNS": [reatt(d.xy, d.U), 0.0],
    })
    out_csv = paths.TABLES_DIR / f"bubble_metrics_{d.name}.csv"
    table.to_csv(out_csv, index=False)
    print(f"[csv] {out_csv.relative_to(paths.ROOT)}")

    print("\n" + table.to_string(index=False))
    print("\nPer-station U-profile RMSE (x/H -> RMSE in Ub):")
    for xs, e in prof["per_station_rmse"].items():
        print(f"  x/H={xs:>4}:  {e:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
