#!/usr/bin/env python3
"""Inspect the downloaded periodic-hill DNS reference data.

This is a *reporting* script: it prints the structure (files, shapes, column
ranges) of whatever DNS cases are present on disk, and sanity-checks the
column interpretation used by mlrans.dns. It does not assume file contents
beyond what has been verified; anything provisional is flagged.

Run from the project root:
    python analysis/01_inspect_dns.py

Prerequisite:
    scripts/fetch_dns_data.sh all      # (or a subset of cases)
"""

from __future__ import annotations

import sys

import numpy as np

# Make the src/ package importable without installation.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from mlrans import dns, paths  # noqa: E402


def main() -> int:
    print("=" * 70)
    print("Periodic-hill DNS reference data — structure inspection")
    print(f"Re_H = {paths.REYNOLDS_H:.0f}  (Ub={paths.BULK_VELOCITY}, "
          f"H={paths.HILL_HEIGHT_M} m, nu={paths.NU})")
    print("=" * 70)

    present = dns.available_cases()
    if not present:
        print("\nNo DNS cases found on disk.")
        print(f"Expected under: {paths.DNS_5CASES_DIR}")
        print("Download them with:  scripts/fetch_dns_data.sh all")
        return 1

    print(f"\nCases present ({len(present)}): {', '.join(present)}")
    missing = [c for c in paths.DNS_CASE_NAMES if c not in present]
    if missing:
        print(f"Cases not yet downloaded: {', '.join(missing)}")

    # Per-file structural summary for each present case.
    for case in present:
        print("\n" + "-" * 70)
        print(f"CASE: {case}   dir: {paths.dns_case_dir(case)}")
        print("-" * 70)
        for fname in dns.DNS_FILES:
            path = paths.dns_case_dir(case) / fname
            if not path.exists():
                print(f"  {fname:18s}  MISSING")
                continue
            info = dns.summarise_file(path)
            print(f"  {info['file']:18s}  shape={info['shape']}")
            print(f"      col_min: {info['col_min']}")
            print(f"      col_max: {info['col_max']}")

    # Physical sanity check on the confirmed mean-field interpretation, using
    # the first available case: separated flow MUST show reverse flow (U<0)
    # in the recirculation region behind the hill.
    case = present[0]
    d = dns.load_dns_case(case)
    print("\n" + "=" * 70)
    print(f"Sanity check on mean-field columns [{case}]")
    print("=" * 70)
    print(f"  n_points           : {d.n_points}")
    print(f"  x/H range          : [{d.x.min():.3f}, {d.x.max():.3f}]  (expect ~0..9)")
    print(f"  y/H range          : [{d.y.min():.3f}, {d.y.max():.3f}]  (expect ~0..3.036)")
    print(f"  U/Ub range         : [{d.U.min():.3f}, {d.U.max():.3f}]")
    frac_reverse = float(np.mean(d.U < 0.0))
    print(f"  reverse-flow cells : {100*frac_reverse:.1f}%  (U<0 => recirculation present)")
    print(f"  |W|/Ub max         : {np.abs(d.W).max():.4f}  (expect ~0, spanwise homogeneous)")
    if d.uv is not None:
        print(f"  <u'v'>/Ub^2 range  : [{d.uv.min():.4f}, {d.uv.max():.4f}]  "
              f"(expect mostly negative)")
    else:
        print("  <u'v'> (Reynolds shear stress): rms_files1.dat not present")

    ok = (frac_reverse > 0.0) and (np.abs(d.W).max() < 0.05)
    print("\nRESULT:", "PASS — column interpretation is physically consistent."
          if ok else "WARN — unexpected ranges; re-check column mapping.")
    print("\nNote: normal-stress columns in rms_files*.dat remain PROVISIONAL;")
    print("only <u'v'> (rms_files1 col 5) is relied upon downstream.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
