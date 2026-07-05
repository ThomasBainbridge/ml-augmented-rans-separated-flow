#!/usr/bin/env python3
"""Build the ML feature/target dataset from the 5-geometry RANS+DNS runs.

Reads each co-located VTK export, computes RANS-local features and the
Boussinesq-inferred beta_nut correction target, and writes a single tidy table
to data/processed/ plus a short per-geometry summary.

Run from the project root (after scripts/run_ml_cases.sh):
    python analysis/03_build_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlrans import dataset  # noqa: E402


def main() -> int:
    print("Building feature/target dataset from co-located RANS+DNS runs...")
    df = dataset.build_all()

    print("\nPer-geometry summary (valid beta_nut = finite target):")
    print(f"{'case':10s} {'cells':>7s} {'valid':>7s} "
          f"{'beta med':>9s} {'beta p10':>9s} {'beta p90':>9s}")
    for c, g in df.groupby("case"):
        valid = g["beta_nut"].replace([np.inf, -np.inf], np.nan).dropna()
        vt = g["log_beta_nut"].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"{c:10s} {len(g):7d} {len(vt):7d} "
              f"{valid.median():9.3f} {valid.quantile(.1):9.3f} "
              f"{valid.quantile(.9):9.3f}")

    feats = dataset.feature_columns()
    print(f"\nFeature columns ({len(feats)}): {feats}")
    print("Target: log_beta_nut  (beta_nut = nut_DNS_eff / nut_RANS)")
    print("\nNext: python analysis/04_train_correction.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
