#!/usr/bin/env python3
"""Statistical honesty of the generalisation claim (MVR-3 supplement).

Addresses three ways the headline R^2 could mislead:
  1. Effective sample size: leave-one-geometry-out has only 5 groups, and the
     ~15k cells within a geometry are strongly spatially correlated, so the
     per-fold R^2 has large uncertainty. We report the spread across folds and a
     bootstrap CI over the 5 fold values (not the correlated cells).
  2. Trivial baseline: R^2 vs a constant predictor (training-median log beta),
     so the reader can see how much the model actually beats "predict the mean".
  3. Honest importance: permutation importance (model-agnostic, unbiased) rather
     than random-forest impurity importance (biased toward high-cardinality
     features).

Run from the project root (after 03):
    python analysis/08_stats_rigor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402

from mlrans import correction, dataset, paths, plots  # noqa: E402

FEATURES = "invariant"


def main() -> int:
    paths.ensure_output_dirs()
    df = dataset.load_dataset()
    feats = dataset.feature_columns(FEATURES)
    m = df["wellposed"] & np.isfinite(df[feats]).all(axis=1)
    d = df[m].reset_index(drop=True)
    X, y, g = d[feats].to_numpy(), d["log_beta_nut"].to_numpy(), d["case"].to_numpy()

    logo = LeaveOneGroupOut()
    fold_r2, naive_r2, held = [], [], []
    perm_rows = []
    for tr, te in logo.split(X, y, g):
        model = correction.make_model("random_forest")
        model.fit(X[tr], y[tr])
        fold_r2.append(r2_score(y[te], model.predict(X[te])))
        # trivial baseline: predict the training median everywhere
        const = np.full(te.shape, np.median(y[tr]))
        naive_r2.append(r2_score(y[te], const))
        held.append(str(g[te][0]))
        # permutation importance on the held-out geometry (subsample for speed)
        idx = np.random.default_rng(0).choice(te, size=min(4000, te.size), replace=False)
        pi = permutation_importance(model, X[idx], y[idx], n_repeats=5,
                                    random_state=0, scoring="r2")
        perm_rows.append(pi.importances_mean)

    fold_r2 = np.array(fold_r2); naive_r2 = np.array(naive_r2)
    # bootstrap CI over the 5 fold values (the ~independent unit is the geometry)
    rng = np.random.default_rng(0)
    boot = [np.mean(rng.choice(fold_r2, size=len(fold_r2), replace=True))
            for _ in range(5000)]
    lo, hi = np.percentile(boot, [5, 95])

    print("Leave-one-geometry-out (random forest, invariant features):")
    for c, r, n in zip(held, fold_r2, naive_r2):
        print(f"  held-out {c:10s}  R2={r:6.3f}   (constant-median baseline R2={n:6.3f})")
    print(f"\n  mean R2 = {fold_r2.mean():.3f}  (std {fold_r2.std():.3f}, "
          f"min {fold_r2.min():.3f})")
    print(f"  90% bootstrap CI over the 5 geometries: [{lo:.3f}, {hi:.3f}]")
    print(f"  trivial constant-median baseline: mean R2 = {naive_r2.mean():.3f}")
    print("  Effective sample size for generalisation ~= 5 geometries (cells "
          "within a geometry are spatially correlated), so this CI is wide "
          "by construction — reported honestly rather than hidden.")

    # permutation importance (averaged over folds)
    perm = np.mean(perm_rows, axis=0)
    order = np.argsort(perm)[::-1]
    imp = {feats[i]: float(perm[i]) for i in order}
    print("\nPermutation importance (mean R2 drop, averaged over folds):")
    for k, v in imp.items():
        print(f"  {k:18s} {v:.4f}")
    plots.plot_feature_importance(
        imp, paths.FIGURES_DIR / "correction_permutation_importance.png",
        title="Permutation importance (unbiased) for log(beta_nut)")

    pd.DataFrame({"held_out": held, "rf_r2": fold_r2,
                  "constant_baseline_r2": naive_r2}).to_csv(
        paths.TABLES_DIR / "generalization_rigor.csv", index=False)
    print("\n[fig] correction_permutation_importance.png  "
          "[csv] generalization_rigor.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
