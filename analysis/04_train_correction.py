#!/usr/bin/env python3
"""Train and validate the interpretable eddy-viscosity correction (MVR-3).

Learns log(beta_nut) from RANS-local features and evaluates it with
LEAVE-ONE-GEOMETRY-OUT cross-validation (train on 4 hill slopes, test on the
unseen 5th). Compares two feature sets:

  * 'position' : includes absolute x/H, y/H (strong in-distribution, but leans
                 on location so it transfers poorly to new geometries);
  * 'invariant': Galilean-invariant, geometry-agnostic physics features only.

Only the well-posed cells (positive, finite inferred nut) are used; the
counter-gradient regions where the eddy-viscosity ansatz breaks down are
excluded and reported.

Run from the project root (after analysis/03_build_dataset.py):
    python analysis/04_train_correction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlrans import correction, dataset, paths, plots  # noqa: E402

MODELS = ("ridge", "random_forest", "gradient_boosting")
FEATURE_SETS = ("position", "invariant")
MAP_CASE = "case_1p0"


def _prepare(df, feats):
    finite = np.isfinite(df[feats]).all(axis=1) & np.isfinite(df["log_beta_nut"])
    mask = finite & df["wellposed"].to_numpy()
    d = df[mask].reset_index(drop=True)
    return d, d[feats].to_numpy(), d["log_beta_nut"].to_numpy(), d["case"].to_numpy()


def main() -> int:
    paths.ensure_output_dirs()
    df = dataset.load_dataset()

    all_rows = []
    for kind in FEATURE_SETS:
        feats = dataset.feature_columns(kind)
        data, X, y, groups = _prepare(df, feats)
        print(f"\n===== feature set: {kind}  ({len(feats)} features, "
              f"{len(data)} well-posed cells) =====")
        for name in MODELS:
            res = correction.geometry_wise_cv(X, y, groups, name)
            print(f"[{name:17s}] mean out-of-geometry R2={res['mean_r2']:.3f}  "
                  f"RMSE={res['mean_rmse']:.3f}")
            for case, r2 in zip(res["held_out_case"], res["fold_r2"]):
                all_rows.append({"features": kind, "model": name,
                                 "held_out": case, "r2": r2})
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(paths.TABLES_DIR / "correction_cv_metrics.csv", index=False)
    print(f"\n[csv] correction_cv_metrics.csv")

    # --- headline comparison: RF, position vs invariant, per geometry ---
    rf = metrics[metrics["model"] == "random_forest"]
    comp = rf.pivot(index="held_out", columns="features", values="r2")
    print("\nRandom-forest out-of-geometry R2 (position vs invariant):")
    print(comp.round(3).to_string())
    _plot_feature_set_comparison(rf, paths.FIGURES_DIR / "feature_set_comparison.png")
    print("[fig] feature_set_comparison.png")

    # --- interpretability + maps use the INVARIANT set (transferable) ----
    feats = dataset.feature_columns("invariant")
    data, X, y, groups = _prepare(df, feats)
    best = (metrics[metrics.features == "invariant"]
            .groupby("model")["r2"].mean().idxmax())
    print(f"\nBest invariant-feature model: {best}")

    oof = correction.geometry_wise_oof(X, y, groups, best)
    plots.plot_pred_vs_true(
        y, oof, paths.FIGURES_DIR / "correction_pred_vs_true.png",
        title=f"Out-of-geometry prediction — invariant features ({best})",
        label=r"$\log\beta_{\nu_t}$")

    imp_model = correction.make_model(best if best != "ridge" else "random_forest")
    imp_model.fit(X, y)
    imp = correction.feature_importances(imp_model, feats)
    plots.plot_feature_importance(
        imp, paths.FIGURES_DIR / "correction_feature_importance.png",
        title=f"Invariant-feature importance for log(beta_nut)")
    print("  ranked invariant features:", " > ".join(list(imp.keys())[:5]))

    cmask = data["case"].to_numpy() == MAP_CASE
    xc, yc = data.loc[cmask, "x"], data.loc[cmask, "y"]
    for tag, vals in [("true", np.exp(y[cmask])), ("pred", np.exp(oof[cmask]))]:
        plots.plot_scatter_field(
            xc, yc, vals, paths.FIGURES_DIR / f"beta_{tag}_{MAP_CASE}.png",
            title=rf"$\beta_{{\nu_t}}$ {tag} — invariant, unseen ({MAP_CASE})",
            cbar_label=r"$\beta_{\nu_t}$", cmap="RdBu_r", vmin=0, vmax=2)
    print(f"[fig] beta_true/pred_{MAP_CASE}.png, correction_pred_vs_true.png, "
          f"correction_feature_importance.png")
    return 0


def _plot_feature_set_comparison(rf_metrics, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    comp = rf_metrics.pivot(index="held_out", columns="features", values="r2")
    cases = list(comp.index)
    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.2, comp["position"], 0.4, label="position (x/H, y/H incl.)")
    ax.bar(x + 0.2, comp["invariant"], 0.4, label="invariant (geometry-agnostic)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel(r"held-out $R^2$ on $\log\beta_{\nu_t}$")
    ax.set_xlabel("geometry held out (unseen hill slope)")
    ax.set_title("Feature-set transfer: does dropping position help extrapolation? "
                 "(random forest)")
    ax.set_ylim(min(-0.5, float(comp.min().min()) - 0.1), 1.0)
    ax.legend(loc="lower center", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
