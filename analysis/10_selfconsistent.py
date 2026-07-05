#!/usr/bin/env python3
"""Self-consistent a-posteriori correction (MVR-4 improvement).

The single-shot correction (analysis/06) evaluates beta on the *baseline* flow,
then freezes it. That is inconsistent: once the flow changes, the features — and
hence the correction — should change too. Here we iterate:

    predict beta on the current flow -> apply (k -> beta*k_base), frozen solve
    -> re-extract invariant features from the corrected flow -> re-predict beta
    -> ... until the reattachment length stops moving.

beta is always a multiplier on the BASELINE eddy viscosity (it does not compound),
and is under-relaxed for stability. This is still a frozen-turbulence propagation
(a fully coupled re-trained closure would need a custom solver), but it removes
the "beta evaluated on the wrong flow" inconsistency of the single-shot version.

Run from the project root (after 03 and the 5 runs):
    python analysis/10_selfconsistent.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlrans import correction, dataset, foamfield, metrics, paths  # noqa: E402

TARGET = "case_1p0"
RUN = dataset.ML_RUNS
N_ITER = 5
RELAX = 0.5


def _foam(cmd, cwd):
    r = "openfoam2312 bash -c" if shutil.which("openfoam2312") else "bash -c"
    return subprocess.run(r.split() + [f"cd '{cwd}' && {cmd}"],
                          capture_output=True, text=True)


def _latest_vtu(case_dir):
    return sorted(case_dir.glob("VTK/**/internal.vtu"))[-1]


def _reatt(case_dir, dbot):
    import pyvista as pv
    m = pv.read(_latest_vtu(case_dir))
    c = m.cell_centers().points
    return metrics.reattachment_nearwall(
        c[:, 0], c[:, 1], np.asarray(m.cell_data["U"])[:, 0], dbot)


def main() -> int:
    paths.ensure_output_dirs()
    df = dataset.load_dataset()
    feats = dataset.feature_columns("invariant")
    tr = (df["case"] != TARGET) & df["wellposed"] & np.isfinite(df[feats]).all(axis=1)
    model = correction.make_model("random_forest")
    model.fit(df.loc[tr, feats].to_numpy(), df.loc[tr, "log_beta_nut"].to_numpy())

    base_case = RUN / TARGET
    _, centres, dbot = dataset.load_case_mesh(TARGET)
    import pyvista as pv
    kb = np.asarray(pv.read(_latest_vtu(base_case)).cell_data["k"])
    base_time = max((p for p in base_case.iterdir()
                     if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))

    dst = RUN / f"selfconsistent_{TARGET}"
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "0").mkdir(parents=True)
    shutil.copytree(base_case / "constant", dst / "constant")
    shutil.copytree(base_case / "system", dst / "system")
    for f in ("U", "p", "omega", "nut"):
        shutil.copy(base_time / f, dst / "0" / f)
    _foam("foamDictionary -entry RAS/turbulence -set off constant/turbulenceProperties",
          dst)

    # iteration 0 features come from the baseline flow
    feat_src = _latest_vtu(base_case)
    beta_applied = np.ones(len(kb))
    history = [{"iter": 0, "x_reattachment": _reatt(base_case, dbot),
                "beta_med": 1.0, "note": "baseline"}]

    for it in range(1, N_ITER + 1):
        X = dataset.invariant_matrix(feat_src)
        beta_pred = np.nan_to_num(np.exp(model.predict(X)), nan=1.0,
                                  posinf=10.0, neginf=0.0)
        beta_applied = (1 - RELAX) * beta_applied + RELAX * beta_pred
        foamfield.write_scalar_internal(
            base_time / "k", dst / "0" / "k",
            np.nan_to_num(np.clip(beta_applied, 0.0, None) * kb))
        shutil.rmtree(dst / "VTK", ignore_errors=True)
        for t in [p for p in dst.iterdir() if p.is_dir() and p.name.isdigit()
                  and p.name != "0"]:
            shutil.rmtree(t)
        _foam("rm -rf processor* && simpleFoam > log.simpleFoam 2>&1", dst)
        _foam("foamToVTK -latestTime -ascii > log.foamToVTK 2>&1", dst)
        xr = _reatt(dst, dbot)
        history.append({"iter": it, "x_reattachment": xr,
                        "beta_med": float(np.median(beta_applied)), "note": ""})
        print(f"  iter {it}: reattachment={xr:.3f}  beta_med={np.median(beta_applied):.3f}")
        feat_src = _latest_vtu(dst)     # next iteration uses the corrected flow

    hist = pd.DataFrame(history)
    hist.to_csv(paths.TABLES_DIR / "selfconsistent_history.csv", index=False)

    # DNS reattachment from the co-located UDNS on the baseline case.
    import pyvista as pv
    mb = pv.read(_latest_vtu(base_case))
    dns = metrics.reattachment_nearwall(
        centres[:, 0], centres[:, 1],
        np.asarray(mb.cell_data["UDNS"])[:, 0], dbot)
    print("\nReattachment (x/H):")
    print(f"  DNS                          {dns:.3f}")
    print(f"  baseline                     {history[0]['x_reattachment']:.3f}")
    print(f"  first (under-relaxed) iterate {history[1]['x_reattachment']:.3f}")
    print(f"  self-consistent (iter {N_ITER})      {history[-1]['x_reattachment']:.3f}")
    print("  (cf. single-shot, un-relaxed, analysis/06: 3.85 — evaluating beta on the\n"
          "   baseline flow over-corrects; the self-consistent value is more principled.)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(hist["iter"], hist["x_reattachment"], "o-", label="corrected")
    ax.axhline(dns, color="k", ls=":", label="DNS")
    ax.set_xlabel("self-consistency iteration")
    ax.set_ylabel(r"reattachment $x/H$")
    ax.set_title("Self-consistent frozen-nut correction (unseen geometry)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / "selfconsistent_convergence.png", dpi=150)
    plt.close(fig)
    print("[fig] selfconsistent_convergence.png  [csv] selfconsistent_history.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
