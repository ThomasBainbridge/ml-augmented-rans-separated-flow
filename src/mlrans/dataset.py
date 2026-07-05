"""Build the ML feature/target dataset from co-located RANS+DNS VTK exports.

Each geometry's VTK (produced by scripts/run_ml_cases.sh) contains, on the same
mesh and in the same cell order:
  * the baseline k-omega SST solution (U, p, k, omega, nut), and
  * the DNS reference (UDNS mean velocity, TauDNS Reynolds-stress tensor).

Because they are co-located, the correction target is computed without any
scattered interpolation. Velocity gradients (needed for strain/rotation
features and for the Boussinesq inversion) are computed on the mesh with
pyvista.

Coordinates are in hill-height units (the provided meshes use H = 1); the bulk
velocity is Ub = 0.020188 (Re_H = 5600).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import features, paths

# Operating point of the provided DNS meshes.
UB = 0.020188
H = 1.0

# TauDNS is a symmetric tensor. NOTE the ordering difference:
#   OpenFOAM ASCII order : (xx, xy, xz, yy, yz, zz)  -> xy at index 1
#   VTK (what pyvista reads): (xx, yy, zz, xy, yz, xz) -> xy at index 3
# We read via pyvista/VTK, so the Reynolds shear stress <u'v'> is index 3.
# (Verified: index 3 is the only component with the expected negative range.)
TAU_UV = 3

ML_RUNS = paths.CASES_DIR / "ml_runs"


def _latest_internal_vtu(case_dir: Path) -> Path:
    cands = sorted(case_dir.glob("VTK/**/internal.vtu"))
    if not cands:
        raise FileNotFoundError(
            f"No internal.vtu under {case_dir}/VTK. Run scripts/run_ml_cases.sh."
        )
    return cands[-1]


def _cell_gradient(mesh, field: str) -> np.ndarray:
    """Cell gradient of a field: (Ncells, 9) for a vector, (Ncells, 3) scalar.

    pyvista's derivative filter works on point data, so we move cell->point,
    differentiate, then average back to cells. The mild smoothing this implies
    is acceptable for a-priori feature construction.
    """
    pd_mesh = mesh.cell_data_to_point_data()
    pd_mesh.set_active_scalars(field, preference="point")
    grad = pd_mesh.compute_derivative(scalars=field)
    back = grad.point_data_to_cell_data()
    return np.asarray(back.cell_data["gradient"])


def _wall_distance(vtu_path: Path, centres: np.ndarray,
                   patches=("bottomWall.vtp", "topWall.vtp")) -> np.ndarray:
    """Nearest distance from each cell centre to the given wall patch(es).

    Uses the wall boundary patches exported alongside the internal mesh; falls
    back to a large distance if none are present.
    """
    import pyvista as pv
    from scipy.spatial import cKDTree

    bdir = vtu_path.parent / "boundary"
    pts = [pv.read(bdir / w).cell_centers().points for w in patches
           if (bdir / w).exists()]
    if not pts:
        return np.full(len(centres), 1.0)
    tree = cKDTree(np.vstack(pts))
    d, _ = tree.query(centres)
    return d


def invariant_matrix(vtu_path: Path) -> np.ndarray:
    """Compute the invariant feature matrix from any solved case VTK.

    Same columns/order as feature_columns('invariant'); used by the
    self-consistent coupling to re-evaluate features on the corrected flow.
    """
    import pyvista as pv
    mesh = pv.read(vtu_path)
    U = np.asarray(mesh.cell_data["U"])
    gradU = _cell_gradient(mesh, "U")
    gradp = _cell_gradient(mesh, "p")
    wall_d = _wall_distance(vtu_path, mesh.cell_centers().points)
    inv = features.invariant_features(
        U, gradU, gradp, np.asarray(mesh.cell_data["k"]),
        np.asarray(mesh.cell_data["omega"]), np.asarray(mesh.cell_data["nut"]),
        wall_d, nu=paths.NU)
    return inv[feature_columns("invariant")].to_numpy()


def load_case_mesh(case_name: str, run_root: Path | None = None):
    """Return (pyvista mesh, cell centres, distance-to-bottomWall) for a run."""
    import pyvista as pv
    run_root = run_root or ML_RUNS
    vtu = _latest_internal_vtu(run_root / case_name)
    mesh = pv.read(vtu)
    centres = mesh.cell_centers().points
    dbot = _wall_distance(vtu, centres, patches=("bottomWall.vtp",))
    return mesh, centres, dbot


def build_case_frame(case_name: str, run_root: Path | None = None) -> pd.DataFrame:
    """Assemble the per-cell feature/target table for one geometry."""
    import pyvista as pv

    run_root = run_root or ML_RUNS
    case_dir = run_root / case_name
    vtu_path = _latest_internal_vtu(case_dir)
    mesh = pv.read(vtu_path)

    need = {"U", "nut", "k", "omega", "UDNS", "TauDNS"}
    missing = need - set(mesh.cell_data.keys())
    if missing:
        raise KeyError(f"{case_name}: VTK missing fields {missing}. "
                       f"Have: {list(mesh.cell_data.keys())}")

    centres = mesh.cell_centers().points
    x, y = centres[:, 0] / H, centres[:, 1] / H

    U = np.asarray(mesh.cell_data["U"])
    nut_rans = np.asarray(mesh.cell_data["nut"])
    k = np.asarray(mesh.cell_data["k"])
    omega = np.asarray(mesh.cell_data["omega"])
    Udns = np.asarray(mesh.cell_data["UDNS"])
    tau = np.asarray(mesh.cell_data["TauDNS"])
    uv_dns = tau[:, TAU_UV]

    # --- features (RANS-local, non-dimensional) -------------------------
    feat = features.nondimensional_features(
        x, y, U[:, 0], U[:, 1], k, omega, nut_rans, nu=paths.NU, Ub=UB, H=H)
    gradU = _cell_gradient(mesh, "U")
    feat = pd.concat([feat, features.strain_rotation_features(gradU, Ub=UB, H=H)],
                     axis=1)

    # Galilean-invariant, geometry-agnostic features (for the transfer study).
    gradp = _cell_gradient(mesh, "p")
    wall_d = _wall_distance(vtu_path, centres)
    inv = features.invariant_features(U, gradU, gradp, k, omega, nut_rans,
                                      wall_d, nu=paths.NU)
    feat = pd.concat([feat, inv.drop(columns=["nut_over_nu"])], axis=1)

    # Nonlocal / transport features (MVR-7).
    gradk = _cell_gradient(mesh, "k")
    nl = features.nonlocal_features(U, gradU, gradk, gradp, k, omega, nut_rans,
                                    nu=paths.NU)
    feat = pd.concat([feat, nl], axis=1)

    # --- target: beta_nut from Boussinesq-inferred DNS eddy viscosity ---
    gradUdns = _cell_gradient(mesh, "UDNS")
    dUdy_dns, dVdx_dns = gradUdns[:, 1], gradUdns[:, 3]
    # Mask where the mean shear is below ~1% of the bulk gradient scale Ub/H:
    # below that the Boussinesq inversion for nut is ill-posed (near-zero
    # denominator) and the inferred beta is meaningless.
    nut_dns = features.effective_eddy_viscosity(uv_dns, dUdy_dns, dVdx_dns,
                                                shear_floor=1e-2 * UB / H)
    beta = features.beta_nut(nut_dns, nut_rans, nut_floor=1e-4 * paths.NU)

    frame = feat.copy()
    frame["case"] = case_name
    # keep raw quantities for maps / diagnostics
    frame["x"] = x
    frame["y"] = y
    frame["nut_rans"] = nut_rans
    frame["nut_dns_eff"] = nut_dns
    frame["uv_dns"] = uv_dns
    frame["U_dns"] = Udns[:, 0]
    frame["U_rans"] = U[:, 0]
    frame["beta_nut"] = beta
    # log target (symmetric about 1); guard the clip floor at 0.
    frame["log_beta_nut"] = np.log(np.clip(beta, 1e-3, None))
    # "Well-posed" cells: the eddy-viscosity correction ansatz is only
    # meaningful where the Boussinesq inversion gives a positive, finite,
    # non-extreme eddy viscosity. Counter-gradient regions (nut_dns < 0) are a
    # known limitation of the ansatz itself, not something to fit with a spike;
    # they are flagged here and excluded from training (and reported).
    frame["wellposed"] = (
        np.isfinite(nut_dns) & (nut_dns > 0) & (beta > 0.05) & (beta < 8.0)
    )

    # --- Reynolds-stress anisotropy target (barycentric map) ------------
    # More expressive than beta_nut: corrects the *shape* of the turbulent
    # stress and is defined even in counter-gradient regions.
    from . import anisotropy
    if "turbulenceProperties:R" in mesh.cell_data:
        bx_r, by_r, _ = anisotropy.barycentric(
            np.asarray(mesh.cell_data["turbulenceProperties:R"]))
        bx_d, by_d, valid_d = anisotropy.barycentric(tau)
        frame["bary_x_rans"] = bx_r
        frame["bary_y_rans"] = by_r
        frame["bary_x_dns"] = bx_d
        frame["bary_y_dns"] = by_d
        frame["dbary_x"] = bx_d - bx_r     # anisotropy-shape discrepancy
        frame["dbary_y"] = by_d - by_r
        frame["bary_valid"] = valid_d
    return frame


# Original feature set (includes absolute position x/H, y/H).
POSITION_FEATURES = [
    "x_over_H", "y_over_H", "U_over_Ub", "V_over_Ub", "k_over_Ub2",
    "omegaH_over_Ub", "nut_over_nu", "tke_intensity", "S_mag", "Omega_mag",
]

# Galilean-invariant, geometry-agnostic set (no absolute position).
INVARIANT_FEATURES = [
    "q1_qcrit", "q2_tke_intensity", "q3_wall_Re", "q4_pgrad_stream",
    "q5_strain_ratio", "q7_nonortho", "nut_over_nu",
]

# Nonlocal / transport / non-equilibrium markers (MVR-7).
NONLOCAL_FEATURES = [
    "Pk_over_eps", "conv_over_prod", "curvature", "adverse_pgrad",
    "U_gradk_misalign",
]


def feature_columns(kind: str = "position") -> list[str]:
    """Model inputs (order-stable).

    kind: 'position' | 'invariant' | 'nonlocal' | 'invariant_nonlocal'.
    """
    if kind == "position":
        return list(POSITION_FEATURES)
    if kind == "invariant":
        return list(INVARIANT_FEATURES)
    if kind == "nonlocal":
        return list(NONLOCAL_FEATURES)
    if kind == "invariant_nonlocal":
        return list(INVARIANT_FEATURES) + list(NONLOCAL_FEATURES)
    raise ValueError(f"Unknown feature kind {kind!r}.")


def load_dataset() -> pd.DataFrame:
    """Load the processed dataset (parquet if available, else csv)."""
    pq = paths.DATA_PROCESSED / "pehill_features_targets.parquet"
    csv = pq.with_suffix(".csv")
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        "Processed dataset not found. Run: python analysis/03_build_dataset.py")


def build_all(cases=paths.DNS_CASE_NAMES, run_root: Path | None = None,
              save: bool = True) -> pd.DataFrame:
    """Build and concatenate the dataset over all available geometries."""
    run_root = run_root or ML_RUNS
    frames = []
    for c in cases:
        cdir = run_root / c
        if not (cdir / "VTK").exists():
            print(f"  [skip] {c}: no VTK (not run yet)")
            continue
        frames.append(build_case_frame(c, run_root))
        print(f"  [ok]   {c}: {len(frames[-1])} cells")
    if not frames:
        raise RuntimeError("No geometries available. Run scripts/run_ml_cases.sh.")
    df = pd.concat(frames, ignore_index=True)
    if save:
        paths.ensure_output_dirs()
        out = paths.DATA_PROCESSED / "pehill_features_targets.parquet"
        try:
            df.to_parquet(out)
        except Exception:                       # parquet engine optional
            out = out.with_suffix(".csv")
            df.to_csv(out, index=False)
        print(f"  saved -> {out.relative_to(paths.ROOT)}  ({len(df)} rows)")
    return df
