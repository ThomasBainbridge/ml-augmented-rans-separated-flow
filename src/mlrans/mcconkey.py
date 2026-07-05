"""Loader for the McConkey, Yee & Lien (2021) curated turbulence dataset.

Scientific Data 8, 255 (2021); Kaggle: ryleymcconkey/ml-turbulence-dataset.

The dataset provides, per flow case, k-omega SST RANS fields AND co-located
DNS/LES labels (the anisotropy tensor b, Reynolds stresses, mean velocity), on
the same meshes used elsewhere in this project. It contains several separated
flows besides the periodic hill, which lets us run a CROSS-FLOW generalisation
test: train the correction on periodic hills, predict on a *different* separated
flow (curved backward-facing step, parametric bump, converging-diverging
channel). The periodic-hill cases are all at Re=5600 (the same geometry sweep as
the Xiao data), so this dataset does not add Reynolds-number variation for that
flow.

Files (git-ignored, extracted from the Kaggle archive):
    data/external/mcconkey-dataset/komegasst/komegasst_<CASE>_<field>.npy
    data/external/mcconkey-dataset/labels/<CASE>_<field>.npy
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import features, paths

MCK = paths.DATA_EXTERNAL / "mcconkey-dataset"

# Separated-flow families used for the cross-flow study (DUCT = secondary flow,
# a different regime, is excluded).
FAMILIES = ("PHLL", "CBFS", "BUMP", "CNDV")


def _rans(case: str, field: str) -> np.ndarray:
    return np.load(MCK / "komegasst" / f"komegasst_{case}_{field}.npy")


def _label(case: str, field: str) -> np.ndarray:
    return np.load(MCK / "labels" / f"{case}_{field}.npy")


def list_cases(family: str) -> list[str]:
    """Case names present on disk for a family (e.g. 'PHLL' -> case_1p0…)."""
    d = MCK / "labels"
    if not d.exists():
        raise FileNotFoundError(
            f"McConkey dataset not found at {MCK}. Extract the Kaggle archive "
            "into data/external/mcconkey-dataset/.")
    names = sorted({p.name[:-6] for p in d.glob(f"{family}_*_b.npy")})
    return names


def available() -> dict:
    return {fam: list_cases(fam) for fam in FAMILIES if list_cases(fam)}


def load_case(case: str) -> dict:
    """Assemble the RANS fields and DNS labels needed for features/targets."""
    U = np.column_stack([_rans(case, "Ux"), _rans(case, "Uy"), _rans(case, "Uz")])
    gradU = _rans(case, "gradU")                       # (N,3,3)
    gradk = np.column_stack([_rans(case, f"gradk{c}") for c in "xyz"])
    gradp = np.column_stack([_rans(case, f"gradp{c}") for c in "xyz"])
    k = _rans(case, "k")
    omega = _rans(case, "omega")
    nu = float(np.ravel(_rans(case, "nu"))[0])
    wall_d = _rans(case, "wallDistance")
    # eddy viscosity (k-omega): nu_t = k / omega (SST limiter neglected — fine
    # for a feature).
    nut = k / np.maximum(omega, 1e-12)
    b_dns = _label(case, "b")                          # (N,3,3) DNS anisotropy
    return dict(case=case, Cx=_rans(case, "Cx"), Cy=_rans(case, "Cy"),
                U=U, gradU=gradU, gradk=gradk, gradp=gradp, k=k, omega=omega,
                nu=nu, nut=nut, wall_d=wall_d, b_dns=b_dns)


def features_target(case: str):
    """(feature DataFrame, DNS anisotropy b (N,3,3), Cx, Cy) for one case.

    Features are the project's geometry-agnostic set: invariant + nonlocal.
    Because they carry no absolute position, they are directly comparable across
    different flow geometries — the whole point of the cross-flow test.
    """
    d = load_case(case)
    inv = features.invariant_features(d["U"], d["gradU"], d["gradp"], d["k"],
                                      d["omega"], d["nut"], d["wall_d"], nu=d["nu"])
    nl = features.nonlocal_features(d["U"], d["gradU"], d["gradk"], d["gradp"],
                                    d["k"], d["omega"], d["nut"], nu=d["nu"])
    feat = pd.concat([inv, nl], axis=1)
    return feat, d["b_dns"], d["Cx"], d["Cy"]


def assemble(feature_cols):
    """Concatenate all separated-flow cases into arrays for the CV experiments.

    Returns (X, Y, family, case, coords) where Y is the DNS anisotropy in
    barycentric-map coordinates. No clipping is applied here — callers clip on
    their own training distribution to avoid leakage.
    """
    from . import anisotropy
    Xs, Ys, fam, case_id, coords = [], [], [], [], []
    for family, cases in available().items():
        for c in cases:
            feat, b, cx, cy = features_target(c)
            x, y = anisotropy.barycentric_from_b(b)
            Xs.append(feat[feature_cols].to_numpy())
            Ys.append(np.column_stack([x, y]))
            fam += [family] * len(feat)
            case_id += [c] * len(feat)
            coords.append(np.column_stack([cx, cy]))
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    return X, Y, np.array(fam), np.array(case_id), np.concatenate(coords)
