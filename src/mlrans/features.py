"""RANS-local features and the interpretable eddy-viscosity correction target.

Two responsibilities, both deliberately kept a-priori (no OpenFOAM coupling):

1. Build local, non-dimensional features from a baseline RANS solution. These
   are the model inputs. Only Galilean-invariant / locally-defined quantities
   are used so a trained correction can transfer between geometries.

2. Define the correction *target*: an eddy-viscosity multiplier

       beta_nut = nut_DNS_effective / nut_RANS

   where nut_DNS_effective is inferred from the DNS Reynolds shear stress via
   the Boussinesq relation. This target is interpretable (beta ~ 1 means RANS
   is locally fine; beta far from 1 flags model-form error) and is the first
   thing we try to learn — an a-priori study, NOT a coupled turbulence model.

The gradient-dependent features and the strain used for beta_nut require
velocity gradients. These are supplied externally (from grad(U) written by
OpenFOAM, or computed on a regular grid) so this module stays free of meshing
concerns; functions document exactly what they need.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A small floor to keep non-dimensional ratios and divisions well-behaved.
EPS = 1e-12


def nondimensional_features(x, y, U, V, k, omega, nut, nu,
                            Ub: float = 1.0, H: float = 1.0) -> pd.DataFrame:
    """Assemble the directly-available non-dimensional RANS features.

    These need only primary RANS fields (no gradients):
        x/H, y/H, U/Ub, V/Ub, k/Ub^2, omega*H/Ub, nut/nu, and the
        turbulence-intensity-like ratio sqrt(k)/(|U|+eps).
    Gradient-based features (strain/rotation rate) are added separately via
    :func:`strain_rotation_features` once grad(U) is available.
    """
    speed = np.sqrt(U**2 + V**2)
    return pd.DataFrame({
        "x_over_H": x / H,
        "y_over_H": y / H,
        "U_over_Ub": U / Ub,
        "V_over_Ub": V / Ub,
        "k_over_Ub2": k / Ub**2,
        "omegaH_over_Ub": omega * H / Ub,
        "nut_over_nu": nut / nu,
        "tke_intensity": np.sqrt(np.maximum(k, 0.0)) / (speed + EPS),
    })


def strain_rotation_features(gradU: np.ndarray, Ub: float = 1.0, H: float = 1.0):
    """Non-dimensional strain- and rotation-rate magnitudes from grad(U).

    Parameters
    ----------
    gradU : (N, 3, 3) or (N, 9) array of the velocity-gradient tensor
            d U_i / d x_j at each point (as written by OpenFOAM's grad(U)).

    Returns a DataFrame with columns S_mag (|S| H/Ub) and Omega_mag (|W| H/Ub),
    where S and W are the symmetric and antisymmetric parts of grad(U).
    """
    g = np.asarray(gradU)
    if g.ndim == 2 and g.shape[1] == 9:
        g = g.reshape(-1, 3, 3)
    if g.shape[1:] != (3, 3):
        raise ValueError(f"gradU must be (N,3,3) or (N,9); got {g.shape}.")
    S = 0.5 * (g + np.transpose(g, (0, 2, 1)))
    W = 0.5 * (g - np.transpose(g, (0, 2, 1)))
    S_mag = np.sqrt(np.einsum("nij,nij->n", S, S))
    W_mag = np.sqrt(np.einsum("nij,nij->n", W, W))
    scale = H / Ub
    return pd.DataFrame({"S_mag": S_mag * scale, "Omega_mag": W_mag * scale})


def invariant_features(U, gradU, gradp, k, omega, nut, wall_dist, nu,
                       Cmu: float = 0.09) -> pd.DataFrame:
    """Galilean-invariant, geometry-agnostic features (no x/H, y/H).

    Physics-based scalar invariants in the spirit of Wang, Wu & Xiao (2017) and
    Ling & Templeton. Each is either bounded or a dimensionless local ratio, so
    a correction learned from them should transfer between hill geometries far
    better than one that leans on absolute position.

    Inputs are per-cell arrays: U (N,3), gradU (N,9|N,3,3), gradp (N,3),
    scalars k, omega, nut, wall_dist (all N,), and molecular nu.
    """
    g = np.asarray(gradU)
    if g.ndim == 2 and g.shape[1] == 9:
        g = g.reshape(-1, 3, 3)
    S = 0.5 * (g + np.transpose(g, (0, 2, 1)))
    W = 0.5 * (g - np.transpose(g, (0, 2, 1)))
    S2 = np.einsum("nij,nij->n", S, S)
    W2 = np.einsum("nij,nij->n", W, W)
    Smag = np.sqrt(2.0 * S2)
    U = np.asarray(U)
    gradp = np.asarray(gradp)
    k = np.asarray(k)
    Umag = np.sqrt(np.einsum("ni,ni->n", U, U))
    eps = Cmu * np.maximum(k, 0.0) * np.maximum(np.asarray(omega), EPS)

    # q1: normalised excess rotation (Q-criterion), bounded (-1, 1)
    q1 = (W2 - S2) / (W2 + S2 + EPS)
    # q2: turbulence intensity, bounded (0, 1)
    q2 = k / (0.5 * Umag**2 + k + EPS)
    # q3: wall-distance Reynolds number (viscous vs log region), bounded (0, 2)
    q3 = np.minimum(np.sqrt(np.maximum(k, 0.0)) * np.asarray(wall_dist) / (50.0 * nu), 2.0)
    # q4: pressure gradient along a streamline, bounded (-1, 1)
    Udp = np.einsum("ni,ni->n", U, gradp)
    dpmag = np.sqrt(np.einsum("ni,ni->n", gradp, gradp))
    q4 = Udp / (Umag * dpmag + np.abs(Udp) + EPS)
    # q5: ratio of turbulent to mean-strain time scale, bounded (0, 1)
    ts = k * Smag
    q5 = ts / (ts + eps + EPS)
    # q7: non-orthogonality of velocity and its gradient, bounded (0, 1)
    UgU = np.abs(np.einsum("ni,nj,nij->n", U, U, g))
    norm7 = np.sqrt(np.einsum("nl,nl,ni,nik,nj,njk->n", U, U, U, g, U, g))
    q7 = UgU / (UgU + norm7 + EPS)

    return pd.DataFrame({
        "q1_qcrit": q1,
        "q2_tke_intensity": q2,
        "q3_wall_Re": q3,
        "q4_pgrad_stream": q4,
        "q5_strain_ratio": q5,
        "q7_nonortho": q7,
        "nut_over_nu": nut / nu,
    })


def nonlocal_features(U, gradU, gradk, gradp, k, omega, nut, nu,
                      Cmu: float = 0.09) -> pd.DataFrame:
    """Transport / non-equilibrium features (beyond the local S,R invariants).

    Motivated by the TBNN diagnostic (analysis/13): the anisotropy's dominant
    coefficient is NOT a function of the local strain/rotation invariants, so
    the flow carries non-local / history information those invariants miss.
    These features encode production/dissipation imbalance, TKE convection,
    streamline curvature, adverse pressure gradient and gradient misalignment —
    all markers of non-equilibrium, history-dependent turbulence.
    """
    g = np.asarray(gradU)
    if g.ndim == 2:
        g = g.reshape(-1, 3, 3)
    U = np.asarray(U)
    gradk = np.asarray(gradk)
    gradp = np.asarray(gradp)
    S = 0.5 * (g + np.transpose(g, (0, 2, 1)))
    S2 = np.einsum("nij,nij->n", S, S)
    Umag = np.sqrt(np.einsum("ni,ni->n", U, U)) + EPS
    eps = Cmu * np.maximum(k, 0.0) * np.maximum(np.asarray(omega), EPS) + EPS

    # production of TKE (Boussinesq): Pk = 2 nut S:S
    Pk = 2.0 * np.asarray(nut) * S2
    # convection of TKE: U.grad(k)
    Uk = np.einsum("ni,ni->n", U, gradk)
    # streamline curvature ~ |U . grad(U)| projected normal to U, /|U|^2
    UgU = np.einsum("ni,nij->nj", U, g)              # (U.grad)U
    curv = np.sqrt(np.einsum("nj,nj->n", UgU, UgU)) / (Umag**2)

    return pd.DataFrame({
        # non-equilibrium: production/dissipation (equilibrium => ~1)
        "Pk_over_eps": Pk / eps,
        # relative importance of TKE convection (history) vs production
        "conv_over_prod": np.abs(Uk) / (Pk + EPS),
        # streamline curvature (non-dim by 1/|U|)
        "curvature": np.tanh(curv),
        # adverse pressure gradient marker (flow decelerating): +ve = adverse
        "adverse_pgrad": np.tanh(np.einsum("ni,ni->n", U, gradp) / (Umag * eps**0.5 + EPS)),
        # misalignment of velocity and TKE gradient (bounded)
        "U_gradk_misalign": np.einsum("ni,ni->n", U, gradk)
        / (Umag * np.sqrt(np.einsum("ni,ni->n", gradk, gradk)) + EPS),
    })


def effective_eddy_viscosity(uv, dUdy, dVdx, shear_floor: float = 1e-4):
    """Infer nut from the DNS Reynolds shear stress via the Boussinesq relation.

    For the dominant shear component, Boussinesq gives
        <u'v'> = -nut * (dU/dy + dV/dx)
    so
        nut_eff = -<u'v'> / (dU/dy + dV/dx).

    Low-shear regions make this ill-posed (division by ~0). We mask points where
    the mean shear magnitude is below ``shear_floor`` and return NaN there, to
    be handled (clipped / down-weighted) by the caller rather than trusted.
    """
    shear = np.asarray(dUdy) + np.asarray(dVdx)
    nut_eff = np.full_like(shear, np.nan, dtype=float)
    ok = np.abs(shear) >= shear_floor
    nut_eff[ok] = -np.asarray(uv)[ok] / shear[ok]
    return nut_eff


def beta_nut(nut_dns_eff, nut_rans, nut_floor: float = 1e-6,
             clip: tuple[float, float] | None = (0.0, 10.0)):
    """Eddy-viscosity correction multiplier beta = nut_DNS_eff / nut_RANS.

    A ``nut_floor`` avoids blow-up where RANS nut ~ 0; the optional ``clip``
    keeps the (heavy-tailed) target in a sane range for regression. Later work
    may instead learn log(beta) for symmetry about 1.
    """
    denom = np.maximum(np.asarray(nut_rans), nut_floor)
    beta = np.asarray(nut_dns_eff) / denom
    if clip is not None:
        beta = np.clip(beta, *clip)
    return beta
