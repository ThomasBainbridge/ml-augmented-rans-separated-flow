# Milestones

Incremental, each producing a defensible result before adding complexity.

## MVR-1 — Baseline RANS failure map  *(done)*

Quantify where and by how much baseline k-ω SST RANS departs from DNS on the
`α = 1.0` periodic hill at `Re = 5600`.

- [x] Baseline k-ω SST case that builds, runs, and converges to steady state
- [x] Targeted, scripted DNS download + structure inspection with sanity checks
- [x] Common-frame RANS↔DNS interpolation
- [x] Velocity-profile overlay, U-error map, separation/reattachment table

**Result:** separation captured (x/H ≈ 0.41 = DNS); reattachment badly
over-predicted (x/H ≈ 7.5 vs DNS 4.5); profile RMSE peaks in the x/H = 5–6
reattachment zone.

## MVR-2 — Feature/target dataset  *(done)*

- [x] k-ω SST on all 5 provided DNS meshes (co-located RANS + DNS in one VTK)
- [x] `grad(U)` / strain / rotation features via pyvista; Reynolds stress `R`
- [x] Boussinesq-inferred `ν_t,DNS` and `β_nut`, with low-shear masking
- [x] Counter-gradient (ν_t,DNS < 0) cells flagged `wellposed=False`
- [x] Tidy feature/target table in `data/processed/`

**Result:** median β_nut ≈ 1.3–1.6, rising with hill steepness (k-ω SST
under-mixes); ~20% counter-gradient cells excluded and reported.

## MVR-3 — Interpretable a-priori correction  *(done)*

- [x] Ridge → random forest → gradient boosting on log β_nut
- [x] Leave-one-geometry-out validation (train on 4 slopes, test on the 5th)
- [x] Feature-importance + out-of-fold predicted-vs-true + β_nut maps
- [x] Honest reporting: interpolation vs extrapolation gap

**Result:** RF mean out-of-geometry R² ≈ 0.82 (0.89–0.96 interpolating,
0.63–0.66 extrapolating to extreme slopes); ridge R² ≈ 0.11 (nonlinear).

## MVR-3+ — Invariant features & robustness  *(done)*

- [x] Galilean-invariant, geometry-agnostic feature set (Q-criterion, TI,
      wall-Re, pressure-gradient-along-streamline, strain-timescale, non-ortho)
- [x] Re-run geometry-wise CV: invariant vs position comparison
- [x] Second closure (k-ε) baseline; wall-shear cross-check; RF uncertainty map

**Result:** invariant features (R² ≈ 0.78) improve the steepest-hill
extrapolation but do worse on the mildest — mixed but physically defensible.
k-ω SST (7.46) and k-ε (3.51) bracket DNS (4.50).

## MVR-4 — A-posteriori (frozen-nut) correction  *(done)*

- [x] Inject β·ν_t via frozen k-scaling; re-solve momentum on a held-out geometry
- [x] Propagate both the exact DNS β and the ML-predicted β
- [x] Compare reattachment/profiles vs baseline and DNS

**Result:** ML correction (unseen geometry) moves reattachment 7.46 → 3.85
(DNS 4.50), ~78% error reduction; exact-β overcorrects (3.48), exposing the
a-priori/a-posteriori gap.

## MVR-5 — Coupled correction via a custom solver  *(done, Track A)*

- [x] Compiled `simpleFoamBeta` (fork of simpleFoam) applying nuEff = nu + betaNut*nut
- [x] Turbulence co-adapts each iteration (genuine coupling, not frozen)
- [x] Held-out case_1p0: profile RMSE 0.106 -> 0.037 (-65%)

**Result:** the coupled mean flow is much closer to DNS, but reattachment barely
moves (7.46 -> 7.18) because the SST model self-regulates (nut -> 0.76x) — frozen
studies overstate the reattachment benefit.

## MVR-6+ — Tensor-Basis Neural Network  *(done, Track B)*

- [x] TBNN (Ling et al. 2016) for the anisotropy, invariance embedded
- [x] Leave-one-geometry-out vs an unconstrained RF on identical inputs
- [x] Diagnostic: predictability of each basis coefficient from the invariants

**Result (honest negative):** TBNN generalises worse than the RF; the leading
eddy-viscosity coefficient g1 is unpredictable from the local invariants
(R^2 ~ 0), so the local-closure assumption fails for separated flow.

## MVR-7 — Nonlocal features + coupled anisotropy stress  *(done)*

- [x] Nonlocal/transport features (`14_nonlocal.py`): g1 −0.78 → −0.44,
      targets improve modestly — necessary but not sufficient
- [x] `simpleFoamAniso`: coupled anisotropy-stress correction (`15_aniso_coupled.py`)

**Result:** even the exact DNS anisotropy discrepancy, coupled, does not recover
the DNS mean flow (reattachment 7.46 → 6.62, profile RMSE not improved) —
reproducing the ill-conditioning of RANS to Reynolds-stress corrections
(Wu, Xiao et al. 2019).

## MVR-8 — Cross-flow generalisation  *(done)*

- [x] Load the McConkey et al. (2021) dataset (periodic hills + curved
      backward-facing step, parametric bumps, converging-diverging channel)
- [x] Train the anisotropy correction on periodic hills; test on unseen flows
- [x] Report the generalisation hierarchy honestly (`16_crossflow.py`)

**Result:** anisotropy R² falls from 0.96 (in-distribution) to ≤ 0 on different
separated flows. Interpolating geometry works, extrapolating degrades,
transferring to a new flow type fails — the central limitation of single-flow
data-driven closures.

## MVR-9 — Multi-flow training  *(done)*

- [x] Leave-one-flow-out across periodic hills + step + bump + channel
- [x] Compare single-flow (MVR-8) vs multi-flow training (`17_multiflow.py`)

**Result:** training diversity substantially rescues cross-flow generalisation
(backward-facing step R² −2.33 → +0.48; every flow improves). The single-flow
failure is a data-coverage problem, not a fundamental one.

## MVR-10 (future) — Toward a general closure

- [ ] Broader multi-flow training (more flow types) + held-out-flow benchmarking
- [ ] Convective-history / vector-cloud nonlocal features
- [ ] Reynolds-number generalisation via external multi-Re DNS (ERCOFTAC Case 81)
- [ ] Conditioning-aware propagation of Reynolds-stress corrections

**Note on Reynolds-number generalisation (investigated).** Not testable with
the available data: both the 5-case and 29-case Xiao databases are entirely at
Re_H = 5600 (the 29-case set varies hill shape, height and streamwise extent,
not Re). A Re-transfer study needs an external multi-Re DNS source with the full
Reynolds-stress fields. The canonical one is **Breuer et al. (2009)**, periodic
hill at Re = 100–10595 — archived as **ERCOFTAC Database Classic Collection
Case 81** (and partly on the NASA TMR). Access is via ERCOFTAC registration /
direct request, not a programmatic download, and it would additionally require
building matched RANS meshes at each Re. So this is scoped as future work with a
concrete data path, rather than faked. (Recent literature — e.g. the 2025
"non-local effect" separated-flow closures — pursues exactly this axis.)

## Planned figures

1. RANS vs DNS velocity profiles
2. RANS error map (model-form error)
3. β_nut target map
4. β_nut predicted map
5. Corrected vs DNS (if defensible)
6. Reattachment-location comparison
7. Wall-shear comparison (data permitting)
8. Feature importance / interpretability
9. Generalisation on unseen geometry
10. Summary figure
