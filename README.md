# ML-Augmented RANS Modelling of Separated Flow Using DNS Reference Data

A CFD + machine-learning research project investigating whether a lightweight,
**interpretable** data-driven correction can reduce systematic RANS
model-form error for separated turbulent flow over periodic hills, with
validation against **unseen** reference geometries.

> This project does **not** claim to "solve turbulence" or replace turbulence
> models. It quantifies where a standard RANS closure (k-ω SST) fails against
> DNS on a canonical separated flow, and tests how far a simple, physically
> grounded correction can close that gap while remaining stable and
> generalisable.

![Summary of headline results](docs/figures/summary.png)

**Results at a glance** *(periodic hill, Re=5600, all validated against DNS):*
(A) an ML eddy-viscosity correction on an **unseen geometry** cuts the
reattachment error; (B) it **generalises** across hill slopes; (C) DNS
anisotropy fills the barycentric map while linear RANS collapses to a line —
*why* the closure fails; (D) a **self-consistent coupled** correction converges
toward DNS. Full write-up: [`docs/methodology.md`](docs/methodology.md) ·
one-command pipeline: [`scripts/run_all.sh`](scripts/run_all.sh) ·
outward-facing summary: [`docs/writeup.md`](docs/writeup.md).

---

## Research question

> Can a lightweight, interpretable machine-learning correction improve RANS
> predictions of separated turbulent flow while remaining physically
> meaningful, stable, and validated against unseen reference data?

## Why the periodic hill?

The periodic hill is the standard benchmark for turbulence-model assessment in
separated flow:

- **Public DNS reference data** across a family of geometries
  (Xiao et al. 2020, `Re_H = 5600`, hill slopes α = 0.5 … 1.5).
- A **smooth-body separation** that fixed-coefficient RANS models systematically
  mispredict — separation point, recirculation bubble, reattachment location,
  and the recovery downstream.
- Streamwise-periodic, statistically 2-D mean flow: cheap to simulate (a
  quasi-2D RANS mesh) yet physically rich.
- The parameterised-geometry family enables an honest **geometry-wise
  generalisation** test: train on some hill slopes, predict on unseen ones.

## Reference data

Xiao, Wu, Laizet & Duan (2020), *Flows over periodic hills of parameterized
geometries: a dataset for data-driven turbulence modeling from direct
simulations*, **Computers & Fluids 200, 104431**.
Public database: <https://github.com/xiaoh/para-database-for-PIML>
(the `pehill-5-cases-DNS` set is used here). `Re_H = U_b H / ν = 5600`.

The [closure-challenge benchmark](https://github.com/rmcconke/closure-challenge-benchmark)
is noted only for its *generalisation-split philosophy*, not as a data source.

---

## Workflow

```
 OpenFOAM (simpleFoam, k-ω SST)          DNS database (Xiao et al. 2020)
   quasi-2D periodic hill, Re=5600           mean U,V,p + <u'v'>
              │                                      │
       foamToVTK export                     ASCII .dat profiles
              │                                      │
              └──────────────┬───────────────────────┘
                             ▼
              Interpolate onto a common (x/H, y/H) frame
                             ▼
        Quantify RANS model-form error  (MVR-1, current milestone)
        · velocity-profile RMSE   · error maps
        · separation / reattachment / bubble length
                             ▼
        Build local, non-dimensional RANS features
                             ▼
        Interpretable correction target:  β_nut = ν_t,DNS / ν_t,RANS
        (Boussinesq-inferred effective eddy viscosity)
                             ▼
        Train interpretable models (ridge → RF → gradient boosting)
        a-priori study — NOT coupled into the solver (yet)
                             ▼
        GEOMETRY-WISE validation on unseen hill slopes
```

## Status

| Milestone | Item | State |
|---|---|---|
| MVR-1 | Baseline k-ω SST case (`simpleFoam`, quasi-2D, `Re=5600`) | ✅ built, converges |
| MVR-1 | DNS structure inspection + sanity checks | ✅ `01_inspect_dns.py` |
| MVR-1 | RANS↔DNS failure map (profiles, error map, reattachment) | ✅ `02_compare_rans_dns.py` |
| MVR-2 | k-ω SST on **all 5 hill geometries** (co-located with DNS) | ✅ `scripts/run_ml_cases.sh` |
| MVR-2 | Feature/target dataset (β_nut from Boussinesq) | ✅ `03_build_dataset.py` |
| MVR-3 | Interpretable correction + leave-one-geometry-out validation | ✅ `04_train_correction.py` |
| MVR-3+ | Invariant (geometry-agnostic) feature study | ✅ `04_train_correction.py` |
| MVR-3+ | Robustness: k-ε closure, wall shear, uncertainty | ✅ `05_robustness.py` |
| MVR-3+ | Mesh-independence + statistical rigor (perm. importance, CI) | ✅ `07_…`, `08_…` |
| MVR-4 | A-posteriori (frozen-nut) correction, single-shot | ✅ `06_aposteriori.py` |
| MVR-4+ | Self-consistent (iterated) frozen coupling | ✅ `10_selfconsistent.py` |
| MVR-5 | **Coupled** custom solver (`simpleFoamBeta`, turbulence co-adapts) | ✅ `12_coupled.py`, `src/solvers/` |
| MVR-5+ | **Coupled anisotropy-stress** solver (`simpleFoamAniso`) | ✅ `15_aniso_coupled.py` |
| MVR-6 | Reynolds-stress **anisotropy** discrepancy target | ✅ `09_anisotropy.py` |
| MVR-6+ | **TBNN** (tensor-basis NN) + local-closure diagnostic | ✅ `13_tbnn.py` |
| MVR-7 | **Nonlocal / transport** features (g₁ recovery test) | ✅ `14_nonlocal.py` |
| MVR-8 | **Cross-flow** generalisation (train PHLL, test other flows) | ✅ `16_crossflow.py` |
| MVR-9 | **Multi-flow** training — does diversity rescue transfer? | ✅ `17_multiflow.py` |

### Key results (real data, `Re_H = 5600`)

Full account in [`docs/methodology.md`](docs/methodology.md).

- **MVR-1 (baseline failure).** k-ω SST predicts separation well (x/H ≈ 0.41 =
  DNS) but **over-predicts reattachment (x/H ≈ 7.5 vs DNS 4.5)** — bubble ~65%
  too long; profile RMSE peaks (~0.10 U_b) at x/H = 5–6. (DNS 4.5 matches the
  published Re=5600 value.) The recirculation region is visibly longer in RANS:

  ![Mean flow: RANS vs DNS recirculation bubble](docs/figures/flowfield_rans_vs_dns.png)
- **MVR-2 (target).** β_nut = ν_t,DNS/ν_t,RANS has **median ≈ 1.3–1.6, rising with
  steepness** → k-ω SST under-mixes. ~20% of cells are counter-gradient
  (ν_t,DNS < 0), where the eddy-viscosity ansatz breaks down — flagged and
  excluded, not fit.
- **MVR-3 (generalisation).** Random forest predicts log β_nut on an **unseen
  hill slope** with mean out-of-geometry **R² ≈ 0.82** (0.89–0.96 interpolating,
  0.63–0.66 extrapolating to extremes). Ridge fails (0.11): strongly nonlinear.
  **Invariant features** (R² ≈ 0.78 overall) are more physically defensible and
  *improve the steepest-hill extrapolation* while doing worse on the mildest — a
  mixed, honest result.
- **MVR-4 (a-posteriori).** Propagating the ML correction (unseen geometry) moves
  reattachment **7.46 → 3.85 (DNS 4.50) — ~78% error reduction**, with corrected
  profiles tracking DNS through the recirculation. Propagating the *exact* DNS β
  **overcorrects** (3.48): the a-priori-optimal eddy viscosity is not
  a-posteriori-optimal — a real limitation of the frozen-Boussinesq ansatz.

  ![Flow field: baseline → corrected → DNS](docs/figures/flowfield_correction.png)
- **Robustness.** k-ω SST (7.46) and k-ε (3.51) bracket DNS (4.50) from opposite
  sides — the error is closure-specific. Baseline reattachment is
  **mesh-independent** (7.48–7.50 over 8k→72k cells, GCI ~0.1%), so the error is
  physics, not discretisation. The generalisation R² beats a trivial
  constant-median baseline decisively, but its 5-geometry CI is honestly wide
  (90% CI ≈ [0.62, 0.92]).
- **Anisotropy target (MVR-6).** The Reynolds-stress **anisotropy** discrepancy
  is defined in **100% of cells** (β_nut only 80%) and generalises *better*
  (out-of-geometry **R² ≈ 0.93 / 0.99**). A linear model collapses onto one line
  of the barycentric map while DNS fills it — the visual signature of the
  closure's structural error.

- **Coupled solver (MVR-5, Track A).** A compiled custom solver `simpleFoamBeta`
  applies the correction with turbulence **co-adapting** each iteration. Velocity-
  profile RMSE vs DNS drops **0.106 → 0.037 (−65%)**, but reattachment moves only
  7.46 → 7.18: the SST model **self-regulates** (its nut falls to 0.76×), so
  *frozen studies overstate the reattachment benefit*. This coupled/frozen
  inconsistency is reproduced here from first principles.
- **TBNN (MVR-6+, Track B).** A tensor-basis neural network (invariance-embedded)
  generalises **worse** than an unconstrained RF on the anisotropy. The
  diagnostic explains why: the leading eddy-viscosity coefficient g₁ is
  **unpredictable from the local invariants (R²≈0)** — the local-closure
  assumption breaks down for separated flow. An honest negative result that
  pinpoints *which* assumption fails.

Making β self-consistent (evaluated on the corrected flow) settles reattachment
at ~5.5 (vs single-shot 3.85) — the single-shot over-corrects.
- **Nonlocal features (MVR-7).** Adding non-equilibrium/transport features
  roughly *halves* the unpredictability of g₁ (−0.78 → −0.44) and modestly
  improves every target — necessary but not sufficient; fuller convective-history
  modelling is the frontier (matching the 2025 "non-local effect" literature).
- **Coupled anisotropy stress (MVR-5+).** A second solver injects the DNS
  anisotropy discrepancy. **Even the exact DNS discrepancy does not recover the
  DNS mean flow** — reproducing the known **ill-conditioning** of RANS to
  Reynolds-stress corrections (Wu, Xiao et al. 2019).

- **Cross-flow generalisation (MVR-8).** Trained on periodic hills only and
  tested on *different* separated flows (McConkey dataset), anisotropy R² falls
  from **0.96 in-distribution to ≤ 0** on a backward-facing step / bump /
  converging-diverging channel — genuinely different geometries:

  ![The separated flows used for the cross-flow test](docs/figures/flowfield_other_flows.png)
- **Multi-flow rescue (MVR-9).** Training on *three* flows and testing on the
  unseen fourth **substantially recovers** transfer — the backward-facing step
  goes from R² **−2.33 → +0.48**, and every flow improves. The single-flow
  failure is a **data-coverage** problem, not a fundamental one — diverse
  training data is a concrete lever that works.

The full generalisation picture: interpolate geometry (~0.9) → extrapolate
geometry (~0.6–0.7) → single-flow transfer (≤ 0, fails) → **multi-flow transfer
(~0.2–0.5, largely recovered)**.

![Why linear RANS fails: Reynolds-stress anisotropy](docs/figures/anisotropy_barycentric.png)

![Generalisation across flow type — the challenge and the lever that works](docs/figures/generalization_story.png)

### The honest through-line

Data-driven closure for separated flow is hard at *every* stage, and this
project demonstrates each failure mode concretely — **predicting** the correction
(g₁ is not a local function of the invariants), **coupling** it (the turbulence
model self-regulates), **propagating** it (RANS is ill-conditioned to stress
corrections), and **generalising** it (single-flow training fails across flow
type) — *and* shows the constructive way forward: **training diversity
substantially rescues generalisation**. Mapping all of this honestly, with
working code for each, is the core contribution. Full account:
[`docs/methodology.md`](docs/methodology.md); one-command pipeline:
`scripts/run_all.sh`.

> **Reynolds-number generalisation is not testable here:** the Xiao database
> (both 5- and 29-case sets) is entirely at Re=5600. It is investigated and
> documented as a data limitation, not faked.

---

## Repository layout

```
cases/periodicHill_kOmegaSST/   Baseline steady RANS case (k-ω SST), quasi-2D
scripts/                        Shell helpers: fetch DNS, run/clean/export case
src/mlrans/                     Importable Python package (the reusable core)
  paths.py       project paths + physical constants (H, Ub, Re, ν)
  dns.py         load/inspect DNS reference data
  foam.py        load OpenFOAM fields from foamToVTK output (pyvista)
  grid.py        hill geometry, common-frame interpolation, profile extraction
  metrics.py     error metrics, separation/reattachment estimation
  features.py    RANS-local features + β_nut correction target
  correction.py  interpretable ML models + geometry-wise cross-validation
  plots.py       velocity-profile and error-map figures
analysis/                       Runnable entry-point scripts (01_, 02_, …)
data/external/                  Downloaded DNS data (git-ignored)
data/processed/                 Derived datasets (git-ignored)
results/figures, results/tables Generated artefacts (git-ignored)
notebooks/                      Exploration only — not the core workflow
docs/                           Milestones and methodology notes
```

## Reproducibility — quick start

Requires OpenFOAM v2312 (WSL; here via the `openfoam2312` wrapper) and Python ≥ 3.10.

> On a fresh WSL/Ubuntu the `venv` module may be missing (`ensurepip` error).
> Install it once: `sudo apt install -y python3-venv python3-pip`.
> Note: creating a venv **on the `/mnt/c` Windows drive can fail**; if so, create
> it on the Linux filesystem, e.g. `python3 -m venv ~/.venvs/mlrans`, then
> `source ~/.venvs/mlrans/bin/activate`.

```bash
# 1. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # or: pip install -e .

# 2. Reference DNS data (targeted download, not the full 1.3 GB repo)
scripts/fetch_dns_data.sh all          # or a single case, e.g. case_1p0
python analysis/01_inspect_dns.py      # inspect structure + sanity checks

# 3. Baseline RANS solve + export
scripts/run_baseline_sst.sh            # blockMesh → topoSet → simpleFoam
scripts/export_latest_fields.sh        # foamToVTK of the converged solution

# 4. Baseline failure map (RANS vs DNS)  [MVR-1]
python analysis/02_compare_rans_dns.py --dns-case case_1p0
#   -> results/figures/velocity_profiles_case_1p0.png, U_error_map_case_1p0.png
#   -> results/tables/bubble_metrics_case_1p0.csv

# 5. Multi-geometry RANS on the provided DNS meshes (all 5 hill slopes) [MVR-2]
#    (requires the sparse clone of pehill-5-cases-OpenFOAM into data/external/_ofrepo;
#     see scripts/run_ml_cases.sh)
scripts/run_ml_cases.sh

# 6. Build the ML feature/target dataset (beta_nut)  [MVR-2]
python analysis/03_build_dataset.py
#   -> data/processed/pehill_features_targets.(parquet|csv)

# 7. Train + leave-one-geometry-out validation (position vs invariant) [MVR-3]
python analysis/04_train_correction.py
#   -> results/figures/{feature_set_comparison,correction_feature_importance,
#                       correction_pred_vs_true,beta_true_case_1p0,beta_pred_case_1p0}.png
#   -> results/tables/correction_cv_metrics.csv

# 8. Robustness: k-epsilon closure, wall shear, prediction uncertainty  [MVR-3+]
#    (first build a k-epsilon run on case_1p0; see docs/methodology.md)
python analysis/05_robustness.py

# 9. A-posteriori: propagate the correction (frozen nut), unseen geometry [MVR-4]
python analysis/06_aposteriori.py

# 10. Verification / rigor / advanced targets
python analysis/07_mesh_independence.py     # grid convergence of the baseline
python analysis/08_stats_rigor.py           # perm. importance, CI, trivial baseline
python analysis/09_anisotropy.py            # Reynolds-stress anisotropy target [MVR-6]
python analysis/10_selfconsistent.py        # self-consistent frozen coupling [MVR-4+]
python analysis/11_summary_figure.py        # -> results/figures/summary.png

# Coupled correction with the custom solver [MVR-5, Track A]
(cd src/solvers/simpleFoamBeta && openfoam2312 wmake)   # compile once
python analysis/12_coupled.py               # turbulence co-adapts to betaNut

# Tensor-basis neural network + local-closure diagnostic [MVR-6+, Track B]
python analysis/13_tbnn.py                   # needs PyTorch (CPU is fine)

# Unit tests for the pure-computation core (no OpenFOAM/pyvista needed)
python -m pytest tests/ -q
```

### Key modelling choices (and why)

- **`Re_H = 5600`** (ν = 5×10⁻⁶): set to *match the DNS database*, not the
  OpenFOAM tutorial's `Re ≈ 10595`. RANS and reference must be the same flow.
- **Quasi-2D mesh** (1 cell in span): the mean flow is spanwise-homogeneous, so
  a single spanwise cell is standard and ~80× cheaper than the LES mesh the
  case is derived from.
- **Second-order momentum** (`linearUpwind`): first-order upwind is diffusive
  and would smear the very separation we are trying to measure.
- **Conventional SIMPLE + under-relaxation**: SIMPLEC with high relaxation
  settled into a limit cycle (the hill wake is weakly unsteady); standard
  under-relaxation converges to a steady state.
- **Interpretable target `β_nut`**: an eddy-viscosity multiplier is physically
  readable (β ≈ 1 → RANS locally fine; β far from 1 → model-form error) rather
  than an opaque field correction.
- **Geometry-wise validation**: leave-one-hill-slope-out, never random
  point-wise splits — a single flow field's points are spatially correlated and
  random splits would overstate generalisation.

---

## Honest limitations

- k-ω SST is one closure; conclusions about "RANS error" are specific to it
  (k-ε may be added later as a second baseline).
- The correction is studied **a-priori** (applied to DNS-derived quantities),
  not yet coupled inside the solver, so it does not prove closed-loop stability.
- The DNS `<u'v'>` column is used with confidence; the individual normal-stress
  columns of the reference files are treated as *provisional* pending
  cross-check against the source paper (see `src/mlrans/dns.py`).
- Boussinesq-inferred effective eddy viscosity is ill-posed in low-shear
  regions; those points are masked/down-weighted, not trusted.
- Generalisation is tested across hill *geometry* at fixed `Re`; Reynolds-number
  extrapolation is out of current scope.

## License / attribution

Reference DNS data © the authors of the para-database-for-PIML repository; cite
Xiao et al. (2020) if you use it. This repository contains analysis code and an
OpenFOAM case derived from the OpenFOAM v2312 `periodicHill` tutorial.
