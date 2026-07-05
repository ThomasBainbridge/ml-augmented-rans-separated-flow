# Outward-facing summaries

Two ready-to-use write-ups of this project: a short post for LinkedIn / a
project page, and a one-page technical summary for a prospective PhD supervisor.
Both are framed around the project's honest through-line rather than hype.

---

## 1. Short summary post (LinkedIn / project page)

**Can machine learning fix the errors that fast CFD makes on separated flow?
I built the full pipeline to find out — and the honest answer is "partly, and
here's exactly why."**

Industrial CFD almost always uses RANS turbulence models: fast, but
systematically wrong when flow separates (think the wake behind a car or over a
wing). High-fidelity DNS is the "right answer" but far too expensive for routine
use. So I asked: can a lightweight, *interpretable* ML correction close the gap —
and does it generalise to shapes it never trained on?

Using the periodic-hill benchmark (Re = 5600, five hill geometries, public DNS):

🔹 Baseline k-ω SST over-predicts the recirculation bubble by ~65% (reattachment
x/H ≈ 7.5 vs DNS 4.5) — and I verified this is real physics, not mesh error
(grid-converged, GCI ~0.1%).

🔹 An interpretable eddy-viscosity correction, trained on four hill shapes and
tested on the **unseen** fifth, cuts the velocity-profile error by **65%** when
propagated through a **custom coupled OpenFOAM solver I wrote and compiled**.

🔹 The most valuable results are the honest limits. Data-driven closure is hard
at *every* stage, and I demonstrated each with working code:
   • **Predicting** the correction — the key term isn't a local function of the
     usual invariants (a tensor-basis neural network actually *underperforms* a
     simple model here, and I show why);
   • **Coupling** it — the turbulence model self-regulates and partly cancels the
     correction;
   • **Propagating** it — even the *exact* DNS Reynolds stresses don't recover the
     DNS mean flow, reproducing a known ill-conditioning of the RANS equations.

No overclaiming, no faked results — every number is from converged CFD vs public
DNS, and the whole pipeline runs from one script.

Tools: OpenFOAM (incl. two custom C++ solvers), Python (NumPy/scikit-learn/
PyVista), PyTorch (tensor-basis NN). Code + write-up on GitHub.

#CFD #MachineLearning #TurbulenceModelling #OpenFOAM #ComputationalScience

*(Optional lead image: `docs/figures/summary.png`.)*

---

## 2. Supervisor-facing technical summary (one page)

**Title.** ML-Augmented RANS Modelling of Separated Flow Using DNS Reference Data.

**Motivation.** Linear-eddy-viscosity RANS closures (here k-ω SST) carry large,
systematic model-form error in separated flow. Data-driven closure is an active
field; this project is a self-contained, reproducible study of *where* such a
correction helps and *why* it fails where it does — on the parameterised
periodic-hill DNS database (Xiao et al. 2020, Re = 5600, five hill slopes).

**Method.** Steady quasi-2D k-ω SST (OpenFOAM v2312) on all five geometries,
co-located with DNS mean velocity and Reynolds stresses. An interpretable
eddy-viscosity multiplier β_nut = ν_t,DNS/ν_t,RANS is the primary target;
geometry-wise (leave-one-hill-out) cross-validation is used throughout to avoid
spatial leakage. The study then extends to a Reynolds-stress anisotropy target,
a tensor-basis neural network (TBNN), nonlocal/transport features, and two
purpose-built coupled solvers.

**Key results (all vs DNS, mesh-independent baseline).**
- Baseline reattachment x/H ≈ 7.5 vs DNS 4.5 (validated grid-converged).
- β_nut generalises to an unseen geometry with random-forest R² ≈ 0.82
  (interpolation 0.9–0.96; extrapolation to the extreme slopes 0.6–0.7); the
  target rises with hill steepness (SST systematically under-mixes).
- The Reynolds-stress **anisotropy** discrepancy is defined everywhere (β_nut only
  ~80% of cells) and generalises better (R² ≈ 0.93/0.99).
- **A-posteriori, coupled** (custom solver `simpleFoamBeta`, turbulence
  co-adapting): velocity-profile RMSE −65%, but reattachment barely moves — the
  SST model self-regulates (ν_t → 0.76×), so frozen studies overstate the benefit.
- **TBNN** underperforms an unconstrained regressor; the diagnostic shows the
  leading (eddy-viscosity) basis coefficient is unpredictable from the local
  invariants (R² ≈ 0) — the local-closure assumption fails for separated flow.
- **Nonlocal features** halve that gap (−0.78 → −0.44) but are not sufficient.
- **Coupled anisotropy correction** (`simpleFoamAniso`): even the *exact* DNS
  discrepancy does not recover the DNS mean flow — reproducing the ill-
  conditioning of RANS to Reynolds-stress corrections (Wu, Xiao et al. 2019).

**Contribution.** A clean, reproducible pipeline (two custom OpenFOAM solvers,
an importable Python package, unit tests, one-command run) and an honest map of
the three failure modes of data-driven closure — prediction, coupling,
propagation — for separated flow.

**Natural PhD directions this raises.** (i) Nonlocal / convective-history
closures (the local invariants are provably insufficient here); (ii)
conditioning-aware propagation of Reynolds-stress corrections; (iii)
Reynolds-number generalisation with external multi-Re DNS (ERCOFTAC Case 81).

**Reproducibility.** `scripts/run_all.sh`; methodology in `docs/methodology.md`;
exact environment in `requirements-lock.txt`.
