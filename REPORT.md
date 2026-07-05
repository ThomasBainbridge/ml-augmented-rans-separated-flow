# ML-Augmented RANS Modelling of Separated Flow

**A two-page summary.** Every figure, table and reproduction command is in
[RESULTS.md](RESULTS.md); the code and a one-command pipeline are in
[README.md](README.md).

---

## Problem

Industrial CFD almost always uses **RANS** turbulence models: cheap, but
systematically wrong when the flow separates (the wake behind a hill, a step, a
wing). High-fidelity **DNS** is the "right answer" but far too expensive for
routine use. This project asks a data-driven-modelling question on the standard
separated-flow benchmark — the periodic hill at `Re = 5600`, with public DNS
across five hill geometries:

> Can a lightweight, **interpretable** machine-learning correction reduce the
> model-form error of RANS (k-ω SST) for separated flow, remain physically
> meaningful and stable, and **generalise to geometries and flows it never saw**?

**Pipeline:** baseline k-ω SST (OpenFOAM) → co-located DNS reference → an
interpretable eddy-viscosity / Reynolds-stress-anisotropy correction target →
geometry-agnostic features → interpretable models (ridge → random forest →
gradient boosting; a tensor-basis neural network) → a-posteriori injection
through two **custom-compiled OpenFOAM solvers** → honest generalisation tests
across geometry and flow type.

---

## What was found

**It works, on unseen geometry.** Baseline k-ω SST over-predicts reattachment
(x/H ≈ 7.5 vs DNS 4.5 — verified mesh-independent). An interpretable
eddy-viscosity correction generalises to an unseen hill shape (random-forest
R² ≈ 0.82) and, propagated through a coupled custom solver, **cuts the
velocity-profile error by 65%**.

**Then the honest limits — data-driven closure is hard at every stage, and each
failure mode is demonstrated with working code:**

- **Predicting it.** A tensor-basis neural network *underperforms* an
  unconstrained regressor; the leading eddy-viscosity coefficient is
  **unpredictable from the local invariants** — the local-closure assumption
  fails for separated flow. Nonlocal features help but are not sufficient.
- **Coupling it.** When the turbulence model co-adapts, it **self-regulates**
  (ν_t drops to 0.76×), partly cancelling a fixed ν_t-multiplier — so frozen
  studies overstate the benefit.
- **Propagating it.** Even the **exact** DNS Reynolds-stress anisotropy,
  injected through a second custom solver, does not recover the DNS mean flow —
  reproducing the known ill-conditioning of RANS to stress corrections.
- **Generalising it.** A correction trained on one flow **fails on different
  separated flows** (R² 0.96 → ≤ 0) — but training on **several** flows
  **substantially rescues it** (a backward-facing step recovers to R² 0.48).
  The failure is a *data-coverage* problem, not a fundamental one.

The generalisation hierarchy that emerges: interpolate geometry (works) →
extrapolate geometry (degrades) → single-flow transfer (fails) → multi-flow
transfer (largely recovered).

---

## Contribution

A clean, reproducible pipeline — an importable Python package, unit tests, CI,
and **two custom OpenFOAM solvers** — plus a clear-eyed map of *where* data-driven
closure helps, *where and why* it does not, and the concrete lever (training
diversity) that improves it. Nothing is overclaimed; the negative results are as
carefully established as the positive one.

## Natural next directions (what this motivates)

1. **Nonlocal / convective-history closures** — the g₁ diagnostic shows local
   invariants are provably insufficient for the anisotropy in separated flow.
2. **Reynolds-number generalisation** — needs external multi-Re DNS (Breuer /
   ERCOFTAC Case 81); the Xiao databases are all Re=5600.
3. **A fully re-trained coupled closure** with conditioning-aware propagation of
   the Reynolds-stress correction.
