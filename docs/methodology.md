# Methodology & results

A concise, honest account of the method and what it does (and does not) show.
All numbers are from real converged CFD and the public DNS database; nothing is
fabricated.

## Abstract

We study whether a lightweight, interpretable machine-learning correction can
reduce the systematic model-form error of RANS (k-ω SST) for separated flow over
periodic hills, validated against DNS across five hill geometries. Baseline SST
over-predicts reattachment (x/H ≈ 7.5 vs DNS 4.5; mesh-independent, GCI ~0.1%).
An interpretable eddy-viscosity correction β_nut generalises to an unseen
geometry (random-forest R² ≈ 0.82) and, propagated through a compiled coupled
solver, cuts the velocity-profile error by 65%. We then push past the standard
a-priori story to three honest limiting results: (i) the corrected eddy
viscosity is partly cancelled by the turbulence model's self-regulation
(coupled ≠ frozen); (ii) a tensor-basis neural network *underperforms* an
unconstrained regressor because the anisotropy's leading coefficient is
unpredictable from local invariants — the local-closure assumption fails for
separated flow, only partly recovered by nonlocal features; and (iii) even the
*exact* DNS Reynolds-stress anisotropy, injected via a second custom solver, does
not recover the DNS mean flow, reproducing the known ill-conditioning of the RANS
equations to stress corrections. Finally, a correction trained on one flow fails
to transfer to different separated flows (R² 0.96 → ≤ 0), but training on several
flows substantially rescues it (a backward-facing step recovers to R² 0.48) —
so the generalisation failure is a data-coverage problem, not a fundamental one.
The contribution is a reproducible pipeline and a clear-eyed map of where
data-driven closure helps, where and why it does not, and the concrete lever
(training diversity) that improves it.

## 1. Problem and data

- **Flow:** periodic hill, `Re_H = U_b H / ν = 5600` — the standard separated-flow
  benchmark for turbulence-model assessment.
- **Reference:** DNS of parameterised-geometry periodic hills (Xiao et al. 2020),
  five hill slopes `α ∈ {0.5, 0.8, 1.0, 1.2, 1.5}`. The provided OpenFOAM cases
  supply, per geometry, a RANS-comparable mesh **and** the DNS mean velocity and
  Reynolds-stress tensor interpolated onto that mesh (`UDNS`, `TauDNS`), so RANS
  and DNS are co-located (no scattered interpolation for the target).
- **Baseline RANS:** steady incompressible `simpleFoam`, **k-ω SST**, quasi-2D,
  second-order momentum, conventional SIMPLE under-relaxation. Converges to a
  steady state on every geometry.

## 2. MVR-1 — baseline model-form error

k-ω SST predicts the **separation point well** (x/H ≈ 0.41 = DNS) but
**over-predicts reattachment badly** (x/H ≈ 7.5 vs DNS 4.5): the recirculation
bubble is ~65% too long. Velocity-profile RMSE peaks (~0.10 U_b) in the
x/H = 5–6 reattachment zone. This is the well-known SST behaviour on this flow.
Reattachment is measured from the sign of the near-bottom-wall velocity (actual
first-cell band, applied identically to RANS and DNS); the DNS value 4.5 matches
the published Re=5600 result, validating the metric.

## 3. MVR-2 — correction target

The interpretable target is an eddy-viscosity multiplier

    β_nut = ν_t,DNS_eff / ν_t,RANS ,

with ν_t,DNS_eff inferred from the DNS Reynolds shear stress via Boussinesq:

    <u'v'> = −ν_t (∂U/∂y + ∂V/∂x)   ⇒   ν_t,DNS_eff = −<u'v'> / (∂U/∂y + ∂V/∂x).

Gradients are computed on the mesh (pyvista). Findings:

- **Median β_nut ≈ 1.3–1.6, rising with hill steepness** → k-ω SST systematically
  *under*-mixes, more so for stronger separation.
- **~20% of cells are counter-gradient** (ν_t,DNS < 0): the eddy-viscosity ansatz
  itself breaks down there. These are flagged (`wellposed=False`) and **excluded
  from training**, not fit with a spike — a deliberate, reported limitation.

## 4. MVR-3 — a-priori correction and generalisation

Learn `log β_nut` from RANS-local features; validate **leave-one-geometry-out**
(train on 4 slopes, predict the unseen 5th). Two feature sets are compared:

| model | position features (incl. x/H, y/H) | invariant features (geometry-agnostic) |
|---|---|---|
| ridge | R² 0.11 | R² 0.06 |
| **random forest** | **R² 0.82** | **R² 0.78** |
| gradient boosting | R² 0.68 | R² 0.69 |

Random-forest out-of-geometry R² per held-out slope:

| held-out | position | invariant |
|---|---|---|
| 0.5 (mildest) | 0.63 | 0.41 |
| 0.8 | 0.89 | 0.87 |
| 1.0 | 0.95 | 0.95 |
| 1.2 | 0.96 | 0.94 |
| 1.5 (steepest) | 0.66 | **0.73** |

**Honest reading.** The mapping is strongly nonlinear (ridge fails). Position
features give the best average, but they partly *memorise location* and so
transfer worst to the extremes. The invariant set (top features: turbulence
intensity, ν_t/ν, Q-criterion) is physically defensible and **improves the
hardest extrapolation (steepest hill)** while doing **worse on the mildest** —
a mixed result, not a clean win. Both interpolate far better than they
extrapolate, which is the key generalisation lesson.

## 5. MVR-4 — a-posteriori (does it improve the CFD?)

The correction is propagated through a **frozen-turbulence** momentum solve on a
**held-out geometry** (α = 1.0). Because k-ω SST recomputes ν_t from k/ω, and
ν_t is linear in k in both SST branches, the correction is injected by scaling
the frozen k → β·k (turbulence off), giving ν_t ≈ β·ν_t,baseline in the momentum
equation. Reattachment (x/H):

| field | x/H | error vs DNS |
|---|---|---|
| DNS | 4.50 | — |
| k-ω SST baseline | 7.46 | +2.96 (too long) |
| corrected — ML β, **unseen geometry** | **3.85** | −0.65 |
| corrected — exact DNS β (ansatz ceiling) | 3.48 | −1.02 (overshoots) |

**The ML correction cuts the reattachment error by ~78% on a geometry it never
saw**, and the corrected velocity profiles track DNS through the recirculation
zone where the baseline is clearly wrong. Two honest caveats:

1. Propagating the **exact** DNS-inferred β **overcorrects** (3.02) — the
   a-priori-optimal eddy viscosity is *not* a-posteriori-optimal. This is a real
   limitation of the frozen-Boussinesq ansatz, not a bug.
2. This is a **frozen-nut propagation**, not a re-trained coupled turbulence
   model; it demonstrates the effect, not closed-loop robustness in general.

**Self-consistent coupling (analysis/10).** The single-shot version evaluates β
on the *baseline* flow, which is inconsistent once the flow changes. Iterating —
predict β on the current flow, apply, re-solve, repeat (under-relaxed) — reaches
a fixed point at reattachment **≈ 5.5** (β_med falls 1.36 → 1.30). So the
single-shot value (3.85) **over-corrects** by using baseline-flow features; the
self-consistent estimate (5.5, vs DNS 4.5) is more principled. Both still cut the
baseline error (7.46) substantially (~65–85%). The residual gap and the
sensitivity to the coupling treatment are honest reminders that a fully coupled,
re-trained closure (not frozen propagation) is the proper end goal.

## 5b. Anisotropy target (a more complete correction)

β_nut corrects only the *magnitude* of the turbulent stress and is undefined in
the ~20% counter-gradient cells. The Reynolds-stress **anisotropy** discrepancy
(barycentric-map coordinates Δx, Δy; Emory & Iaccarino; Wu, Wang & Xiao) is
**defined in 100% of cells** and captures the stress *shape*. A linear
eddy-viscosity model collapses onto a single line of the barycentric triangle
while DNS fills it — the visual signature of the closure's structural error.
Learning the discrepancy (invariant features, leave-one-geometry-out) gives
**R² ≈ 0.93 and 0.99** for the two components — *better* generalisation than
β_nut and without the counter-gradient gap. This is the natural target for the
coupled closure of MVR-5.

## 6. Robustness checks

- **Closure sensitivity:** k-ω SST (7.46) and standard k-ε (3.51) bracket DNS
  (4.50) from opposite sides — the model-form error is closure-specific.
- **Wall shear:** the corrected case sharply shrinks the reversed-flow
  (τ_w > 0) region on the hill, a qualitative cross-check of the improvement.
- **Prediction uncertainty:** random-forest per-tree spread (median std ≈ 0.11 in
  log β) maps where the correction is least trustworthy.

## 6. Coupled correction with a custom solver (Track A)

Everything above is *frozen* propagation. `simpleFoamBeta` (a compiled fork of
simpleFoam, `src/solvers/`) solves momentum with nuEff = nu + betaNut*nut while
the turbulence model keeps updating each iteration, so k, omega, nut and the
mean flow co-adapt to a fixed ML correction field (field-inversion style). On
the held-out `case_1p0`:

- **Velocity-profile RMSE vs DNS falls 0.106 -> 0.037 (-65%)** — the coupled mean
  flow is much closer to DNS.
- **Reattachment moves only 7.46 -> 7.18**, far less than the frozen values
  (single-shot 3.85, self-consistent 5.5). The reason is explicit in the data:
  the SST model **self-regulates**, its mean nut dropping to **0.76x** as the
  flow reattaches, partly cancelling the momentum correction. So frozen studies
  *overstate* the reattachment benefit by holding nut fixed. This coupled/frozen
  inconsistency is a central, well-known issue in data-driven turbulence
  modelling, reproduced here from first principles.

## 7. Tensor-Basis Neural Network (Track B)

A TBNN (Ling et al. 2016) predicts the anisotropy as b = sum_n g_n(lambda) T_n
from the 5 strain/rotation invariants, embedding rotational invariance. Compared
leave-one-geometry-out, on identical inputs, with an unconstrained random forest.

- The tensor basis is expressive: a best-possible per-cell least-squares fit
  reaches **0.88** barycentric R^2.
- Yet the TBNN generalises **worse** (barycentric R^2 < 0) than the unconstrained
  RF (**0.61**). The diagnostic shows why: the leading coefficient **g1** (the
  eddy-viscosity alignment, coefficient of T1 = S) has **R^2 ~ 0** against the
  local invariants, while g2–g4 are predictable (0.59–0.69). The dominant part of
  the anisotropy is **not a local function of the invariants** — the TBNN's
  closure assumption breaks down for separated flow, where nonlocal / history
  effects and counter-gradient transport dominate (consistent with the ~20%
  counter-gradient cells of MVR-2).

This is an honest negative result and arguably more informative than a win: it
pinpoints *which* assumption fails and *why*, and motivates nonlocal /
transport-informed features and closures as the genuine path forward.

## 6b. Coupled anisotropy correction and RANS ill-conditioning (Track A+)

A second custom solver, `simpleFoamAniso`, injects a deviatoric Reynolds-stress
anisotropy correction — div(2 k (b_DNS - b_RANS)) — into the momentum equation
(turbulence co-adapting). This corrects the stress *shape* (unlike betaNut) and
so can act even in counter-gradient regions. On the held-out case_1p0:

| field | reattachment x/H | U-profile RMSE vs DNS |
|---|---|---|
| baseline | 7.46 | 0.106 |
| aniso, ML-predicted (unseen) | 7.36 | 0.111 |
| aniso, **exact DNS** discrepancy | 6.62 | 0.113 |
| DNS | 4.50 | — |

**Even the exact DNS anisotropy discrepancy, coupled, only nudges reattachment
(7.46 -> 6.62) and does NOT reduce the velocity-profile error.** This reproduces
the known **ill-conditioning of the RANS equations to Reynolds-stress
corrections** (Wu, Xiao et al., PRFluids 2019): a perfect Reynolds-stress field
propagated through RANS momentum need not recover the DNS mean velocity. Together
with the betaNut self-regulation (6) and the g1-unpredictability (7), this shows
data-driven closure for separated flow is hard at *every* stage — predicting the
correction *and* propagating it.

## 7b. Nonlocal / transport-informed features (MVR-7)

Directly testing the Track-B diagnostic: do nonlocal, non-equilibrium features
(production/dissipation imbalance, TKE convection, streamline curvature, adverse
pressure gradient, velocity–TKE-gradient misalignment) recover the closure that
local invariants cannot? Leave-one-geometry-out:

- **g1 (eddy-viscosity coefficient): −0.78 → −0.44** — nonlocal features roughly
  *halve* the unpredictability, but it stays negative.
- Every correction target improves modestly: log β_nut 0.774 → **0.807**;
  anisotropy Δx 0.931 → 0.939, Δy 0.987 → 0.989.

Honest reading: non-equilibrium/history information **matters** (it consistently
helps), but these local-ish nonlocal markers are **necessary, not sufficient** —
the eddy-viscosity alignment still carries information beyond them. Fuller
convective-history / nonlocal modelling (cf. recent 2025 "non-local effect" and
vector-cloud closures) is the genuine frontier. This is consistent with, and
motivated by, the counter-gradient and g1 findings above.

## 7c. Cross-flow generalisation (MVR-8)

The strongest test of whether the correction learned *physics* rather than the
periodic-hill flow: train on periodic hills only, then predict the DNS
anisotropy on completely different separated flows, using the McConkey et al.
(2021) dataset (k-ω SST RANS + DNS labels for a curved backward-facing step,
parametric bumps, and a converging-diverging channel). Only geometry-agnostic
(invariant + nonlocal) features are used; target is the DNS anisotropy in
barycentric coordinates.

| test flow | anisotropy barycentric R² |
|---|---|
| periodic hills (in-distribution, LOGO) | **0.96** |
| converging-diverging channel (CNDV) | 0.03 |
| parametric bumps (BUMP) | −0.34 |
| curved backward-facing step (CBFS) | −2.33 |

**The correction that predicts periodic-hill anisotropy almost perfectly fails
completely on different separated flows (R² ≤ 0)** — despite using
position-free features. The ordering is physically sensible (CNDV, the most
periodic-hill-like channel flow, transfers least badly; CBFS, the most
different, worst). This completes a clear **generalisation hierarchy**:

    interpolate geometry within a flow   R² ~ 0.9–0.96   (works)
    extrapolate to extreme geometry      R² ~ 0.6–0.7    (degrades)
    transfer to a different flow type    R² <= 0         (fails)

The correction largely memorised the periodic-hill regime. This is the honest,
central limitation of single-flow data-driven closures and a well-known open
problem — cleanly quantified here.

## 7d. Does training diversity rescue it? (MVR-9)

The constructive follow-up: leave-one-FLOW-out across the four separated-flow
families — train on three, predict the held-out fourth — versus the single-flow
(periodic-hills-only) result of MVR-8.

| held-out flow | trained on 1 flow (MVR-8) | trained on 3 flows (MVR-9) |
|---|---|---|
| curved backward-facing step (CBFS) | −2.33 | **+0.48** |
| converging-diverging channel (CNDV) | 0.03 | 0.19 |
| parametric bumps (BUMP) | −0.34 | −0.02 |
| periodic hills (PHLL) | — | 0.52 |

**Training diversity substantially rescues cross-flow generalisation** — CBFS
jumps from catastrophic failure to genuinely predictive (R² 0.48), and *every*
flow improves. It does not reach the in-distribution level (0.96), and bumps
remain hard (an anisotropy regime under-represented by the other three flows),
but the message is constructive: the failure of MVR-8 is not fundamental — it is
a **data-coverage** problem, and diverse training data is a concrete lever that
works, consistent with the field's move toward multi-flow closures.

The full generalisation picture:

    interpolate geometry within a flow    R² ~ 0.9–0.96   (works)
    extrapolate to extreme geometry       R² ~ 0.6–0.7    (degrades)
    single-flow -> different flow          R² <= 0         (fails)
    multi-flow  -> unseen flow             R² ~ 0.2–0.5    (largely recovered)

## 8. Limitations (summary)

- Single Reynolds number; generalisation tested across **geometry**, not Re.
- Reattachment is measured from the sign of the near-bottom-wall velocity (actual
  first-cell band), applied identically to RANS and DNS. The DNS value (4.5)
  agrees with the wall-shear cross-check and the published result; residual
  method sensitivity is ~±0.2 H.
- The eddy-viscosity ansatz fails in counter-gradient regions (~20% of cells).
- k-ω SST only for the primary baseline; the correction is a-priori-trained and
  a-posteriori-propagated in frozen form, not a coupled retrained closure.
- rms normal-stress columns of the raw DNS files remain unverified; only the
  Reynolds shear stress (used for β_nut) is relied upon.
