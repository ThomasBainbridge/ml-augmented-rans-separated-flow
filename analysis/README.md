# Analysis scripts

Runnable entry points (run from the project root). They import the reusable
core from `src/mlrans/` and write artefacts to `results/`. They operate on
**real** data only and refuse to fabricate output when inputs are missing.

| Script | Purpose | Requires |
|---|---|---|
| `01_inspect_dns.py` | Report DNS file structure + physical sanity checks | DNS ascii download |
| `02_compare_rans_dns.py` | Baseline RANS-vs-DNS failure map (profiles, error map, reattachment) | DNS + exported baseline RANS |
| `03_build_dataset.py` | Build the β_nut feature/target table from the 5 co-located RANS+DNS runs | `scripts/run_ml_cases.sh` |
| `04_train_correction.py` | Train ridge/RF/GBM + leave-one-geometry-out validation; position vs invariant features | dataset from `03` |
| `05_robustness.py` | k-ε closure comparison, wall-shear cross-check, RF prediction-uncertainty map | `03`, `06`, a k-ε run |
| `06_aposteriori.py` | Frozen-nut propagation of the correction on a held-out geometry (MVR-4) | `03`, the 5 runs |
| `07_mesh_independence.py` | Grid-convergence of the baseline reattachment (GCI) | OpenFOAM |
| `08_stats_rigor.py` | Permutation importance, constant-median baseline, 5-geometry bootstrap CI | `03` |
| `09_anisotropy.py` | Reynolds-stress anisotropy discrepancy target + barycentric map (MVR-6) | `03` |
| `10_selfconsistent.py` | Iterated (self-consistent) frozen-nut coupling (MVR-4+) | `03`, the 5 runs |
| `11_summary_figure.py` | Four-panel summary from the saved tables | `04,05,09,10` |
| `12_coupled.py` | Coupled correction via the custom `simpleFoamBeta` solver (turbulence co-adapts) | compiled solver, `03` |
| `13_tbnn.py` | Tensor-basis neural network + local-closure (g₁ predictability) diagnostic | PyTorch, the 5 runs |
| `14_nonlocal.py` | Nonlocal/transport features: g₁-recovery + target-generalisation test (MVR-7) | `03` |
| `15_aniso_coupled.py` | Coupled anisotropy-stress correction via `simpleFoamAniso` (+ ill-conditioning ceiling) | compiled solver, `03` |
| `16_crossflow.py` | Cross-flow generalisation: train on periodic hills, test on unseen separated flows (MVR-8) | McConkey dataset |
| `17_multiflow.py` | Multi-flow leave-one-flow-out: does training diversity rescue transfer? (MVR-9) | McConkey dataset |
| `18_figures.py` | Clean figures from the saved tables → `docs/figures/` | tables from the above |
| `19_flowviz.py` | Flow-field visualisations: RANS-vs-DNS + corrected recirculation bubble, the other cross-flow geometries, the 5 hill shapes, and the correction field map | runs + McConkey data |

Full ordered pipeline: **`scripts/run_all.sh`**.

```bash
python analysis/01_inspect_dns.py
python analysis/02_compare_rans_dns.py --dns-case case_1p0
python analysis/03_build_dataset.py
python analysis/04_train_correction.py
python analysis/06_aposteriori.py
python analysis/05_robustness.py
```

MVR-4 here is a **frozen-nut propagation** (see `RESULTS.md`), not a
re-trained coupled closure — it demonstrates the effect, not closed-loop
robustness in general.
