#!/usr/bin/env bash
#
# End-to-end pipeline for the ML-augmented RANS periodic-hill study.
# Each step is idempotent-ish and can be run on its own; this documents the
# canonical order. Assumes: OpenFOAM v2312 (openfoam2312), Python deps
# installed (requirements.txt; PyTorch optional for step 13).
#
# Usage:  scripts/run_all.sh            # run the whole pipeline
#         scripts/run_all.sh data       # just fetch data
#         scripts/run_all.sh analysis   # just the Python analysis (needs runs)
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAGE="${1:-all}"

banner() { printf '\n============================================================\n  %s\n============================================================\n' "$1"; }

if [[ "$STAGE" == "all" || "$STAGE" == "data" ]]; then
    banner "1. Reference data"
    scripts/fetch_dns_data.sh all           # DNS ascii profiles (all 5 slopes)
    scripts/fetch_dns_data.sh openfoam      # provided meshes + co-located DNS
    python analysis/01_inspect_dns.py
fi

if [[ "$STAGE" == "all" || "$STAGE" == "cfd" ]]; then
    banner "2. Baseline + multi-geometry RANS"
    scripts/run_baseline_sst.sh             # standalone MVR-1 baseline
    scripts/export_latest_fields.sh
    scripts/run_ml_cases.sh                 # k-omega SST on all 5 geometries
    scripts/run_kepsilon_case1p0.sh         # second closure
    banner "3. Compile custom solvers"
    (cd src/solvers/simpleFoamBeta  && openfoam2312 wmake)
    (cd src/solvers/simpleFoamAniso && openfoam2312 wmake)
fi

if [[ "$STAGE" == "all" || "$STAGE" == "analysis" ]]; then
    banner "4. Analysis pipeline"
    python analysis/02_compare_rans_dns.py --dns-case case_1p0   # MVR-1 failure map
    python analysis/03_build_dataset.py                          # MVR-2 features/targets
    python analysis/04_train_correction.py                       # MVR-3 beta_nut + features
    python analysis/05_robustness.py                             # k-eps, Cf, uncertainty
    python analysis/06_aposteriori.py                            # frozen single-shot
    python analysis/07_mesh_independence.py                      # grid convergence
    python analysis/08_stats_rigor.py                            # stats honesty
    python analysis/09_anisotropy.py                             # MVR-6 anisotropy target
    python analysis/10_selfconsistent.py                         # self-consistent frozen
    python analysis/12_coupled.py                                # MVR-5 coupled betaNut
    python analysis/13_tbnn.py                                   # Track B TBNN (needs torch)
    python analysis/14_nonlocal.py                               # MVR-7 nonlocal features
    python analysis/15_aniso_coupled.py                          # coupled anisotropy stress
    python analysis/16_crossflow.py                              # MVR-8 cross-flow (needs McConkey data)
    python analysis/17_multiflow.py                              # MVR-9 multi-flow (needs McConkey data)
    python analysis/19_flowviz.py                              # flow-field visualisation (RANS vs DNS)
    python analysis/18_figures.py                      # figures -> docs/figures/
fi

banner "Done. Figures in results/figures/, tables in results/tables/."
