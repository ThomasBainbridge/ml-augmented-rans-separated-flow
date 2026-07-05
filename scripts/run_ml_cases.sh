#!/usr/bin/env bash
#
# Build and run the k-omega SST baseline on ALL 5 periodic-hill geometries,
# using the provided DNS meshes (which encode the alpha=0.5..1.5 hill shapes)
# so the RANS solution is co-located with the DNS reference on the same mesh.
#
# For each geometry it:
#   1. assembles a run case from the shared template + that geometry's polyMesh;
#   2. solves steady k-omega SST (simpleFoam);
#   3. copies the DNS fields (UDNS, TauDNS) into the converged time directory;
#   4. exports everything to VTK (RANS + DNS, co-located) for the ML pipeline.
#
# The mesh is NOT renumbered, so the copied DNS fields (written in the original
# cell order) stay aligned with the solved RANS fields.
#
# Usage:  scripts/run_ml_cases.sh [case_1p0 case_0p8 ...]   (default: all 5)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/data/external/_ofrepo/pehill-5-cases-OpenFOAM"
TPL="${ROOT}/cases/templates/kOmegaSST_pehill"
RUN="${ROOT}/cases/ml_runs"

if [[ ! -d "${SRC}" ]]; then
    echo "ERROR: provided OpenFOAM cases not found at ${SRC}" >&2
    echo "       Fetch them first (sparse clone) — see scripts/fetch_dns_data.sh notes." >&2
    exit 1
fi

if command -v openfoam2312 >/dev/null 2>&1; then
    FOAM=(openfoam2312 bash -c)
else
    FOAM=(bash -c)
fi

CASES=("$@")
[[ ${#CASES[@]} -eq 0 ]] && CASES=(case_0p5 case_0p8 case_1p0 case_1p2 case_1p5)

for c in "${CASES[@]}"; do
    echo "==================================================================="
    echo "  ${c}"
    echo "==================================================================="
    dst="${RUN}/${c}"
    rm -rf "${dst}"
    mkdir -p "${dst}/constant"
    cp -r "${TPL}/0.orig" "${dst}/0.orig"
    cp -r "${TPL}/system" "${dst}/system"
    cp "${TPL}/constant/transportProperties" "${TPL}/constant/turbulenceProperties" "${dst}/constant/"
    cp -r "${SRC}/${c}/constant/polyMesh" "${dst}/constant/polyMesh"

    "${FOAM[@]}" "cd '${dst}' && rm -rf 0 && cp -r 0.orig 0 && \
        simpleFoam > log.simpleFoam 2>&1"
    conv=$(grep -c 'SIMPLE solution converged' "${dst}/log.simpleFoam" || true)
    last=$(grep -E '^Time = ' "${dst}/log.simpleFoam" | tail -1)
    echo "  solve: ${last}  (converged flag: ${conv})"

    # Integer-only match so we never pick up 0.orig (the '[0-9]*' glob would).
    latest=$(find "${dst}" -maxdepth 1 -type d -regextype posix-extended \
                 -regex '.*/[0-9]+' | sort -V | tail -1)
    if [[ "$(basename "${latest}")" == "0" ]]; then
        echo "  WARNING: no converged time written for ${c}; skipping DNS/VTK." >&2
        continue
    fi
    cp "${SRC}/${c}/0/UDNS" "${SRC}/${c}/0/TauDNS" "${latest}/"
    echo "  DNS fields copied into $(basename "${latest}")"

    "${FOAM[@]}" "cd '${dst}' && foamToVTK -latestTime -ascii > log.foamToVTK 2>&1"
    vtu=$(find "${dst}/VTK" -name 'internal.vtu' | head -1)
    echo "  VTK: ${vtu#${ROOT}/}"
done

echo
echo "All requested geometries done. Build the ML dataset next:"
echo "  python analysis/03_build_dataset.py"
