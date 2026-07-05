#!/usr/bin/env bash
#
# Clean the baseline periodic-hill case back to its definition (0.orig,
# constant/{transport,turbulence}Properties, system, Allrun/Allclean).
# Uses OpenFOAM's own cleanCase0, so it ONLY touches the case directory and
# never deletes anything outside it.
#
# Usage:
#   scripts/clean_baseline_sst.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE="${ROOT}/cases/periodicHill_kOmegaSST"

if [[ ! -d "${CASE}" ]]; then
    echo "ERROR: baseline case not found at ${CASE}" >&2
    exit 1
fi

if command -v openfoam2312 >/dev/null 2>&1; then
    RUNNER=(openfoam2312 bash -c)
else
    RUNNER=(bash -c)
fi

echo "Cleaning case: ${CASE}"
"${RUNNER[@]}" "cd '${CASE}' && ./Allclean"
echo "Done. Case reset to its definition."
