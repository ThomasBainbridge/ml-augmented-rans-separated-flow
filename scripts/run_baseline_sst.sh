#!/usr/bin/env bash
#
# Run the baseline steady RANS (simpleFoam + k-omega SST) periodic-hill case.
# Sources OpenFOAM v2312 via the openfoam2312 wrapper (opencfd Debian package)
# and executes the case's own Allrun.
#
# Usage:
#   scripts/run_baseline_sst.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE="${ROOT}/cases/periodicHill_kOmegaSST"

if [[ ! -d "${CASE}" ]]; then
    echo "ERROR: baseline case not found at ${CASE}" >&2
    exit 1
fi

# Prefer the versioned wrapper if present, else assume the environment is
# already sourced (FOAM_TUTORIALS set).
if command -v openfoam2312 >/dev/null 2>&1; then
    RUNNER=(openfoam2312 bash -c)
elif [[ -n "${WM_PROJECT_DIR:-}" ]]; then
    RUNNER=(bash -c)
else
    echo "ERROR: OpenFOAM not found. Install openfoam2312 or source its bashrc." >&2
    exit 1
fi

echo "Running baseline case: ${CASE}"
"${RUNNER[@]}" "cd '${CASE}' && ./Allrun"

echo
echo "Run complete. Latest time directory:"
ls -d "${CASE}"/[0-9]* 2>/dev/null | sort -V | tail -1
echo "Next: scripts/export_latest_fields.sh"
