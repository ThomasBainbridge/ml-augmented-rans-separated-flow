#!/usr/bin/env bash
#
# Export the latest solved time of the baseline case to VTK for the Python
# pipeline (pyvista reads the foamToVTK output directly).
#
# Produces:
#   cases/periodicHill_kOmegaSST/VTK/    -> internal + boundary meshes with
#                                           U, p, k, omega, nut, R, wallShear...
#
# Usage:
#   scripts/export_latest_fields.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE="${ROOT}/cases/periodicHill_kOmegaSST"

if [[ ! -d "${CASE}" ]]; then
    echo "ERROR: baseline case not found at ${CASE}" >&2
    exit 1
fi

LATEST="$(ls -d "${CASE}"/[0-9]* 2>/dev/null | sort -V | tail -1 || true)"
if [[ -z "${LATEST}" ]]; then
    echo "ERROR: no solved time directory found in ${CASE}." >&2
    echo "       Run scripts/run_baseline_sst.sh first." >&2
    exit 1
fi
echo "Latest time: $(basename "${LATEST}")"

if command -v openfoam2312 >/dev/null 2>&1; then
    RUNNER=(openfoam2312 bash -c)
else
    RUNNER=(bash -c)
fi

# writeCellCentres: adds C (cell-centre coordinates) so downstream code can
# work in physical space without re-reading the mesh geometry separately.
# foamToVTK -latestTime: export only the converged solution, ASCII for
# transparency, including the 'hills' boundary patch for wall-shear analysis.
"${RUNNER[@]}" "cd '${CASE}' && \
    postProcess -func writeCellCentres -latestTime && \
    foamToVTK -latestTime -ascii"

echo
echo "Exported to: ${CASE}/VTK"
echo "Next: python analysis/02_compare_rans_dns.py"
