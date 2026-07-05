#!/usr/bin/env bash
#
# Build and run a standard k-epsilon baseline on the case_1p0 provided mesh, as
# a SECOND turbulence closure for the robustness comparison (analysis/05).
# Reuses the k-omega SST template's numerics/mesh, swapping the model and the
# omega field for epsilon.
#
# Usage:  scripts/run_kepsilon_case1p0.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/data/external/_ofrepo/pehill-5-cases-OpenFOAM/case_1p0"
TPL="${ROOT}/cases/templates/kOmegaSST_pehill"
dst="${ROOT}/cases/ml_runs/kEpsilon_case_1p0"

if [[ ! -d "${SRC}" ]]; then
    echo "ERROR: provided mesh not found. Run: scripts/fetch_dns_data.sh openfoam" >&2
    exit 1
fi

rm -rf "${dst}"; mkdir -p "${dst}/constant" "${dst}/0.orig"
cp -r "${TPL}/system" "${dst}/system"
cp "${TPL}/constant/transportProperties" "${dst}/constant/"
cp -r "${SRC}/constant/polyMesh" "${dst}/constant/polyMesh"
cp "${TPL}/0.orig/U" "${TPL}/0.orig/p" "${TPL}/0.orig/k" "${dst}/0.orig/"

cat > "${dst}/constant/turbulenceProperties" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }
simulationType RAS;
RAS { RASModel kEpsilon; printCoeffs no; turbulence yes; }
EOF

cat > "${dst}/0.orig/epsilon" <<'EOF'
FoamFile { version 2.0; format ascii; class volScalarField; object epsilon; }
dimensions [0 2 -3 0 0 0 0];
internalField uniform 3e-9;   // Cmu^0.75 k^1.5 / L, k~1.5e-6, L~0.1
boundaryField
{
    "(inlet|outlet)" { type cyclic; }
    defaultFaces     { type empty; }
    "(bottomWall|topWall)" { type epsilonWallFunction; value $internalField; }
}
EOF

cat > "${dst}/0.orig/nut" <<'EOF'
FoamFile { version 2.0; format ascii; class volScalarField; object nut; }
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{
    "(inlet|outlet)" { type cyclic; }
    defaultFaces     { type empty; }
    "(bottomWall|topWall)" { type nutkWallFunction; value uniform 0; }
}
EOF

# Swap omega -> epsilon in the linear-solver and scheme dictionaries.
sed -i 's/(U|k|omega)/(U|k|epsilon)/; s/"(k|omega)"/"(k|epsilon)"/' "${dst}/system/fvSolution"
sed -i 's/div(phi,omega)/div(phi,epsilon)/' "${dst}/system/fvSchemes"

RUN=(bash -c); command -v openfoam2312 >/dev/null 2>&1 && RUN=(openfoam2312 bash -c)
"${RUN[@]}" "cd '${dst}' && rm -rf 0 && cp -r 0.orig 0 && \
    simpleFoam > log.simpleFoam 2>&1 && foamToVTK -latestTime -ascii > log.foamToVTK 2>&1"
echo "k-epsilon: $(grep -E '^Time = ' "${dst}/log.simpleFoam" | tail -1)"
