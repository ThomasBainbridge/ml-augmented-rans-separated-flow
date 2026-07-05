#!/usr/bin/env bash
#
# Fetch the periodic-hill DNS reference data (Xiao et al. 2020, Re_H = 5600)
# from the public para-database-for-PIML repository.
#
# We download ONLY the ASCII DNS profile files for the 5 parameterized-geometry
# cases (alpha = 0.5, 0.8, 1.0, 1.2, 1.5) rather than git-cloning the whole
# ~1.3 GB repository. This is lighter, resumable, and reproducible.
#
# Source:
#   H. Xiao, J.-L. Wu, S. Laizet, L. Duan, "Flows over periodic hills of
#   parameterized geometries", Computers & Fluids 200 (2020) 104431.
#   https://github.com/xiaoh/para-database-for-PIML
#
# Usage:
#   scripts/fetch_dns_data.sh            # fetch the default (case_1p0) only
#   scripts/fetch_dns_data.sh all        # fetch all 5 DNS ascii cases
#   scripts/fetch_dns_data.sh case_0p8 case_1p2
#   scripts/fetch_dns_data.sh openfoam   # sparse-clone the 5 OpenFOAM meshes
#                                        # (+ co-located DNS fields) for MVR-2/3
#
set -euo pipefail

REPO_URL="https://github.com/xiaoh/para-database-for-PIML"
REPO_RAW="https://raw.githubusercontent.com/xiaoh/para-database-for-PIML/master"
SUBPATH="pehill-5-cases-DNS"
# Files present in each case's dns-data/ directory (verified via GitHub API).
FILES=(mean_files.dat rms_files.dat rms_files1.dat rms_files2.dat)

# Resolve project root (this script lives in <root>/scripts/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ROOT="${ROOT}/data/external/pehill-5-cases-DNS"

ALL_CASES=(case_0p5 case_0p8 case_1p0 case_1p2 case_1p5)

# Mode: sparse-clone the OpenFOAM cases (meshes + co-located DNS fields).
# This uses a blob-filtered sparse checkout so we only pull that subdirectory
# (~57 MB) instead of the whole ~1.3 GB repository.
if [[ "${1:-}" == "openfoam" ]]; then
    OFDIR="${ROOT}/data/external/_ofrepo"
    if [[ -d "${OFDIR}/pehill-5-cases-OpenFOAM" ]]; then
        echo "OpenFOAM cases already present at ${OFDIR}"
        exit 0
    fi
    echo "Sparse-cloning pehill-5-cases-OpenFOAM into ${OFDIR} ..."
    rm -rf "${OFDIR}"
    git clone --no-checkout --filter=blob:none --sparse "${REPO_URL}" "${OFDIR}"
    ( cd "${OFDIR}" \
        && git sparse-checkout set pehill-5-cases-OpenFOAM \
        && git checkout master )
    echo "Done. Now run:  scripts/run_ml_cases.sh"
    exit 0
fi

# Decide which DNS ascii cases to fetch.
if [[ $# -eq 0 ]]; then
    CASES=(case_1p0)
elif [[ "${1}" == "all" ]]; then
    CASES=("${ALL_CASES[@]}")
else
    CASES=("$@")
fi

echo "Project root : ${ROOT}"
echo "Destination  : ${DEST_ROOT}"
echo "Cases        : ${CASES[*]}"
echo

for CASE in "${CASES[@]}"; do
    OUT_DIR="${DEST_ROOT}/${CASE}/dns-data"
    mkdir -p "${OUT_DIR}"
    for F in "${FILES[@]}"; do
        URL="${REPO_RAW}/${SUBPATH}/${CASE}/dns-data/${F}"
        OUT="${OUT_DIR}/${F}"
        if [[ -s "${OUT}" ]]; then
            echo "  [skip] ${CASE}/${F} (already present)"
            continue
        fi
        echo "  [get ] ${CASE}/${F}"
        # -f: fail on HTTP error; -L: follow redirects; -C -: resume.
        curl -fL -C - -o "${OUT}" "${URL}" || {
            echo "  [WARN] failed to download ${URL}" >&2
            rm -f "${OUT}"
        }
    done
done

echo
echo "Done. Inspect the data with:"
echo "  python analysis/01_inspect_dns.py"
