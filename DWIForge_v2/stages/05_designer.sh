#!/usr/bin/env bash
# stages/04_designer.sh — DESIGNER preprocessing (eddy + Rician + mask)
# =============================================================================
# Runs DESIGNER-v2 on the Gibbs-corrected DWI using the EPI correction result
# from stage 04. Handles three cases:
#
#   sdc_method = topup or synb0  → -rpe_pair <b0_synthetic>
#   sdc_method = lastresort_ants → -rpe_pair <b0_synthetic>  (with warning)
#   sdc_method = none            → -rpe_none  (motion + eddy only)
#
# Inputs (from prior stages):
#   dwi_degibbs.mif      — stage 02 output
#   dwi_sigma.nii        — stage 02 sigma map (for Rician correction)
#   b0_synthetic.nii.gz  — stage 04 output
#
# Output:
#   dwi_preprocessed.mif — final preprocessed DWI (consumed by stage 06+)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DWIFORGE_ROOT="$(dirname "$SCRIPT_DIR")"

source "${DWIFORGE_ROOT}/lib/logging.sh"
source "${DWIFORGE_ROOT}/lib/utils.sh"
source "${DWIFORGE_ROOT}/lib/env_setup.sh"

setup_environment

_log() { log_sub "$1" "$SUB" "${*:2}"; }

# ---------------------------------------------------------------------------
# Resolve subject and paths
# ---------------------------------------------------------------------------

SUB="${1:?Usage: $0 <subject_id>}"
export DWIFORGE_SUBJECT="$SUB"
WORK="${DWIFORGE_DIR_WORK}/${SUB}"
LOGS="${DWIFORGE_DIR_LOGS}/${SUB}"
CAPABILITY_JSON="${LOGS}/capability.json"

dirs_init "$SUB"
log_stage_start "05_designer" "$SUB"

# ---------------------------------------------------------------------------
# Check DESIGNER is available
# ---------------------------------------------------------------------------

if [[ "${DESIGNER_AVAILABLE:-false}" != "true" ]]; then
    _log ERROR "DESIGNER not available — cannot run stage 05"
    _log ERROR "Install: pip install git+https://github.com/NYU-DiffusionMRI/DESIGNER-v2.git"
    exit 1
fi

# ---------------------------------------------------------------------------
# Read stage 02 and 03 outputs
# ---------------------------------------------------------------------------

MIF_DEGIBBS="${WORK}/dwi/dwi_degibbs.mif"
SIGMA_MAP="${WORK}/dwi/dwi_sigma.nii"
B0_SYNTHETIC="${WORK}/b0_synthetic.nii.gz"
EPI_METHOD_FILE="${WORK}/epi_method.txt"

for f in "$MIF_DEGIBBS" "$B0_SYNTHETIC"; do
    if [[ ! -f "$f" ]]; then
        _log ERROR "Required input not found: ${f}"
        _log ERROR "Ensure stages 01 and 03 completed successfully"
        exit 1
    fi
done

EPI_METHOD=$(cat "$EPI_METHOD_FILE" 2>/dev/null || echo "none")
_log INFO "EPI correction method from stage 04: ${EPI_METHOD}"

# ---------------------------------------------------------------------------
# Read phase encoding direction
# ---------------------------------------------------------------------------

PE_DIR="j"
if [[ -f "$CAPABILITY_JSON" ]]; then
    PE_DIR=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
acq = d.get('acquisition', {})
print(acq.get('phase_encoding_direction') or acq.get('phase_encoding_axis') or 'j')
" 2>/dev/null || echo "j"
    )
fi

# ---------------------------------------------------------------------------
# Run DESIGNER
# ---------------------------------------------------------------------------

DESIGNER_OUT_BASE="${WORK}/dwi_designer"
DESIGNER_OUT="${DESIGNER_OUT_BASE}.nii"  # DESIGNER writes NIfTI, converted to .mif after
MIF_FINAL="${WORK}/dwi_preprocessed.mif"

if [[ ! -f "$MIF_FINAL" ]]; then
    _log INFO "Running DESIGNER (eddy + Rician + mask)"

    # Build base command — PYTHONPATH isolation critical:
    # DESIGNER needs dipy==1.9.0 from user site-packages.
    # DO NOT inherit DWIFORGE_DEPS_DIR (contains dipy>=1.12).
    DESIGNER_PF=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
acq = d.get('acquisition', {})
pf = acq.get('partial_fourier_fraction', 'none')
print('1' if pf in ('none', '1', '8/8') or acq.get('fully_sampled', True) else pf)
" 2>/dev/null || echo "1"
    )

    DESIGNER_CMD=(
        env
        "PYTHONPATH=${MRTRIX_PYTHON_PATH}"
        "${DESIGNER_BIN}"
        -eddy
        -pe_dir "${PE_DIR}"
        -rician
        -mask
        -pf "${DESIGNER_PF}"
        -scratch "${WORK}/designer_scratch"
        -nocleanup
        -n_cores "${OMP_NUM_THREADS:-4}"
        -nthreads "${OMP_NUM_THREADS:-4}"
        -force
    )

    # EPI correction flag — depends on stage 04 result
    if [[ "$EPI_METHOD" == "none" ]]; then
        DESIGNER_CMD+=(-rpe_none)
        _log WARN "  No SDC performed — DESIGNER will correct motion and eddy only"
        _log WARN "  Geometric distortion will remain in the output"
    else
        DESIGNER_CMD+=(-rpe_pair "$B0_SYNTHETIC")
        _log INFO "  Using synthetic b0 for SDC: ${B0_SYNTHETIC##*/}"
        if [[ "$EPI_METHOD" == "lastresort_ants" ]]; then
            _log WARN "  SDC derived from last-resort ANTs warp — lower accuracy than Synb0"
        fi
    fi

    # Sigma map for Rician correction (from stage 02 step 3b)
    if [[ -f "$SIGMA_MAP" ]]; then
        DESIGNER_CMD+=(-noisemap "$SIGMA_MAP")
        _log INFO "  Sigma map: ${SIGMA_MAP##*/}"
    else
        _log WARN "  No sigma map — Rician correction will use internal estimate"
    fi

    # Optional config flags
    if [[ "${DWIFORGE_DESIGNER_B1CORRECT:-false}" == "true" ]]; then
        DESIGNER_CMD+=(-b1correct)
        _log INFO "  B1 field correction enabled"
    fi
    if [[ "${DWIFORGE_DESIGNER_NORMALIZE:-false}" == "true" ]]; then
        DESIGNER_CMD+=(-normalize)
    fi

    # Input and output basename (DESIGNER appends .mif)
    DESIGNER_CMD+=("${MIF_DEGIBBS}" "${DESIGNER_OUT_BASE}")

    _log INFO "  DESIGNER command built — running..."
    "${DESIGNER_CMD[@]}"

    # DESIGNER writes NIfTI + bvec/bval (not .mif directly)
    DESIGNER_NII="${DESIGNER_OUT_BASE}.nii"
    DESIGNER_BVEC="${DESIGNER_OUT_BASE}.bvec"
    DESIGNER_BVAL="${DESIGNER_OUT_BASE}.bval"

    # Wait briefly for filesystem
    sleep 1

    if [[ ! -f "$DESIGNER_NII" ]]; then
        _log ERROR "DESIGNER did not produce expected output: ${DESIGNER_NII}"
        _log ERROR "Expected: ${DESIGNER_OUT_BASE}.nii"
        ls -la "${WORK}/dwi_designer"* 2>/dev/null || true
        exit 1
    fi

    # Convert DESIGNER NIfTI output to .mif embedding corrected gradient table.
    # CRITICAL: use eddy-rotated bvecs (dwi_post_eddy.eddy_rotated_bvecs).
    # Eddy resamples voxels and rotates gradients to match.
    # Using original bvecs with eddy-corrected data causes chaotic tensor fits.
    _log INFO "  Converting DESIGNER output to .mif..."

    # Prefer eddy-rotated bvecs from scratch directory
    EDDY_ROTATED_BVEC=$(find "${WORK}/designer_scratch" \
        -name "*.eddy_rotated_bvecs" 2>/dev/null | head -1 || true)

    if [[ -n "$EDDY_ROTATED_BVEC" && -f "$EDDY_ROTATED_BVEC" ]]; then
        _log INFO "  Using eddy-rotated bvecs: ${EDDY_ROTATED_BVEC##*/}"
        BVEC_FOR_MIF="$EDDY_ROTATED_BVEC"
    elif [[ -f "$DESIGNER_BVEC" ]]; then
        _log WARN "  eddy_rotated_bvecs not found — using original bvecs (tensors may be wrong)"
        BVEC_FOR_MIF="$DESIGNER_BVEC"
    else
        BVEC_FOR_MIF=$(find "${WORK}/designer_scratch" -name "*.bvec" 2>/dev/null | head -1 || true)
    fi

    BVAL_FOR_MIF="$DESIGNER_BVAL"
    if [[ ! -f "$BVAL_FOR_MIF" ]]; then
        BVAL_FOR_MIF=$(find "${WORK}/designer_scratch" -name "*.bval" 2>/dev/null | head -1 || true)
    fi

    if [[ -z "$BVEC_FOR_MIF" || ! -f "$BVEC_FOR_MIF" ]]; then
        _log ERROR "No bvec found for mrconvert — cannot create .mif"
        exit 1
    fi

    mrconvert "$DESIGNER_NII" "$MIF_FINAL" \
        -fslgrad "$BVEC_FOR_MIF" "$BVAL_FOR_MIF" \
        -quiet -force

    if ! output_sanity_check "$SUB" "$MIF_FINAL" 5; then
        _log ERROR "DESIGNER output too small: ${MIF_FINAL}"
        exit 1
    fi
    _log OK "  DESIGNER complete: ${MIF_FINAL##*/}"

else
    _log INFO "Stage 04 output already exists, skipping"
fi

# ---------------------------------------------------------------------------
# Update capability profile
# ---------------------------------------------------------------------------

"${PYTHON_EXECUTABLE:-python3}" - <<PYEOF
import json

cap_path = '${CAPABILITY_JSON}'
with open(cap_path) as f:
    d = json.load(f)

d['preprocessing'] = {
    'status':            'complete',
    'output':            '${MIF_FINAL}',
    'epi_method':        '${EPI_METHOD}',
    'designer_version':  '2.0.15',
    'sdc_performed':     '${EPI_METHOD}' != 'none',
}

with open(cap_path, 'w') as f:
    json.dump(d, f, indent=2)
print('capability.json updated with preprocessing status')
PYEOF

if ! output_sanity_check "$SUB" "$MIF_FINAL" 5; then
    _log ERROR "Final output missing or too small: ${MIF_FINAL}"
    exit 1
fi

_log OK "Stage 04 complete"
_log OK "  Preprocessed DWI: ${MIF_FINAL}"

log_stage_end "05_designer" "$SUB"
