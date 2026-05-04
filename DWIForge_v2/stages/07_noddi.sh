#!/usr/bin/env bash
# stages/06_noddi.sh -- NODDI fitting via AMICO 2.x
# =============================================================================
# Fits the NODDI model using AMICO on the preprocessed DWI from stage 05.
# Delegates to python/noddi.py for all AMICO API calls.
#
# Inputs:
#   dwi_preprocessed.mif    (stage 05)
#   wm_mask_dwi.nii.gz      (stage 06 FAST mask, or brain mask fallback)
#
# Outputs (in DIR_WORK/<sub>/noddi/):
#   NODDI_icvf.nii.gz       NDI  -- neurite density index
#   NODDI_odi.nii.gz        ODI  -- orientation dispersion index
#   NODDI_isovf.nii.gz      ISOVF -- free water fraction
#   NODDI_directions.nii.gz -- principal fiber direction
#
# Environment:
#   AMICO runs under user site-packages (dipy==1.9.0).
#   DWIFORGE_DEPS_DIR must NOT be in PYTHONPATH for this stage.
#   The stage calls python/noddi.py with a clean PYTHONPATH.
#
# Kernel caching:
#   NODDI kernels are protocol-specific. They are generated once and
#   stored at DIR_WORK/../noddi_kernels/ (study level) so all subjects
#   share them. Set DWIFORGE_NODDI_KERNEL_CACHE to override the location.
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
log_stage_start "07_noddi" "$SUB"

# ---------------------------------------------------------------------------
# Check NODDI is appropriate for this subject
# ---------------------------------------------------------------------------

NODDI_CONFIDENCE="standard"
NODDI_ENABLED=true

if [[ -f "$CAPABILITY_JSON" ]]; then
    NODDI_CONFIDENCE=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('capabilities',{}).get('noddi_confidence','standard') or 'standard')
" 2>/dev/null || echo "standard"
    )
    NODDI_ENABLED=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(str(d.get('capabilities',{}).get('noddi',False)).lower())
" 2>/dev/null || echo "true"
    )
fi

if [[ "$NODDI_ENABLED" == "false" ]]; then
    _log WARN "NODDI disabled for ${SUB} in capability profile"
    _log WARN "Reason: insufficient directions or b-value"
    log_stage_end "07_noddi" "$SUB"
    exit 0
fi

_log INFO "NODDI confidence: ${NODDI_CONFIDENCE}"

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------

MIF_FINAL="${WORK}/dwi_preprocessed.mif"
if [[ ! -f "$MIF_FINAL" ]]; then
    _log ERROR "Preprocessed DWI not found: ${MIF_FINAL}"
    _log ERROR "Run stages 02–05 before stage 07"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Export NIfTI + bvals/bvecs from preprocessed .mif
# ---------------------------------------------------------------------------
# AMICO reads NIfTI + FSL gradient files, not .mif.
# mrconvert from the .mif ensures the gradient table is the post-DESIGNER
# corrected table (eddy-corrected rotated bvecs embedded by DESIGNER).

NODDI_WORK="${WORK}/noddi_inputs"
mkdir -p "$NODDI_WORK"

DWI_NII="${NODDI_WORK}/dwi.nii.gz"
DWI_BVAL="${NODDI_WORK}/dwi.bval"
DWI_BVEC="${NODDI_WORK}/dwi.bvec"

if [[ ! -f "$DWI_NII" ]]; then
    _log INFO "Step 1: Exporting NIfTI and gradient files from preprocessed .mif"
    mrconvert "$MIF_FINAL" "$DWI_NII" \
        -export_grad_fsl "$DWI_BVEC" "$DWI_BVAL" \
        -quiet -force
    _log OK "  DWI NIfTI: ${DWI_NII##*/}"
else
    _log INFO "Step 1: NIfTI export already done, skipping"
fi

# ---------------------------------------------------------------------------
# Step 2: Locate mask
# ---------------------------------------------------------------------------
# Prefer FAST WM mask from stage 06; fall back to brain mask from DESIGNER.

MASK="${WORK}/wm_mask_dwi.nii.gz"
if [[ ! -f "$MASK" ]]; then
    _log WARN "WM mask not found — looking for brain mask fallback"
    MASK=$(find "${WORK}/designer_scratch" -name "brain_mask*" \
        2>/dev/null | head -1 || true)
    if [[ -z "$MASK" || ! -f "$MASK" ]]; then
        _log ERROR "No mask found — cannot run NODDI without a mask"
        _log ERROR "Run stage 06 (tensor fitting) before stage 07"
        exit 1
    fi
    _log WARN "  Using brain mask: ${MASK##*/}"
else
    _log INFO "  Using WM mask: ${MASK##*/}"
fi

# ---------------------------------------------------------------------------
# Step 3: Run NODDI via noddi.py
# ---------------------------------------------------------------------------
# IMPORTANT: run without DWIFORGE_DEPS_DIR in PYTHONPATH.
# AMICO 2.x needs dipy==1.9.0 from user site-packages.
# We clear PYTHONPATH entirely and let Python find packages normally.

NODDI_OUT="${WORK}/noddi"
mkdir -p "$NODDI_OUT"

NODDI_DONE="${NODDI_OUT}/NODDI_icvf.nii.gz"

if [[ ! -f "$NODDI_DONE" ]]; then
    _log INFO "Step 3: Running NODDI fitting"

    env PYTHONPATH="" \
        "${PYTHON_EXECUTABLE:-python3}" \
        "${DWIFORGE_ROOT}/python/noddi.py" \
        --dwi            "$DWI_NII" \
        --bval           "$DWI_BVAL" \
        --bvec           "$DWI_BVEC" \
        --mask           "$MASK" \
        --output         "$NODDI_OUT" \
        --b0_threshold   "${DWIFORGE_B0_THRESHOLD:-50}" \
        --nthreads       "${OMP_NUM_THREADS:-4}" \
        --noddi_confidence "$NODDI_CONFIDENCE" \
        --capability_json  "$CAPABILITY_JSON"

    if [[ ! -f "$NODDI_DONE" ]]; then
        _log ERROR "NODDI fitting did not produce expected output"
        exit 1
    fi

    _log OK "  NODDI complete: ${NODDI_OUT##*/}/"
else
    _log INFO "Step 4: NODDI output already exists, skipping"
fi

# ---------------------------------------------------------------------------
# Verify key outputs
# ---------------------------------------------------------------------------

for metric in icvf odi isovf; do
    if [[ ! -f "${NODDI_OUT}/NODDI_${metric}.nii.gz" ]]; then
        _log WARN "Expected NODDI output missing: NODDI_${metric}.nii.gz"
    fi
done

_log OK "Stage 06 complete"
_log OK "  Output: ${NODDI_OUT}"
_log OK "  NDI:    ${NODDI_OUT}/NODDI_icvf.nii.gz"
_log OK "  ODI:    ${NODDI_OUT}/NODDI_odi.nii.gz"
_log OK "  ISOVF:  ${NODDI_OUT}/NODDI_isovf.nii.gz"

log_stage_end "07_noddi" "$SUB"
