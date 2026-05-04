#!/usr/bin/env bash
# stages/03_epi_correction.sh — EPI susceptibility distortion correction
# =============================================================================
# Corrects susceptibility-induced geometric distortions in the DWI b0.
# Produces a synthetic reverse-PE b0 (or flags that SDC was skipped) for use
# by stage 05 (DESIGNER with -rpe_pair).
#
# Decision tree:
#   Path A: Reverse PE b0 available → topup field map
#   Path B: No reverse PE, T1w available, Synb0 available → Synb0-DisCo
#   Path C: No reverse PE, T1w available, Synb0 absent → last resort (ANTs)
#   Path D: No T1w, no reverse PE → skip SDC (eddy motion-only)
#
# For the CLS dataset: Path B (Synb0) applies.
# All subjects have T1w; none have reverse PE b0.
#
# Outputs:
#   b0_synthetic.nii.gz   — undistorted b0 (Synb0 output or topup result)
#   fieldmap.nii.gz       — displacement field for eddy (if topup was run)
#   epi_method.txt        — records which path was taken (read by stage 05)
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
SRC="${DWIFORGE_DIR_SOURCE:?DWIFORGE_DIR_SOURCE not set}"
WORK="${DWIFORGE_DIR_WORK}/${SUB}"
LOGS="${DWIFORGE_DIR_LOGS}/${SUB}"
CAPABILITY_JSON="${LOGS}/capability.json"

dirs_init "$SUB"
log_stage_start "04_epi_correction" "$SUB"

# ---------------------------------------------------------------------------
# Read acquisition metadata from capability profile
# ---------------------------------------------------------------------------

RPE_AVAILABLE=false
T1W_BRAIN=""
T1W_AVAILABLE=false
TOPUP_READY=false
PE_DIR="j"

if [[ -f "$CAPABILITY_JSON" ]]; then
    RPE_AVAILABLE=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(str(d.get('acquisition',{}).get('reverse_pe_available', False)).lower())
" 2>/dev/null || echo "false"
    )
    TOPUP_READY=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(str(d.get('acquisition',{}).get('topup_ready', False)).lower())
" 2>/dev/null || echo "false"
    )
    T1W_BRAIN=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('t1w_prep', {}).get('t1w_brain', '') or '')
" 2>/dev/null || echo ""
    )
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

[[ -n "$T1W_BRAIN" && -f "$T1W_BRAIN" ]] && T1W_AVAILABLE=true

# Locate primary b0 in the degibbs mif from stage 02
MIF_DEGIBBS="${WORK}/dwi/dwi_degibbs.mif"

if [[ ! -f "$MIF_DEGIBBS" ]]; then
    _log ERROR "Stage 01 output not found: ${MIF_DEGIBBS}"
    _log ERROR "Run stage 02 before stage 04"
    exit 1
fi

# Outputs
B0_DISTORTED="${WORK}/b0_distorted.nii.gz"
B0_SYNTHETIC="${WORK}/b0_synthetic.nii.gz"
FIELDMAP_OUT="${WORK}/fieldmap.nii.gz"
EPI_METHOD_FILE="${WORK}/epi_method.txt"

# Extract b0 from degibbs mif (used by all paths)
if [[ ! -f "$B0_DISTORTED" ]]; then
    _log INFO "Extracting b0 from degibbs volume"
    dwiextract "$MIF_DEGIBBS" - -bzero -quiet | \
        mrconvert - "$B0_DISTORTED" -coord 3 0 -quiet -force
fi

# ---------------------------------------------------------------------------
# Path A: Reverse PE b0 available — run topup
# ---------------------------------------------------------------------------

if [[ "$TOPUP_READY" == "true" && "$RPE_AVAILABLE" == "true" ]]; then
    _log INFO "Path A: Reverse PE b0 available — running topup"

    RPE_NII=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('acquisition',{}).get('reverse_pe_file','') or '')
" 2>/dev/null || echo ""
    )
    TRT=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('acquisition',{}).get('total_readout_time', 0.04))
" 2>/dev/null || echo "0.04"
    )

    TOPUP_DIR="${WORK}/topup"
    mkdir -p "$TOPUP_DIR"

    # Merge b0 pair and write datain
    MERGED_B0="${TOPUP_DIR}/b0_pair.nii.gz"
    fslmerge -t "$MERGED_B0" "$B0_DISTORTED" "$RPE_NII"

    DATAIN="${TOPUP_DIR}/datain.txt"
    printf "%s 0 %s\n%s 0 -%s\n" \
        "${PE_DIR//j/0 1}" "$TRT" \
        "${PE_DIR//j/0 1}" "$TRT" \
        > "$DATAIN"

    topup \
        --imain="$MERGED_B0" \
        --datain="$DATAIN" \
        --config=b02b0.cnf \
        --out="${TOPUP_DIR}/topup" \
        --fout="$FIELDMAP_OUT" \
        --iout="$B0_SYNTHETIC"

    echo "topup" > "$EPI_METHOD_FILE"
    _log OK "  Topup complete — fieldmap: ${FIELDMAP_OUT##*/}"

# ---------------------------------------------------------------------------
# Path B: No reverse PE, T1w available — Synb0-DisCo
# ---------------------------------------------------------------------------

elif [[ "$T1W_AVAILABLE" == "true" ]] && \
     ( command -v docker >/dev/null 2>&1 || command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1 ); then
    _log INFO "Path B: No reverse PE — running Synb0-DisCo"
    _log INFO "  T1w brain:   ${T1W_BRAIN}"
    _log INFO "  b0:          ${B0_DISTORTED}"

    SYNB0_WORK="${WORK}/synb0"
    SYNB0_IN="${SYNB0_WORK}/INPUTS"
    SYNB0_OUT="${SYNB0_WORK}/OUTPUTS"
    mkdir -p "$SYNB0_IN" "$SYNB0_OUT"

    cp "$B0_DISTORTED" "${SYNB0_IN}/b0.nii.gz"
    cp "$T1W_BRAIN"    "${SYNB0_IN}/T1.nii.gz"

    TRT=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
acq = d.get('acquisition', {})
trt = acq.get('total_readout_time') or 0.04
print(trt)
" 2>/dev/null || echo "0.04"
    )
    case "${PE_DIR}" in
        j|AP)  printf "0 1 0 %s\n"  "$TRT" > "${SYNB0_IN}/acqparams.txt" ;;
        j-|PA) printf "0 -1 0 %s\n" "$TRT" > "${SYNB0_IN}/acqparams.txt" ;;
        i|LR)  printf "1 0 0 %s\n"  "$TRT" > "${SYNB0_IN}/acqparams.txt" ;;
        i-|RL) printf "-1 0 0 %s\n" "$TRT" > "${SYNB0_IN}/acqparams.txt" ;;
        *)     printf "0 1 0 %s\n"  "$TRT" > "${SYNB0_IN}/acqparams.txt" ;;
    esac
    _log INFO "  acqparams: $(cat ${SYNB0_IN}/acqparams.txt)"

    SYNB0_IMAGE="leonyichencai/synb0-disco:v3.1"
    FS_LICENSE="${FS_LICENSE:-${FREESURFER_HOME}/license.txt}"
    if [[ ! -f "$FS_LICENSE" ]]; then
        _log ERROR "FreeSurfer license not found: ${FS_LICENSE}"
        exit 1
    fi

    _log INFO "  Running Synb0-DisCo via Docker (~20 min)..."
    if command -v docker >/dev/null 2>&1; then
        docker run --rm \
            -v "${SYNB0_IN}:/INPUTS/" \
            -v "${SYNB0_OUT}:/OUTPUTS/" \
            -v "${FS_LICENSE}:/extra/freesurfer/license.txt" \
            "${SYNB0_IMAGE}" --notopup \
            > "${SYNB0_OUT}/synb0_log.txt" 2>&1
    else
        _SING=$(command -v apptainer 2>/dev/null || command -v singularity)
        _SIF="${WORK}/synb0_v3.1.sif"
        if [[ ! -f "$_SIF" ]]; then
            _log INFO "  Building Singularity image (one-time)..."
            "$_SING" build "$_SIF" "docker://${SYNB0_IMAGE}"
        fi
        "$_SING" exec \
            --bind "${SYNB0_IN}:/INPUTS/,${SYNB0_OUT}:/OUTPUTS/,${FS_LICENSE}:/extra/freesurfer/license.txt" \
            "$_SIF" --notopup \
            > "${SYNB0_OUT}/synb0_log.txt" 2>&1
    fi

    SYNB0_B0="${SYNB0_OUT}/b0_u.nii.gz"
    if [[ ! -f "$SYNB0_B0" ]]; then
        _log ERROR "Synb0 did not produce expected output: ${SYNB0_B0}"
        _log ERROR "Check: ${SYNB0_OUT}/synb0_log.txt"
        exit 1
    fi

    cp "$SYNB0_B0" "$B0_SYNTHETIC"
    echo "synb0" > "$EPI_METHOD_FILE"
    _log OK "  Synb0 complete — synthetic b0: ${B0_SYNTHETIC##*/}"

# ---------------------------------------------------------------------------
# Path C: No reverse PE, T1w available, Synb0 absent — last resort ANTs warp
# ---------------------------------------------------------------------------

elif [[ "$T1W_AVAILABLE" == "true" ]]; then
    _log WARN "Path C: Synb0 not available — using last resort ANTs warp"
    _log WARN "This approach is less accurate than Synb0. Consider adding"
    _log WARN "Synb0-DisCo to the container for better SDC."

    T1W_N4=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('t1w_prep', {}).get('t1w_n4', '') or '')
" 2>/dev/null || echo ""
    )
    [[ -z "$T1W_N4" || ! -f "$T1W_N4" ]] && T1W_N4="$T1W_BRAIN"

    LASTRESORT_DIR="${WORK}/epi_lastresort"
    mkdir -p "$LASTRESORT_DIR"

    # Step 1: Linear registration b0 → T1w brain (9 DOF, no shearing)
    _log INFO "  Step 1: Linear registration b0 → T1w (9 DOF)"
    B0_LIN="${LASTRESORT_DIR}/b0_linear.nii.gz"
    flirt \
        -in "$B0_DISTORTED" \
        -ref "$T1W_BRAIN" \
        -out "$B0_LIN" \
        -omat "${LASTRESORT_DIR}/b0_to_t1w_affine.mat" \
        -dof 9

    # Step 2: Nonlinear warp b0 → T1w with SyNQuick
    _log INFO "  Step 2: Nonlinear registration (ANTs SyNQuick)"
    antsRegistrationSyNQuick.sh \
        -d 3 \
        -f "$T1W_BRAIN" \
        -m "$B0_LIN" \
        -o "${LASTRESORT_DIR}/b0_nonlin_" \
        -t s

    # Step 3: Apply warp to produce undistorted b0
    antsApplyTransforms \
        -d 3 \
        -i "$B0_DISTORTED" \
        -r "$T1W_BRAIN" \
        -o "$B0_SYNTHETIC" \
        -t "${LASTRESORT_DIR}/b0_nonlin_1Warp.nii.gz" \
        -t "${LASTRESORT_DIR}/b0_nonlin_0GenericAffine.mat"

    echo "lastresort_ants" > "$EPI_METHOD_FILE"
    _log WARN "  Last resort warp complete — geometric distortion partially corrected"
    _log WARN "  Results should be visually inspected before proceeding"

# ---------------------------------------------------------------------------
# Path D: No T1w, no reverse PE — skip SDC entirely
# ---------------------------------------------------------------------------

else
    _log WARN "Path D: No T1w and no reverse PE — susceptibility distortion"
    _log WARN "        correction SKIPPED. Eddy will correct motion and eddy"
    _log WARN "        currents only. Geometric distortion will remain."
    _log WARN "        This will affect tractography accuracy."

    # Copy distorted b0 as placeholder so stage 05 has consistent inputs
    cp "$B0_DISTORTED" "$B0_SYNTHETIC"
    echo "none" > "$EPI_METHOD_FILE"
fi

# ---------------------------------------------------------------------------
# Update capability profile with EPI correction result
# ---------------------------------------------------------------------------

EPI_METHOD=$(cat "$EPI_METHOD_FILE" 2>/dev/null || echo "unknown")

"${PYTHON_EXECUTABLE:-python3}" - <<PYEOF
import json

cap_path = '${CAPABILITY_JSON}'
with open(cap_path) as f:
    d = json.load(f)

d['epi_correction'] = {
    'method':          '${EPI_METHOD}',
    'b0_synthetic':    '${B0_SYNTHETIC}',
    'fieldmap':        '${FIELDMAP_OUT}' if '${EPI_METHOD}' == 'topup' else None,
    'sdc_performed':   '${EPI_METHOD}' != 'none',
}

with open(cap_path, 'w') as f:
    json.dump(d, f, indent=2)
print(f'capability.json updated: epi_method=${EPI_METHOD}')
PYEOF

_log OK "Stage 03 complete — EPI method: ${EPI_METHOD}"
_log OK "  Synthetic b0: ${B0_SYNTHETIC}"

log_stage_end "04_epi_correction" "$SUB"
