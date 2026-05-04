#!/usr/bin/env bash
# stages/08_response_functions.sh — 3-tissue response function estimation
# =============================================================================
# Estimates single-fibre WM, GM, and CSF response functions from the
# preprocessed DWI using the unsupervised dhollander algorithm (MRtrix3).
#
# This is stage 1 of a two-pass FOD estimation workflow:
#
#   Stage 08 (this stage): per-subject response function estimation
#   [Orchestrator barrier]: responsemean — group-averaged response functions
#   Stage 09:              ss3t_csd_beta1 using group-averaged responses
#
# All subjects MUST complete stage 08 before stage 09 can run for any subject.
# The orchestrator (dwiforge.sh) handles the barrier and responsemean call.
#
# Per-subject outputs written to:
#   DIR_WORK/<sub>/responses/
#     response_wm.txt     — single-fibre WM response function
#     response_gm.txt     — GM response function
#     response_csf.txt    — CSF response function
#     response_voxels.mif — voxels used for response estimation (QC)
#
# Group-level outputs written to:
#   DIR_WORK/group/responses/<sub>/
#     response_wm.txt     — copy for group averaging
#     response_gm.txt
#     response_csf.txt
#
# The group/ directory is read by the orchestrator's responsemean step.
#
# Inputs:
#   dwi_preprocessed.mif    (stage 05)
#   DESIGNER brain mask     (stage 05 designer_scratch/)
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
log_stage_start "08_response_functions" "$SUB"

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------

MIF_FINAL="${WORK}/dwi_preprocessed.mif"

if [[ ! -f "$MIF_FINAL" ]]; then
    _log ERROR "Preprocessed DWI not found: ${MIF_FINAL}"
    _log ERROR "Run stages 02-05 before stage 08"
    exit 1
fi

# ---------------------------------------------------------------------------
# Locate DESIGNER brain mask
# ---------------------------------------------------------------------------
# DESIGNER writes its mask to the scratch directory. We search for it
# rather than hardcoding the path since the exact filename varies by version.

BRAIN_MASK=""
DESIGNER_SCRATCH="${WORK}/designer_scratch"

if [[ -d "$DESIGNER_SCRATCH" ]]; then
    BRAIN_MASK=$(find "$DESIGNER_SCRATCH" \
        -name "brain_mask*" -o -name "*brain*mask*" 2>/dev/null | \
        head -1 || true)
fi

# Fallback: check capability.json for mask path
if [[ -z "$BRAIN_MASK" || ! -f "$BRAIN_MASK" ]]; then
    if [[ -f "$CAPABILITY_JSON" ]]; then
        BRAIN_MASK=$(
            "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('tensor_fitting', {}).get('mask_path', '') or '')
" 2>/dev/null || echo ""
        )
    fi
fi

if [[ -z "$BRAIN_MASK" || ! -f "$BRAIN_MASK" ]]; then
    _log ERROR "DESIGNER brain mask not found"
    _log ERROR "Searched: ${DESIGNER_SCRATCH}"
    _log ERROR "Run stage 05 (designer) before stage 08"
    exit 1
fi

_log INFO "Brain mask: ${BRAIN_MASK##*/}"

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

RESP_DIR="${WORK}/responses"
mkdir -p "$RESP_DIR"

# Group-level directory — one subdir per subject for responsemean
GROUP_RESP_DIR="${DWIFORGE_DIR_WORK}/group/responses/${SUB}"
mkdir -p "$GROUP_RESP_DIR"

WM_RESP="${RESP_DIR}/response_wm.txt"
GM_RESP="${RESP_DIR}/response_gm.txt"
CSF_RESP="${RESP_DIR}/response_csf.txt"
RESP_VOXELS="${RESP_DIR}/response_voxels.mif"

# ---------------------------------------------------------------------------
# Step 1: Run dwi2response dhollander
# ---------------------------------------------------------------------------
# The dhollander algorithm is unsupervised — it automatically identifies
# single-fibre WM, GM, and CSF voxels from the data itself. This is
# important for group studies: the same algorithm applied consistently
# across subjects produces comparable response functions.
#
# -voxels: saves the voxels used for response estimation — useful for QC.
# Inspect response_voxels.mif in mrview overlaid on the DWI to confirm
# the algorithm selected anatomically appropriate voxels.

if [[ ! -f "$WM_RESP" ]]; then
    _log INFO "Step 1: Estimating 3-tissue response functions (dhollander)"

    # dwi2response is an MRtrix3 Python script — needs MRTRIX_PYTHON_PATH
    env "PYTHONPATH=${MRTRIX_PYTHON_PATH}" \
        dwi2response dhollander \
            "$MIF_FINAL" \
            "$WM_RESP" \
            "$GM_RESP" \
            "$CSF_RESP" \
            -mask    "$BRAIN_MASK" \
            -voxels  "$RESP_VOXELS" \
            -nthreads "${OMP_NUM_THREADS:-4}" \
            -quiet \
            -force

    _log OK "  WM response:  ${WM_RESP##*/}"
    _log OK "  GM response:  ${GM_RESP##*/}"
    _log OK "  CSF response: ${CSF_RESP##*/}"
    _log OK "  QC voxels:    ${RESP_VOXELS##*/}"
else
    _log INFO "Step 1: Response functions already estimated, skipping"
fi

# Verify outputs exist and are non-empty
for f in "$WM_RESP" "$GM_RESP" "$CSF_RESP"; do
    if [[ ! -s "$f" ]]; then
        _log ERROR "Response function empty or missing: ${f}"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Step 2: Copy to group responses directory
# ---------------------------------------------------------------------------
# The orchestrator's responsemean call will read from group/responses/*/
# after all subjects have completed this stage.

_log INFO "Step 2: Copying responses to group directory"
cp "$WM_RESP"  "${GROUP_RESP_DIR}/response_wm.txt"
cp "$GM_RESP"  "${GROUP_RESP_DIR}/response_gm.txt"
cp "$CSF_RESP" "${GROUP_RESP_DIR}/response_csf.txt"
_log OK "  Group dir: ${GROUP_RESP_DIR}"

# ---------------------------------------------------------------------------
# Log response function values for QC
# ---------------------------------------------------------------------------
# The WM response should show clear b-value dependence (decreasing signal
# with b-value) and single-fibre characteristics. Print first line of each
# for quick sanity check in the log.

_log INFO "Response function summary (first row = b=0 term):"
_log INFO "  WM:  $(head -1 "$WM_RESP")"
_log INFO "  GM:  $(head -1 "$GM_RESP")"
_log INFO "  CSF: $(head -1 "$CSF_RESP")"

# ---------------------------------------------------------------------------
# Update capability profile
# ---------------------------------------------------------------------------

"${PYTHON_EXECUTABLE:-python3}" - << PYEOF
import json

cap_path = '${CAPABILITY_JSON}'
with open(cap_path) as f:
    d = json.load(f)

d['response_functions'] = {
    'status':         'complete',
    'algorithm':      'dhollander',
    'response_dir':   '${RESP_DIR}',
    'group_dir':      '${GROUP_RESP_DIR}',
    'wm_response':    '${WM_RESP}',
    'gm_response':    '${GM_RESP}',
    'csf_response':   '${CSF_RESP}',
    'voxels_qc':      '${RESP_VOXELS}',
    'group_averaged': False,  # updated by orchestrator after responsemean
}

with open(cap_path, 'w') as f:
    json.dump(d, f, indent=2)
print('capability.json updated with response function paths')
PYEOF

_log OK "Stage 08 complete"
_log OK "  Responses: ${RESP_DIR}"
_log OK "  Group:     ${GROUP_RESP_DIR}"
_log INFO "NOTE: All subjects must complete stage 08 before stage 09 can run"
_log INFO "      Orchestrator will run responsemean across all subjects"

log_stage_end "08_response_functions" "$SUB"
