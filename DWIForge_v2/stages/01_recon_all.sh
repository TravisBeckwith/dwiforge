#!/usr/bin/env bash
# stages/01_recon_all.sh — FreeSurfer recon-all
# =============================================================================
# Runs FreeSurfer's recon-all cortical reconstruction pipeline on the subject's
# T1w image. This is stage 01 because it is independent of all DWI preprocessing
# stages (02–07) and takes ~8 hours — running it early minimises idle time
# before tractography (stage 09).
#
# Outputs written to DWIFORGE_DIR_FREESURFER/<sub>/:
#   mri/         — volume reconstructions (T1, aseg, aparc+aseg, etc.)
#   surf/        — cortical surface meshes (lh/rh pial, white, inflated)
#   label/       — cortical parcellation labels
#   stats/       — regional thickness, volume, surface area
#
# Downstream consumers:
#   Stage 09 (tractography):  5ttgen hsvs, labelconvert, tck2connectome
#                              parcellation image for connectome construction
#
# Notes:
#   - recon-all uses the RAW T1w (not skull-stripped). FreeSurfer performs
#     its own internal skull stripping. Do NOT pass the stage 03 brain image.
#   - SUBJECTS_DIR is set to DWIFORGE_DIR_FREESURFER by env_setup.sh.
#   - recon-all is resumable: if interrupted, re-running this stage will
#     continue from where it left off (FreeSurfer checks which steps are done).
#   - The FreeSurfer license must be present at $FREESURFER_HOME/license.txt
#     or at the path specified by $FS_LICENSE.
#   - Runtime: ~8 hours on a single core. Use -parallel -openmp N to speed up.
#     Set DWIFORGE_FS_THREADS in environment or dwiforge.toml to control this.
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
# Also export as DWIFORGE_SUBJECT for any subprocess that needs it
export DWIFORGE_SUBJECT="$SUB"
SRC="${DWIFORGE_DIR_SOURCE:?DWIFORGE_DIR_SOURCE not set}"
LOGS="${DWIFORGE_DIR_LOGS}/${SUB}"
CAPABILITY_JSON="${LOGS}/capability.json"

# FreeSurfer subjects directory — set by env_setup.sh as SUBJECTS_DIR
FS_DIR="${DWIFORGE_DIR_FREESURFER:?DWIFORGE_DIR_FREESURFER not set}"
mkdir -p "$FS_DIR" "$LOGS"

log_stage_start "01_recon_all" "$SUB"

# ---------------------------------------------------------------------------
# Check FreeSurfer is available and licensed
# ---------------------------------------------------------------------------

if ! command -v recon-all >/dev/null 2>&1; then
    _log ERROR "recon-all not found — FreeSurfer is not installed or not on PATH"
    _log ERROR "Set FREESURFER_HOME and source SetUpFreeSurfer.sh"
    exit 1
fi

FS_LICENSE_PATH="${FS_LICENSE:-${FREESURFER_HOME}/license.txt}"
if [[ ! -f "$FS_LICENSE_PATH" ]]; then
    _log ERROR "FreeSurfer license not found: ${FS_LICENSE_PATH}"
    _log ERROR "Mount or copy license.txt to \$FREESURFER_HOME/license.txt"
    _log ERROR "Or set FS_LICENSE=/path/to/license.txt"
    exit 1
fi

_log INFO "FreeSurfer: $(recon-all --version 2>&1 | head -1 || echo 'version unknown')"
_log INFO "SUBJECTS_DIR: ${FS_DIR}"

# ---------------------------------------------------------------------------
# Find raw T1w — from capability profile or BIDS anat directory
# ---------------------------------------------------------------------------
# recon-all needs the RAW un-skull-stripped T1w.
# Stage 03 (t1w_prep) skull-strips for Synb0 — that output is NOT used here.

T1W_SRC=""
if [[ -f "$CAPABILITY_JSON" ]]; then
    T1W_SRC=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('acquisition', {}).get('t1w_file', '') or '')
" 2>/dev/null || echo ""
    )
fi

if [[ -z "$T1W_SRC" || ! -f "$T1W_SRC" ]]; then
    ANAT_DIR="${SRC}/${SUB}/anat"
    for pattern in \
        "${ANAT_DIR}/${SUB}_T1w.nii.gz" \
        "${ANAT_DIR}/${SUB}_acq-MPRAGE_T1w.nii.gz"; do
        if [[ -f "$pattern" ]]; then
            T1W_SRC="$pattern"
            break
        fi
    done
    if [[ -z "$T1W_SRC" ]]; then
        T1W_SRC=$(find "${ANAT_DIR}" -name "*_T1w.nii.gz" \
            2>/dev/null | head -1 || true)
    fi
fi

if [[ -z "$T1W_SRC" || ! -f "$T1W_SRC" ]]; then
    _log WARN "No T1w found for ${SUB} — skipping recon-all"
    _log WARN "Tractography parcellation will not be available"
    log_stage_end "01_recon_all" "$SUB"
    exit 0
fi

_log INFO "T1w: ${T1W_SRC##*/}"

# ---------------------------------------------------------------------------
# Check if recon-all is already complete
# ---------------------------------------------------------------------------
# FreeSurfer writes recon-all.done when the full pipeline finishes.
# If this file exists, skip entirely. If it doesn't but partial output
# exists, recon-all will resume from where it left off (-resume flag).

FS_SUBJECT_DIR="${FS_DIR}/${SUB}"
RECON_DONE="${FS_SUBJECT_DIR}/scripts/recon-all.done"
RECON_RUNNING="${FS_SUBJECT_DIR}/scripts/recon-all.running"

if [[ -f "$RECON_DONE" ]]; then
    _log INFO "recon-all already complete for ${SUB} — skipping"
    log_stage_end "01_recon_all" "$SUB"
    exit 0
fi

# Warn if a previous run may have crashed mid-stream
if [[ -f "$RECON_RUNNING" ]]; then
    _log WARN "recon-all.running flag found — previous run may have crashed"
    _log WARN "FreeSurfer will resume from the last completed step"
    rm -f "$RECON_RUNNING"
fi

# ---------------------------------------------------------------------------
# Run recon-all
# ---------------------------------------------------------------------------

FS_THREADS="${DWIFORGE_FS_THREADS:-1}"
_log INFO "Running recon-all (threads: ${FS_THREADS}, ~8 hrs at 1 thread)"
_log INFO "Output: ${FS_SUBJECT_DIR}"

# Check for existing partial output (resume) vs fresh run
if [[ -d "$FS_SUBJECT_DIR" ]]; then
    _log INFO "Partial output found — resuming recon-all"
    FS_EXTRA="${DWIFORGE_FS_EXTRA_FLAGS:--3T}"
    RECON_CMD=(
        recon-all
        -subjid    "$SUB"
        -sd        "$FS_DIR"
        -all
        -openmp    "$FS_THREADS"
        ${FS_EXTRA}
    )
else
    _log INFO "Starting fresh recon-all"
    # Extra flags from dwiforge.toml [freesurfer] extra_flags
    FS_EXTRA="${DWIFORGE_FS_EXTRA_FLAGS:--3T}"

    RECON_CMD=(
        recon-all
        -subjid    "$SUB"
        -i         "$T1W_SRC"
        -sd        "$FS_DIR"
        -all
        -openmp    "$FS_THREADS"
        ${FS_EXTRA}
    )
fi

"${RECON_CMD[@]}"

# ---------------------------------------------------------------------------
# Verify output
# ---------------------------------------------------------------------------

if [[ ! -f "$RECON_DONE" ]]; then
    _log ERROR "recon-all did not complete — recon-all.done not found"
    _log ERROR "Check FreeSurfer log: ${FS_SUBJECT_DIR}/scripts/recon-all.log"
    exit 1
fi

# Check key outputs that downstream stages depend on
REQUIRED_OUTPUTS=(
    "${FS_SUBJECT_DIR}/mri/aparc+aseg.mgz"       # parcellation volume
    "${FS_SUBJECT_DIR}/mri/aseg.mgz"              # subcortical segmentation
    "${FS_SUBJECT_DIR}/surf/lh.white"             # left hemisphere white surface
    "${FS_SUBJECT_DIR}/surf/rh.white"             # right hemisphere white surface
    "${FS_SUBJECT_DIR}/surf/lh.pial"              # left hemisphere pial surface
    "${FS_SUBJECT_DIR}/surf/rh.pial"              # right hemisphere pial surface
)

MISSING=()
for f in "${REQUIRED_OUTPUTS[@]}"; do
    [[ ! -f "$f" ]] && MISSING+=("${f##*/}")
done

if [[ "${#MISSING[@]}" -gt 0 ]]; then
    _log WARN "Some expected outputs missing: ${MISSING[*]}"
    _log WARN "Check: ${FS_SUBJECT_DIR}/scripts/recon-all.log"
fi

# ---------------------------------------------------------------------------
# Update capability profile
# ---------------------------------------------------------------------------

"${PYTHON_EXECUTABLE:-python3}" - << PYEOF
import json, os

cap_path = '${CAPABILITY_JSON}'
with open(cap_path) as f:
    d = json.load(f)

fs_dir = '${FS_SUBJECT_DIR}'
d['recon_all'] = {
    'status':       'complete',
    'subject_dir':  fs_dir,
    'aparc_aseg':   os.path.join(fs_dir, 'mri', 'aparc+aseg.mgz'),
    'aseg':         os.path.join(fs_dir, 'mri', 'aseg.mgz'),
    'surf_dir':     os.path.join(fs_dir, 'surf'),
    'label_dir':    os.path.join(fs_dir, 'label'),
    'threads':      int('${FS_THREADS}'),
}

with open(cap_path, 'w') as f:
    json.dump(d, f, indent=2)
print('capability.json updated with recon_all paths')
PYEOF

_log OK "Stage 01 complete — recon-all done"
_log OK "  Subject dir: ${FS_SUBJECT_DIR}"
_log OK "  aparc+aseg:  ${FS_SUBJECT_DIR}/mri/aparc+aseg.mgz"

log_stage_end "01_recon_all" "$SUB"
