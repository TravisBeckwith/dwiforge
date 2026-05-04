#!/usr/bin/env bash
# stages/02_t1w_prep.sh — T1w preprocessing
# =============================================================================
# Prepares the T1w structural image for use by downstream stages:
#   Stage 03: epi_correction.sh  (Synb0 requires skull-stripped + N4 T1w)
#   Stage 06: connectivity.sh    (recon-all / SynthMorph registration)
#
# Steps:
#   1. Reorient T1w to standard (fslreorient2std)
#   2. N4 bias field correction (ANTs N4BiasFieldCorrection)
#   3. Skull stripping (SynthStrip if available, else BET)
#
# Outputs written to DIR_WORK/<sub>/:
#   t1w_reoriented.nii.gz   — after fslreorient2std
#   t1w_n4.nii.gz           — after N4 correction
#   t1w_brain.nii.gz        — skull-stripped (consumed by stage 04 + 06)
#   t1w_brain_mask.nii.gz   — binary brain mask
#
# If no T1w is available this stage exits with a warning, not an error.
# Downstream stages that require T1w (Synb0, recon-all) will log accordingly.
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
log_stage_start "03_t1w_prep" "$SUB"

# ---------------------------------------------------------------------------
# Find T1w — read path from capability profile
# ---------------------------------------------------------------------------

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

# Fallback: search BIDS anat directory directly
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
    # Broad glob fallback
    if [[ -z "$T1W_SRC" ]]; then
        T1W_SRC=$(find "${ANAT_DIR}" -name "*_T1w.nii.gz" 2>/dev/null | head -1 || true)
    fi
fi

if [[ -z "$T1W_SRC" || ! -f "$T1W_SRC" ]]; then
    _log WARN "No T1w found for ${SUB} — skipping T1w prep"
    _log WARN "Synb0-DisCo and recon-all will not be available for this subject"
    log_stage_end "03_t1w_prep" "$SUB"
    exit 0
fi

_log INFO "T1w source: ${T1W_SRC##*/}"

# Output paths
T1W_REORIENTED="${WORK}/t1w_reoriented.nii.gz"
T1W_N4="${WORK}/t1w_n4.nii.gz"
T1W_BRAIN="${WORK}/t1w_brain.nii.gz"
T1W_MASK="${WORK}/t1w_brain_mask.nii.gz"

# ---------------------------------------------------------------------------
# Step 1: Reorient to standard orientation
# ---------------------------------------------------------------------------

if [[ ! -f "$T1W_REORIENTED" ]]; then
    _log INFO "Step 1: Reorienting T1w to standard orientation"
    fslreorient2std "$T1W_SRC" "$T1W_REORIENTED"
    _log OK "  Reoriented: ${T1W_REORIENTED##*/}"
else
    _log INFO "Step 1: Reorientation already done, skipping"
fi

# ---------------------------------------------------------------------------
# Step 2: N4 bias field correction
# ---------------------------------------------------------------------------
# Required before Synb0-DisCo and recon-all for accurate registration.
# Uses ANTs N4BiasFieldCorrection if available, else skips with warning.

if [[ ! -f "$T1W_N4" ]]; then
    _log INFO "Step 2: N4 bias field correction"

    if command -v N4BiasFieldCorrection >/dev/null 2>&1; then
        N4BiasFieldCorrection \
            -d 3 \
            -i "$T1W_REORIENTED" \
            -o "$T1W_N4" \
            -s 4 \
            -b "[200]" \
            -c "[50x50x50x50,0.0001]"
        _log OK "  N4 corrected: ${T1W_N4##*/}"
    else
        _log WARN "  N4BiasFieldCorrection not found — copying reoriented T1w as fallback"
        _log WARN "  Synb0 and recon-all accuracy may be reduced without N4 correction"
        cp "$T1W_REORIENTED" "$T1W_N4"
    fi
else
    _log INFO "Step 2: N4 correction already done, skipping"
fi

# ---------------------------------------------------------------------------
# Step 3: Skull stripping
# ---------------------------------------------------------------------------
# Priority: SynthStrip (FreeSurfer 7.3+) → BET
# SynthStrip is preferred — more robust at tissue boundaries, no tuning needed.
# BET is reliable but may require -f threshold adjustment for some subjects.

if [[ ! -f "$T1W_BRAIN" ]]; then
    _log INFO "Step 3: Skull stripping"

    if command -v mri_synthstrip >/dev/null 2>&1; then
        _log INFO "  Using SynthStrip"
        mri_synthstrip \
            -i "$T1W_N4" \
            -o "$T1W_BRAIN" \
            -m "$T1W_MASK"
        _log OK "  SynthStrip complete: ${T1W_BRAIN##*/}"

    elif command -v bet >/dev/null 2>&1; then
        _log INFO "  Using FSL BET (SynthStrip not found)"
        # -f 0.3 is gentler than default 0.5 — errs toward keeping tissue
        # -R refines the bet estimate, improving robustness for T1w
        # -m generates the binary brain mask
        bet "$T1W_N4" "${T1W_BRAIN%.nii.gz}" \
            -f 0.3 \
            -R \
            -m
        # BET adds _brain to output name — ensure expected path exists
        if [[ ! -f "$T1W_BRAIN" ]]; then
            _log ERROR "  BET did not produce expected output: ${T1W_BRAIN}"
            exit 1
        fi
        _log OK "  BET complete: ${T1W_BRAIN##*/}"
    else
        _log ERROR "  No skull stripping tool available (need SynthStrip or BET)"
        exit 1
    fi
else
    _log INFO "Step 3: Skull stripping already done, skipping"
fi

# ---------------------------------------------------------------------------
# Verify outputs
# ---------------------------------------------------------------------------

if ! output_sanity_check "$SUB" "$T1W_BRAIN" 1; then
    _log ERROR "T1w brain output too small or missing: ${T1W_BRAIN}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Update capability profile with T1w prep paths
# ---------------------------------------------------------------------------

"${PYTHON_EXECUTABLE:-python3}" - <<PYEOF
import json, os

cap_path = '${CAPABILITY_JSON}'
with open(cap_path) as f:
    d = json.load(f)

d.setdefault('t1w_prep', {}).update({
    't1w_reoriented':  '${T1W_REORIENTED}',
    't1w_n4':          '${T1W_N4}',
    't1w_brain':       '${T1W_BRAIN}',
    't1w_brain_mask':  '${T1W_MASK}',
    'skull_strip_tool': 'synthstrip' if os.path.exists('/usr/local/bin/mri_synthstrip') else 'bet',
    'n4_corrected':    True,
})

with open(cap_path, 'w') as f:
    json.dump(d, f, indent=2)
print('capability.json updated with t1w_prep paths')
PYEOF

_log OK "Stage 02 complete"
_log OK "  N4 T1w:      ${T1W_N4}"
_log OK "  Brain:       ${T1W_BRAIN}"
_log OK "  Mask:        ${T1W_MASK}"

log_stage_end "03_t1w_prep" "$SUB"
