#!/usr/bin/env bash
# stages/12_gnc.sh — Gradient nonlinearity correction (DWI)
# =============================================================================
# Corrects spatial geometric distortion caused by gradient coil nonlinearity,
# using the HCP Pipelines' gradunwarp tool (gradient_unwarp.py) and a
# vendor-supplied gradient coefficient file (e.g. Siemens coeff.grad).
#
# This matters most on high-gradient-strength systems (e.g. Connectome-class
# scanners), where nonlinearity distortion near the edges of the FOV can be
# substantial. On a standard clinical gradient set it is a smaller effect.
#
# SCOPE / LIMITATION — read before trusting downstream metrics:
#   This stage corrects the *spatial* geometry of the DWI volume (voxel
#   positions), the same thing FreeSurfer's own GNC step applies to
#   anatomicals. It does NOT recompute a per-voxel b-matrix. True
#   gradient-nonlinearity-aware diffusion modeling (used by e.g. HCP's own
#   pipelines for NODDI-grade precision) adjusts the effective b-value and
#   b-vector at every voxel, since the actual diffusion gradient strength
#   and direction also vary spatially, not just position. That correction
#   is NOT implemented here. For most tractography and DTI use cases the
#   spatial correction below is the dominant term and is worth having; if
#   you need voxel-wise b-matrix correction for precise microstructure
#   modeling, treat this stage as a partial fix, not a complete one.
#
# Runs only if DWIFORGE_RUN_GNC=true AND a coefficient file is configured.
# Degrades gracefully (skip, not fail) if either is missing — this is an
# enhancement stage, not a required one. See dwiforge.toml [options]:
#
#   [options]
#   run_gnc        = true
#   gnc_coeff_file = "/path/to/coeff.grad"   # vendor-supplied, not redistributable
#
# Requires: gradient_unwarp.py (pip install
#   git+https://github.com/Washington-University/gradunwarp.git), FSL
#   (convertwarp, applywarp).
#
# Input:  ${WORK}/dwi_preprocessed.mif        (stage 05 output)
# Output: ${WORK}/dwi_preprocessed.mif        (overwritten, GNC-corrected)
#         ${WORK}/dwi_preprocessed_pre_gnc.mif (pre-GNC backup, kept for QC)
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
log_stage_start "12_gnc" "$SUB"

MIF_FINAL="${WORK}/dwi_preprocessed.mif"
MIF_PRE_GNC="${WORK}/dwi_preprocessed_pre_gnc.mif"

if [[ ! -f "$MIF_FINAL" ]]; then
    _log ERROR "Stage 05 output not found: ${MIF_FINAL}"
    _log ERROR "Run stage 05 (designer) before stage 12 (gnc)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Gate: only run if explicitly enabled AND a coefficient file is configured
# ---------------------------------------------------------------------------

RUN_GNC="${DWIFORGE_RUN_GNC:-false}"
COEFF_FILE="${DWIFORGE_GNC_COEFF_FILE:-}"

if [[ "$RUN_GNC" != "true" ]]; then
    _log INFO "GNC disabled (DWIFORGE_RUN_GNC=false) — skipping"
    log_stage_end "12_gnc" "$SUB"
    exit 0
fi

if [[ -z "$COEFF_FILE" || ! -f "$COEFF_FILE" ]]; then
    _log WARN "run_gnc is true but no valid gnc_coeff_file is configured"
    _log WARN "  (looked for: '${COEFF_FILE:-<unset>}')"
    _log WARN "Gradient coefficient files are vendor-supplied and not"
    _log WARN "redistributable — obtain yours from your site physicist"
    _log WARN "or scanner vendor. Skipping GNC for this subject."
    log_stage_end "12_gnc" "$SUB"
    exit 0
fi

if ! command -v gradient_unwarp.py >/dev/null 2>&1; then
    _log WARN "gradient_unwarp.py not found on PATH — skipping GNC"
    _log WARN "  Install: pip install git+https://github.com/Washington-University/gradunwarp.git"
    log_stage_end "12_gnc" "$SUB"
    exit 0
fi

# Already corrected? (checkpoint-style idempotency, matches other stages)
if [[ -f "$MIF_PRE_GNC" ]]; then
    _log INFO "GNC already applied (${MIF_PRE_GNC##*/} exists) — skipping"
    log_stage_end "12_gnc" "$SUB"
    exit 0
fi

_log INFO "Applying gradient nonlinearity correction"
_log INFO "  Coefficient file: ${COEFF_FILE}"

GNC_DIR="${WORK}/gnc"
mkdir -p "$GNC_DIR"

# ---------------------------------------------------------------------------
# Step 1: Export to NIfTI + bvec/bval (gradunwarp and FSL operate on NIfTI)
# ---------------------------------------------------------------------------

DWI_NII="${GNC_DIR}/dwi_pre_gnc.nii.gz"
DWI_BVEC="${GNC_DIR}/dwi_pre_gnc.bvec"
DWI_BVAL="${GNC_DIR}/dwi_pre_gnc.bval"

_log INFO "  Step 1: Exporting to NIfTI"
mrconvert "$MIF_FINAL" "$DWI_NII" \
    -export_grad_fsl "$DWI_BVEC" "$DWI_BVAL" \
    -quiet -force

# gradient_unwarp.py works on a single 3D volume — use the mean b0 to
# compute the warp field, then apply that field to every volume. The
# nonlinearity field is a static property of the scanner, not per-volume.
B0_MEAN="${GNC_DIR}/b0_mean.nii.gz"
dwiextract "$MIF_FINAL" - -bzero -quiet | mrmath - mean "$B0_MEAN" -axis 3 -quiet -force

# ---------------------------------------------------------------------------
# Step 2: Compute the nonlinearity warp field from the mean b0
# ---------------------------------------------------------------------------

_log INFO "  Step 2: Computing gradient nonlinearity warp field"
GNC_OUT="${GNC_DIR}/b0_unwarped.nii.gz"

( cd "$GNC_DIR" && gradient_unwarp.py \
    "$B0_MEAN" "$GNC_OUT" siemens \
    -g "$COEFF_FILE" \
    --fovmin -1 --fovmax 1 --numpoints 60 \
    --interp_order 3 \
    --verbose ) > "${GNC_DIR}/gradient_unwarp.log" 2>&1

FULLWARP="${GNC_DIR}/fullWarp_abs.nii.gz"
if [[ ! -f "$FULLWARP" ]]; then
    _log ERROR "gradient_unwarp.py did not produce fullWarp_abs.nii.gz"
    _log ERROR "Check: ${GNC_DIR}/gradient_unwarp.log"
    _log ERROR "Continuing pipeline WITHOUT gradient nonlinearity correction"
    log_stage_end "12_gnc" "$SUB"
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 3: Apply the warp field to every DWI volume
# ---------------------------------------------------------------------------

_log INFO "  Step 3: Applying warp field to all ${SUB} DWI volumes"
DWI_GNC="${GNC_DIR}/dwi_gnc.nii.gz"

applywarp \
    --in="$DWI_NII" \
    --ref="$DWI_NII" \
    --warp="$FULLWARP" \
    --out="$DWI_GNC" \
    --interp=spline

if ! output_sanity_check "$SUB" "$DWI_GNC" 5; then
    _log ERROR "GNC output too small or missing: ${DWI_GNC}"
    _log ERROR "Continuing pipeline WITHOUT gradient nonlinearity correction"
    log_stage_end "12_gnc" "$SUB"
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 4: Reassemble .mif with the original gradient table, replace stage 05
# output in place. bvecs/bvals are NOT modified — GNC is a spatial warp of
# voxel positions, not a change in gradient direction (see SCOPE note above
# re: b-matrix correction, which is a separate, unimplemented refinement).
# ---------------------------------------------------------------------------

_log INFO "  Step 4: Reassembling corrected .mif"
cp "$MIF_FINAL" "$MIF_PRE_GNC"

mrconvert "$DWI_GNC" "$MIF_FINAL" \
    -fslgrad "$DWI_BVEC" "$DWI_BVAL" \
    -quiet -force

if ! output_sanity_check "$SUB" "$MIF_FINAL" 5; then
    _log ERROR "Final GNC-corrected output too small or missing: ${MIF_FINAL}"
    _log ERROR "Restoring pre-GNC volume"
    cp "$MIF_PRE_GNC" "$MIF_FINAL"
    log_stage_end "12_gnc" "$SUB"
    exit 0
fi

_log OK "  GNC complete — ${MIF_FINAL##*/} replaced with corrected volume"
_log OK "  Pre-GNC backup kept at: ${MIF_PRE_GNC##*/}"

# ---------------------------------------------------------------------------
# Update capability profile
# ---------------------------------------------------------------------------

"${PYTHON_EXECUTABLE:-python3}" - <<PYEOF
import json

cap_path = '${CAPABILITY_JSON}'
with open(cap_path) as f:
    d = json.load(f)

d['gnc'] = {
    'status':               'applied',
    'coeff_file':           '${COEFF_FILE}',
    'method':               'gradunwarp (gradient_unwarp.py)',
    'bmatrix_corrected':    False,
    'pre_gnc_backup':       '${MIF_PRE_GNC}',
}

with open(cap_path, 'w') as f:
    json.dump(d, f, indent=2)
print('capability.json updated: gnc status=applied')
PYEOF

_log OK "Stage 12 complete"

log_stage_end "12_gnc" "$SUB"
