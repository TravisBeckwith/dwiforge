#!/usr/bin/env bash
# stages/05_tensor_fitting.sh — Tensor and diffusion model fitting
# =============================================================================
# Fits diffusion models to the preprocessed DWI using tmi (DESIGNER-v2).
#
# Models run depend on acquisition capability (read from capability.json):
#
#   DTI   — always run; appropriate for all single-shell data
#   DKI   — optional (off by default); requires b>=2000 for reliable estimates.
#           CLS data (b=1000) does NOT meet this threshold. Enable with:
#           DWIFORGE_FIT_DKI=true (produces high-variance estimates, use with caution)
#   WDKI  — same constraint as DKI
#   WMTI  — requires multi-shell; skipped for single-shell data
#   SMI   — requires multi-shell + b-tensor encoding; skipped for single-shell
#
# White matter mask:
#   FSL FAST is run on the T1w to produce a WM probability map, which is
#   registered to diffusion space and used to constrain tensor fitting.
#   Falls back to the brain mask from DESIGNER if T1w is unavailable.
#
# Outputs (in DIR_WORK/<sub>/tmi/):
#   fa_dti.nii, md_dti.nii, ad_dti.nii, rd_dti.nii  — DTI scalar metrics (tmi naming)
#   eigenvectors.nii                       — principal diffusion direction
#   eigenvalues.nii                        — sorted eigenvalues
#   [mk.nii, ak.nii, rk.nii]              — DKI metrics (if enabled)
#
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
log_stage_start "06_tensor_fitting" "$SUB"

# ---------------------------------------------------------------------------
# Read capability profile — decide which models to run
# ---------------------------------------------------------------------------

MIF_FINAL="${WORK}/dwi_preprocessed.mif"

if [[ ! -f "$MIF_FINAL" ]]; then
    _log ERROR "Preprocessed DWI not found: ${MIF_FINAL}"
    _log ERROR "Run stages 02–05 before stage 06"
    exit 1
fi

# Read shell structure and counts
N_DWI=0
N_SHELLS=1
B0_THRESHOLD="${DWIFORGE_B0_THRESHOLD:-50}"
if [[ -f "$CAPABILITY_JSON" ]]; then
    N_DWI=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('n_dwi', 0) or 0)
" 2>/dev/null || echo "0"
    )
    N_SHELLS=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('shells', {}).get('count', 1))
" 2>/dev/null || echo "1"
    )
    IS_SINGLE=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(str(d.get('shells', {}).get('is_single_shell', True)).lower())
" 2>/dev/null || echo "true"
    )
fi

_log INFO "DWI directions: ${N_DWI}"
_log INFO "Shells:         ${N_SHELLS}"

# Model selection
RUN_DTI=true
RUN_DKI="${DWIFORGE_FIT_DKI:-false}"
RUN_WDKI="${DWIFORGE_FIT_WDKI:-false}"
RUN_SMI=false

# Warn and override if DKI is requested but b-value is insufficient
if [[ "$RUN_DKI" == "true" || "$RUN_WDKI" == "true" ]]; then
    if [[ "${IS_SINGLE:-true}" == "true" ]]; then
        _log WARN "DKI/WDKI requested but data is single-shell (b=1000)"
        _log WARN "Kurtosis estimates at b=1000 have high variance and are"
        _log WARN "unreliable. Proceeding anyway (DWIFORGE_FIT_DKI=true)."
        _log WARN "Results should be interpreted with caution."
    fi
fi

# SMI always skipped for single-shell
if [[ "${IS_SINGLE:-true}" == "true" ]]; then
    RUN_SMI=false
    _log INFO "SMI skipped (requires multi-shell + b-tensor acquisition)"
fi

_log INFO "Models: DTI=yes DKI=${RUN_DKI} WDKI=${RUN_WDKI} SMI=${RUN_SMI}"

# ---------------------------------------------------------------------------
# Locate T1w outputs from stage 03
# ---------------------------------------------------------------------------

T1W_BRAIN=""
T1W_N4=""
if [[ -f "$CAPABILITY_JSON" ]]; then
    T1W_BRAIN=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('t1w_prep',{}).get('t1w_brain','') or '')
" 2>/dev/null || echo ""
    )
    T1W_N4=$(
        "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('t1w_prep',{}).get('t1w_n4','') or '')
" 2>/dev/null || echo ""
    )
fi

T1W_AVAILABLE=false
[[ -n "$T1W_BRAIN" && -f "$T1W_BRAIN" ]] && T1W_AVAILABLE=true

# ---------------------------------------------------------------------------
# Step 1: Extract mean b0 from preprocessed DWI
# ---------------------------------------------------------------------------

B0_MEAN="${WORK}/b0_mean.nii.gz"

if [[ ! -f "$B0_MEAN" ]]; then
    _log INFO "Step 1: Extracting mean b0"
    dwiextract "$MIF_FINAL" - -bzero -quiet | \
        mrmath - mean "$B0_MEAN" -axis 3 -quiet -force
    _log OK "  Mean b0: ${B0_MEAN##*/}"
else
    _log INFO "Step 1: Mean b0 already extracted, skipping"
fi

# ---------------------------------------------------------------------------
# Step 2: Register b0 → T1w (6 DOF — no shearing for diffusion→structural)
# ---------------------------------------------------------------------------
# This transform is needed to map the FAST WM mask back to diffusion space.
# 6 DOF (rigid) is appropriate: susceptibility distortion has been corrected
# in stage 04, so only rigid motion between sessions remains.

B0_TO_T1W_MAT="${WORK}/b0_to_t1w.mat"
T1W_TO_B0_MAT="${WORK}/t1w_to_b0.mat"
WM_MASK_T1W_SPACE="${WORK}/wm_mask_t1w.nii.gz"
WM_MASK="${WORK}/wm_mask_dwi.nii.gz"
BRAIN_MASK=""

# Locate DESIGNER brain mask for fallback
DESIGNER_SCRATCH="${WORK}/designer_scratch"
if [[ -d "$DESIGNER_SCRATCH" ]]; then
    BRAIN_MASK=$(find "$DESIGNER_SCRATCH" -name "brain_mask*" -o -name "*mask*" \
        2>/dev/null | head -1 || true)
fi

if [[ "$T1W_AVAILABLE" == "true" ]]; then

    if [[ ! -f "$B0_TO_T1W_MAT" ]]; then
        _log INFO "Step 2: Registering b0 → T1w (6 DOF rigid)"
        flirt \
            -in  "$B0_MEAN" \
            -ref "$T1W_BRAIN" \
            -out "${WORK}/b0_in_t1w_space.nii.gz" \
            -omat "$B0_TO_T1W_MAT" \
            -dof 6 \
            -cost normmi
        convert_xfm -omat "$T1W_TO_B0_MAT" -inverse "$B0_TO_T1W_MAT"
        _log OK "  Registration complete: ${B0_TO_T1W_MAT##*/}"
    else
        _log INFO "Step 2: b0→T1w registration already done, skipping"
        [[ ! -f "$T1W_TO_B0_MAT" ]] && \
            convert_xfm -omat "$T1W_TO_B0_MAT" -inverse "$B0_TO_T1W_MAT"
    fi

    # -------------------------------------------------------------------------
    # Step 3: FAST tissue segmentation on T1w
    # -------------------------------------------------------------------------

    FAST_WM_PVE="${WORK}/fast/t1w_brain_pve_2.nii.gz"

    if [[ ! -f "$FAST_WM_PVE" ]]; then
        _log INFO "Step 3: FAST tissue segmentation"
        mkdir -p "${WORK}/fast"
        fast \
            -t 1 \
            -n 3 \
            -H 0.1 \
            -I 4 \
            -l 20.0 \
            -o "${WORK}/fast/t1w_brain" \
            "$T1W_BRAIN"
        _log OK "  FAST complete — WM PVE: ${FAST_WM_PVE##*/}"
    else
        _log INFO "Step 3: FAST already done, skipping"
    fi

    # -------------------------------------------------------------------------
    # Step 4: WM mask → diffusion space
    # -------------------------------------------------------------------------
    # Threshold WM PVE at 0.5 in T1w space, then transform to DWI space.

    if [[ ! -f "$WM_MASK" ]]; then
        _log INFO "Step 4: Transforming WM mask to diffusion space"

        # Threshold WM probability
        fslmaths "$FAST_WM_PVE" -thr 0.5 -bin "$WM_MASK_T1W_SPACE"

        # Transform binary mask to DWI space
        flirt \
            -in  "$WM_MASK_T1W_SPACE" \
            -ref "$B0_MEAN" \
            -out "$WM_MASK" \
            -init "$T1W_TO_B0_MAT" \
            -applyxfm \
            -interp nearestneighbour

        # If DESIGNER brain mask exists, intersect WM mask with it
        if [[ -n "$BRAIN_MASK" && -f "$BRAIN_MASK" ]]; then
            fslmaths "$WM_MASK" -mas "$BRAIN_MASK" "$WM_MASK"
        fi

        _log OK "  WM mask in DWI space: ${WM_MASK##*/}"
    else
        _log INFO "Step 4: WM mask already exists, skipping"
    fi

else
    _log WARN "Step 2-4: No T1w available — skipping FAST, using brain mask only"

    # Use DESIGNER brain mask directly
    if [[ -n "$BRAIN_MASK" && -f "$BRAIN_MASK" ]]; then
        WM_MASK="$BRAIN_MASK"
        _log WARN "  Using DESIGNER brain mask as fitting mask"
    else
        _log WARN "  No mask found — fitting over full image (slower, noisier)"
        WM_MASK=""
    fi
fi

# ---------------------------------------------------------------------------
# Step 5: Run tmi — tensor and diffusion model fitting
# ---------------------------------------------------------------------------

TMI_OUT_DIR="${WORK}/tmi"
FA_CHECK="${TMI_OUT_DIR}/fa_dti.nii"

if [[ ! -f "$FA_CHECK" ]]; then
    _log INFO "Step 5: Running tmi model fitting"
    mkdir -p "$TMI_OUT_DIR"

    # Build tmi command — same PYTHONPATH isolation as DESIGNER
    # tmi requires:
    #   1. designer2 Python package  — in neuroimaging_env
    #   2. mrtrix3 Python bindings   — in MRTRIX_PYTHON_PATH
    # The tmi script shebang may point to any Python. We must call it
    # explicitly with the neuroimaging_env Python so designer2 is importable,
    # while also setting PYTHONPATH for mrtrix3.
    TMI_BIN="${HOME}/.local/bin/tmi"

    # Use DESIGNER_BIN's Python (same venv as designer2)
    # DESIGNER_BIN is already resolved and validated by env_setup.sh
    TMI_PYTHON="$(dirname "${DESIGNER_BIN}")/python3"
    if [[ ! -x "$TMI_PYTHON" ]]; then
        # Fallback: derive from VIRTUAL_ENV or well-known path
        TMI_PYTHON="${VIRTUAL_ENV:-${HOME}/neuroimaging_env}/bin/python3"
    fi
    _log INFO "  tmi Python: ${TMI_PYTHON}"

    TMI_CMD=(
        env
        "PYTHONPATH=${MRTRIX_PYTHON_PATH}"
        "$TMI_PYTHON"
        "$TMI_BIN"
        -DTI
        -n_cores "${OMP_NUM_THREADS:-4}"
        -nthreads "${OMP_NUM_THREADS:-4}"
        -force
    )

    # Add mask if available
    if [[ -n "$WM_MASK" && -f "$WM_MASK" ]]; then
        TMI_CMD+=(-mask "$WM_MASK")
        _log INFO "  Mask: ${WM_MASK##*/}"
    fi

    # Optional models
    [[ "$RUN_DKI"  == "true" ]] && TMI_CMD+=(-DKI -akc_outliers)
    [[ "$RUN_WDKI" == "true" ]] && TMI_CMD+=(-WDKI)

    # Sigma map for models that use it (DKI outlier correction benefits from it)
    SIGMA_MAP="${WORK}/dwi/dwi_sigma.nii"
    if [[ -f "$SIGMA_MAP" && ("$RUN_SMI" == "true") ]]; then
        TMI_CMD+=(-sigma "$SIGMA_MAP")
    fi

    # Input and output directory
    TMI_CMD+=("$MIF_FINAL" "$TMI_OUT_DIR")

    _log INFO "  Running tmi..."
    "${TMI_CMD[@]}"

    if [[ ! -f "$FA_CHECK" ]]; then
        _log ERROR "tmi did not produce FA output: ${FA_CHECK}"
        exit 1
    fi

    _log OK "  tmi complete: ${TMI_OUT_DIR##*/}/"
else
    _log INFO "Step 5: tmi output already exists, skipping"
fi

# ---------------------------------------------------------------------------
# Verify key outputs exist
# ---------------------------------------------------------------------------

EXPECTED_METRICS=("fa" "md" "ad" "rd" "eigenvalues" "eigenvectors")
MISSING=()
for m in "${EXPECTED_METRICS[@]}"; do
    [[ ! -f "${TMI_OUT_DIR}/${m}.nii" ]] && MISSING+=("$m")
done

if [[ "${#MISSING[@]}" -gt 0 ]]; then
    _log WARN "Some expected tmi outputs missing: ${MISSING[*]}"
    _log WARN "Check tmi logs for errors"
fi

# ---------------------------------------------------------------------------
# Update capability profile
# ---------------------------------------------------------------------------

"${PYTHON_EXECUTABLE:-python3}" - <<PYEOF
import json, os, glob

cap_path = '${CAPABILITY_JSON}'
with open(cap_path) as f:
    d = json.load(f)

tmi_dir = '${TMI_OUT_DIR}'
metrics = {}
for nii in glob.glob(os.path.join(tmi_dir, '*.nii')):
    name = os.path.basename(nii).replace('.nii', '')
    metrics[name] = nii

d['tensor_fitting'] = {
    'status':        'complete',
    'output_dir':    tmi_dir,
    'models_run':    ['DTI'] + (['DKI'] if '${RUN_DKI}' == 'true' else []),
    'mask_type':     'wm_fast' if os.path.exists('${WM_MASK_T1W_SPACE}') else 'brain_mask',
    'mask_path':     '${WM_MASK}',
    'metrics':       metrics,
    'dki_warning':   '${IS_SINGLE}' == 'true' and '${RUN_DKI}' == 'true',
}

with open(cap_path, 'w') as f:
    json.dump(d, f, indent=2)
print(f"capability.json updated — {len(metrics)} metric files recorded")
PYEOF

_log OK "Stage 05 complete"
_log OK "  Output dir: ${TMI_OUT_DIR}"
_log OK "  FA:         ${FA_CHECK}"

log_stage_end "06_tensor_fitting" "$SUB"
