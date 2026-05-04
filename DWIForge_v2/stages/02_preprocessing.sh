#!/usr/bin/env bash
# stages/01_preprocessing.sh — DWI Preprocessing Stage
#
# Called by dwiforge.sh orchestrator:
#   bash stages/01_preprocessing.sh <subject_id>
#
# Reads all configuration from DWIFORGE_* environment variables.
# Writes outputs to DWIFORGE_DIR_WORK/<sub>/dwi/
# Writes final output: dwi_preprocessed.mif (ready for registration stage)
#
# Steps (in order):
#   1. Input validation and BIDS discovery
#   2. Initial import to MRtrix .mif (embeds gradient table)
#   3. Denoising          ← DIPY Patch2Self v3/v1/MP-PCA fallback
#   4. Gibbs ringing correction
#   5. Susceptibility distortion correction (topup / Synb0-DisCo)
#   6. Eddy current and motion correction
#   7. Bias field correction
#   8. Brain mask generation
#   9. Final export

set -euo pipefail

STAGE_NAME="preprocessing"
SUB="${1:?Usage: $0 <subject_id>}"

# ---------------------------------------------------------------------------
# Source libraries (path relative to this script's location)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DWIFORGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${DWIFORGE_ROOT}/lib/logging.sh"
source "${DWIFORGE_ROOT}/lib/utils.sh"

# ---------------------------------------------------------------------------
# Convenience: per-subject paths
# ---------------------------------------------------------------------------

WORK="${DWIFORGE_DIR_WORK}/${SUB}/dwi"
TMP="${DWIFORGE_DIR_WORK}/${SUB}/tmp"
SRC="${DWIFORGE_DIR_SOURCE}/${SUB}/dwi"
LOG_SUB="${DWIFORGE_DIR_LOGS}/${SUB}"

# Per-stage log (in addition to the per-subject log set by the orchestrator)
STAGE_LOG="${LOG_SUB}/${STAGE_NAME}.log"
mkdir -p "${LOG_SUB}"

_log() { log_sub "$1" "$SUB" "${*:2}"; }

log_stage_start "$STAGE_NAME" "$SUB"

# ---------------------------------------------------------------------------
# Step 1: Input validation and BIDS discovery
# ---------------------------------------------------------------------------

_log INFO "Step 1: Validating inputs"

# Locate DWI NIfTI (prefer AP-encoded if both exist)
DWI_NII=""
for candidate in \
    "${SRC}"/*_dir-AP_dwi.nii.gz \
    "${SRC}"/*_dwi.nii.gz \
    "${SRC}"/*.nii.gz; do
    if [[ -f "$candidate" ]]; then
        DWI_NII="$candidate"
        break
    fi
done

if [[ -z "$DWI_NII" ]]; then
    _log ERROR "No DWI NIfTI found under ${SRC}"
    exit 1
fi

# Locate bvec/bval (must exist alongside the NIfTI)
BVAL="${DWI_NII%.nii.gz}.bval"
BVEC="${DWI_NII%.nii.gz}.bvec"

# BIDS fallback: separate gradient files without direction encoding in name
if [[ ! -f "$BVAL" ]]; then
    BVAL="$(find "$SRC" -maxdepth 1 -name '*.bval' | head -1)"
fi
if [[ ! -f "$BVEC" ]]; then
    BVEC="$(find "$SRC" -maxdepth 1 -name '*.bvec' | head -1)"
fi

for f in "$DWI_NII" "$BVAL" "$BVEC"; do
    if [[ ! -f "$f" ]]; then
        _log ERROR "Required file not found: ${f}"
        exit 1
    fi
done

# Check for reverse phase-encode b0 (enables topup)
RPE_NII=""
for candidate in \
    "${SRC}"/*_dir-PA_dwi.nii.gz \
    "${SRC}"/*_dir-PA_epi.nii.gz \
    "${SRC}"/*_acq-PA_dwi.nii.gz; do
    if [[ -f "$candidate" ]]; then
        RPE_NII="$candidate"
        break
    fi
done

N_VOLS=$(mrinfo "$DWI_NII" -size 2>/dev/null | awk '{print $NF}' || echo "?")
_log INFO "  DWI:      ${DWI_NII##*/} (${N_VOLS} volumes)"
_log INFO "  bval:     ${BVAL##*/}"
_log INFO "  bvec:     ${BVEC##*/}"
_log INFO "  Reverse PE: ${RPE_NII:+${RPE_NII##*/}}${RPE_NII:-not found}"

# Locate JSON sidecar (needed for mrconvert -json_import in step 2)
JSON_SIDECAR=""
_stem="${DWI_NII%.nii.gz}"
_stem="${_stem##*/}"
for _j in \
    "${SRC}/${_stem}.json" \
    "${SRC}/${SUB}_dwi.json"; do
    if [[ -f "$_j" ]]; then
        JSON_SIDECAR="$_j"
        break
    fi
done
_log INFO "  JSON sidecar: ${JSON_SIDECAR:+${JSON_SIDECAR##*/}}${JSON_SIDECAR:-not found}"

# ---------------------------------------------------------------------------
# Step 2: Import to MRtrix .mif (embeds gradient table)
# ---------------------------------------------------------------------------

MIF_RAW="${WORK}/dwi_raw.mif"

if [[ ! -f "$MIF_RAW" ]]; then
    _log INFO "Step 2: Importing to MRtrix .mif"

    # Build mrconvert command.
    # -json_import embeds phase encoding metadata (TotalReadoutTime,
    # PhaseEncodingAxis, etc.) required by topup and DESIGNER.
    # Without it the .mif has no PE scheme and topup cannot run.
    MRCONVERT_CMD=(
        mrconvert
        "$DWI_NII"
        "$MIF_RAW"
        -fslgrad "$BVEC" "$BVAL"
        -quiet
        -force
    )

    # Embed JSON sidecar if present.
    # For Philips data: patch PartialFourier into a temp sidecar so DESIGNER
    # can read the PF factor from the .mif header automatically.
    if [[ -n "$JSON_SIDECAR" && -f "$JSON_SIDECAR" ]]; then
        # Check if PercentSampling == 100 (fully sampled despite PF flag)
        PATCHED_JSON="${TMP}/dwi_patched.json"
        "${PYTHON_EXECUTABLE:-python3}" - << PYEOF2
import json, sys
with open('${JSON_SIDECAR}') as f:
    d = json.load(f)
pct = d.get('PercentSampling', 0)
# Fully sampled: inject PartialFourier=1.0 so DESIGNER skips PF correction
if pct >= 99:
    d['PartialFourier'] = 1.0
    d['PartialFourierDirection'] = 'PHASE'
# Ensure phase encoding fields use BIDS standard names
if 'PhaseEncodingAxis' in d and 'PhaseEncodingDirection' not in d:
    d['PhaseEncodingDirection'] = d['PhaseEncodingAxis']
if 'EstimatedTotalReadoutTime' in d and 'TotalReadoutTime' not in d:
    d['TotalReadoutTime'] = d['EstimatedTotalReadoutTime']
if 'EstimatedEffectiveEchoSpacing' in d and 'EffectiveEchoSpacing' not in d:
    d['EffectiveEchoSpacing'] = d['EstimatedEffectiveEchoSpacing']
with open('${PATCHED_JSON}', 'w') as f:
    json.dump(d, f, indent=2)
print('JSON patched: PartialFourier=' + str(d.get('PartialFourier', 'not set')))
PYEOF2
        MRCONVERT_CMD+=(-json_import "$PATCHED_JSON")
        _log INFO "  Embedding patched JSON sidecar (PartialFourier injected)"
    else
        _log WARN "  No JSON sidecar — phase encoding metadata will be absent from .mif"
        _log WARN "  topup will not be available for this subject"
    fi

    "${MRCONVERT_CMD[@]}"
    _log OK "  Imported: ${MIF_RAW##*/}"
else
    _log INFO "Step 2: .mif already exists, skipping import"
fi

# ---------------------------------------------------------------------------
# Step 3: Denoising
# ---------------------------------------------------------------------------
#
# Uses python/denoise.py which implements the priority chain:
#   DIPY Patch2Self v3 (>=1.10)  →  P2S v1  →  standalone P2S2  →  MP-PCA
#
# Inputs/outputs are NIfTI because DIPY works natively with NIfTI.
# We convert back to .mif after denoising.

MIF_DENOISED="${WORK}/dwi_denoised.mif"
NII_DENOISED="${WORK}/dwi_denoised.nii.gz"
DENOISE_SIDECAR="${LOG_SUB}/denoise_qc.json"
DENOISE_SCRIPT="${DWIFORGE_ROOT}/python/denoise.py"

if [[ ! -f "$MIF_DENOISED" ]]; then
    _log INFO "Step 3: Denoising"

    # Select method from config; default to auto
    DENOISE_METHOD="${DWIFORGE_DENOISE_METHOD:-auto}"

    # Point P2S temp dir at our subject's tmp/ to keep work contained
    DENOISE_TMP="${TMP}/patch2self"
    mkdir -p "$DENOISE_TMP"

    "${PYTHON_EXECUTABLE:-python3}" "$DENOISE_SCRIPT" \
        "$DWI_NII" \
        "$BVAL" \
        "$NII_DENOISED" \
        "$SUB" \
        --method    "$DENOISE_METHOD" \
        --model     "${DWIFORGE_DENOISE_MODEL:-ols}" \
        --b0-threshold "${DWIFORGE_B0_THRESHOLD:-50}" \
        --tmp-dir   "$DENOISE_TMP" \
        --sidecar   "$DENOISE_SIDECAR" \
        --nthreads  "${OMP_NUM_THREADS:-1}"

    # Convert denoised NIfTI back to .mif preserving gradient table
    mrconvert "$NII_DENOISED" "$MIF_DENOISED" \
        -fslgrad "$BVEC" "$BVAL" \
        -quiet -force

    # Log which method was actually used
    if [[ -f "$DENOISE_SIDECAR" ]]; then
        METHOD_USED=$(python3 -c \
            "import json,sys; d=json.load(open('${DENOISE_SIDECAR}')); \
             print(d.get('method','unknown'))" 2>/dev/null || echo "unknown")
        _log OK "  Denoising complete: ${METHOD_USED}"
        _log OK "  SNR estimate: $(python3 -c \
            "import json; d=json.load(open('${DENOISE_SIDECAR}')); \
             print(d.get('snr_estimate','N/A'))" 2>/dev/null || echo 'N/A')"
    fi

    # Clean up NIfTI intermediate if denoised .mif was created successfully
    if [[ -f "$MIF_DENOISED" && "${DWIFORGE_CLEANUP_TIER:-0}" -ge 1 ]]; then
        rm -f "$NII_DENOISED" "${DENOISE_TMP}"/*
    fi
else
    _log INFO "Step 3: Denoising already complete, skipping"
fi

# ---------------------------------------------------------------------------
# Step 3b: Noise map for Rician bias correction
# ---------------------------------------------------------------------------
# Run dwidenoise -noise on the RAW data (before P2S) to get a sigma map.
# This is NOT the denoising step — the actual output is discarded.
# The sigma map is passed to DESIGNER's -noisemap for Rician correction.
# MP-PCA (dwidenoise) provides a good noise floor estimate even when P2S
# is used for the actual denoising.

SIGMA_MAP="${WORK}/dwi_sigma.nii"

if [[ ! -f "$SIGMA_MAP" ]]; then
    _log INFO "Step 3b: Generating sigma map for Rician correction"
    dwidenoise "$MIF_RAW" /dev/null \
        -noise "$SIGMA_MAP" \
        -nthreads "${OMP_NUM_THREADS:-1}" \
        -quiet -force 2>/dev/null && \
        _log OK "  Sigma map: ${SIGMA_MAP##*/}" || {
        _log WARN "  dwidenoise sigma map failed — Rician correction will be skipped"
        SIGMA_MAP=""
    }
else
    _log INFO "Step 3b: Sigma map already exists, skipping"
fi

# ---------------------------------------------------------------------------
# Step 4: Gibbs ringing correction
# ---------------------------------------------------------------------------
# Method is determined by qc_bids.py and recorded in capability.json:
#   mrdegibbs      — standard 2D slice-wise (fully sampled or PF 6/8, 7/8)
#   mrdegibbs_3d   — 3D volume-wise extension (-mode 3d)
# PF = 5/8: mrdegibbs is applied with a warning in the capability profile;
# no reliable correction exists for PF=5/8 from reconstructed magnitude data.

MIF_DEGIBBS="${WORK}/dwi_degibbs.mif"

if [[ ! -f "$MIF_DEGIBBS" ]]; then
    _log INFO "Step 4: Gibbs ringing correction"

    # Read recommended method from capability profile written by stage 00
    CAPABILITY_JSON="${DWIFORGE_DIR_LOGS}/${SUB}/capability.json"
    GIBBS_METHOD="mrdegibbs"   # safe default if profile absent
    PF_FRACTION="unknown"
    ACQ_3D="false"

    if [[ -f "$CAPABILITY_JSON" ]]; then
        GIBBS_METHOD=$(
            "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('acquisition', {}).get('recommended_gibbs_method', 'mrdegibbs'))
" 2>/dev/null || echo "mrdegibbs"
        )
        PF_FRACTION=$(
            "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('acquisition', {}).get('partial_fourier_fraction', 'unknown'))
" 2>/dev/null || echo "unknown"
        )
    fi

    _log INFO "  Partial Fourier: ${PF_FRACTION}"
    _log INFO "  Method:          ${GIBBS_METHOD}"

    # Warn if PF=5/8 — no reliable correction available from magnitude NIfTI
    if [[ "$PF_FRACTION" == "5/8" ]]; then
        _log WARN "  PF=5/8: mrdegibbs applied but ringing may not be fully corrected"
        _log WARN "  Consider PF=6/8 or 7/8 in future acquisitions"
    fi

    # Build mrdegibbs command
    DEGIBBS_CMD=(
        mrdegibbs
        "$MIF_DENOISED"
        "$MIF_DEGIBBS"
        -nthreads "${OMP_NUM_THREADS:-1}"
        -quiet
        -force
    )

    # 3D mode for volumetrically encoded acquisitions
    if [[ "$GIBBS_METHOD" == "mrdegibbs_3d" ]]; then
        DEGIBBS_CMD+=(-mode 3d)
        _log INFO "  Using 3D volume-wise mode"
    fi

    # IMPORTANT: mrdegibbs must run BEFORE any interpolation or motion correction.
    # This step runs on the denoised .mif immediately after dwidenoise.
    "${DEGIBBS_CMD[@]}"

    # Verify output
    if ! output_sanity_check "$SUB" "$MIF_DEGIBBS" 5; then
        _log ERROR "  mrdegibbs output missing or too small"
        exit 1
    fi

    _log OK "  Gibbs correction complete (${GIBBS_METHOD})"
else
    _log INFO "Step 4: Gibbs correction already complete, skipping"
fi

# ---------------------------------------------------------------------------
# Stage 01 complete — handoff to downstream stages
# ---------------------------------------------------------------------------
# dwi_degibbs.mif and dwi_sigma.nii are the outputs consumed by later stages:
#   Stage 02: t1w_prep.sh  (skull strip + N4 the T1w)
#   Stage 03: epi_correction.sh  (Synb0 / last resort / eddy-only)
#   Stage 04: designer.sh  (eddy + Rician + mask via DESIGNER with -rpe_pair)

if ! output_sanity_check "$SUB" "$MIF_DEGIBBS" 5; then
    _log ERROR "Stage 01 output missing or too small: ${MIF_DEGIBBS}"
    exit 1
fi

_log OK "Stage 01 complete"
_log OK "  Degibbs output:  ${MIF_DEGIBBS}"
_log OK "  Sigma map:       ${SIGMA_MAP:-not generated}"
