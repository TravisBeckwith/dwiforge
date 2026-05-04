#!/usr/bin/env bash
# stages/09_tractography.sh — FOD estimation, tractography, and connectome
# =============================================================================
# Runs the full structural connectivity pipeline using SS3T-CSD FODs,
# ACT-guided probabilistic tractography, SIFT2 streamline weighting,
# and FreeSurfer Desikan-Killiany parcellation for connectome construction.
#
# Prerequisites (all subjects must be complete before this stage runs):
#   Stage 01: recon_all        — FreeSurfer parcellation + surfaces
#   Stage 05: designer         — dwi_preprocessed.mif + brain mask
#   Stage 06: tensor_fitting   — FA map (for mean-FA connectome)
#   Stage 08: response_functions — per-subject responses estimated
#   Orchestrator barrier       — responsemean.done (group responses averaged)
#
# Steps:
#   1.  Check responsemean.done sentinel
#   2.  dwi2fod msmt_csd       — SS3T-CSD FOD estimation (group responses)
#   3.  mtnormalise            — multi-tissue intensity normalisation
#   4.  5ttgen hsvs            — 5-tissue-type image from FreeSurfer
#   5.  5tt2gmwmi              — GM/WM interface seeding image
#   6.  tckgen -act -seed_gmwmi — 10M streamline probabilistic tractography
#   7.  tcksift2               — streamline weight estimation
#   8.  labelconvert           — FreeSurfer parcellation → connectome labels
#   9.  tck2connectome (count) — SIFT2-weighted streamline count matrix
#   10. tcksample + tck2connectome (FA) — mean FA along streamlines matrix
#
# Outputs (in DIR_WORK/<sub>/tractography/):
#   wmfod_norm.mif             — normalised WM FOD (for visualisation/QC)
#   5tt.mif                    — 5-tissue-type image
#   gmwmi.mif                  — GM/WM interface mask
#   tracks_10M.tck             — full tractogram (10M streamlines)
#   sift2_weights.txt          — per-streamline SIFT2 weights
#   parcellation.mif           — DK atlas in diffusion space
#   connectome_count.csv       — SIFT2-weighted streamline count
#   connectome_fa.csv          — mean FA along streamlines
#   connectome_length.csv      — mean streamline length
#
# Connectome note (SIFT2 weights):
#   tck2connectome and tcksample MUST receive -tck_weights_in sift2_weights.txt
#   otherwise the full unweighted tractogram is processed silently.
#   See: https://community.mrtrix.org/t/sift-or-sift2/3494
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
log_stage_start "09_tractography" "$SUB"

# ---------------------------------------------------------------------------
# Step 1: Check group responsemean.done sentinel
# ---------------------------------------------------------------------------

GROUP_DIR="${DWIFORGE_DIR_WORK}/group"
RESPONSEMEAN_DONE="${GROUP_DIR}/responsemean.done"

if [[ ! -f "$RESPONSEMEAN_DONE" ]]; then
    _log ERROR "responsemean.done not found: ${RESPONSEMEAN_DONE}"
    _log ERROR "All subjects must complete stage 08 before stage 09 can run"
    _log ERROR "In SLURM mode: submit responsemean as a dependency job first"
    _log ERROR "In local mode:  set DWIFORGE_LAST_SUBJECT=true on final subject"
    exit 1
fi

_log INFO "Group response functions: confirmed"

# Group-averaged response functions
GROUP_WM="${GROUP_DIR}/group_response_wm.txt"
GROUP_GM="${GROUP_DIR}/group_response_gm.txt"
GROUP_CSF="${GROUP_DIR}/group_response_csf.txt"

for f in "$GROUP_WM" "$GROUP_GM" "$GROUP_CSF"; do
    if [[ ! -s "$f" ]]; then
        _log ERROR "Group response function missing or empty: ${f}"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Read prerequisite paths from capability profile
# ---------------------------------------------------------------------------

MIF_FINAL="${WORK}/dwi_preprocessed.mif"
if [[ ! -f "$MIF_FINAL" ]]; then
    _log ERROR "Preprocessed DWI not found: ${MIF_FINAL}"
    _log ERROR "Run stages 02-05 before stage 09"
    exit 1
fi

# FreeSurfer subject directory
FS_SUBJECT_DIR=$(
    "${PYTHON_EXECUTABLE:-python3}" -c "
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('recon_all', {}).get('subject_dir', '') or '')
" 2>/dev/null || echo ""
)

if [[ -z "$FS_SUBJECT_DIR" || ! -d "$FS_SUBJECT_DIR" ]]; then
    _log ERROR "FreeSurfer subject dir not found in capability profile"
    _log ERROR "Run stage 01 (recon-all) before stage 09"
    exit 1
fi

APARC_ASEG="${FS_SUBJECT_DIR}/mri/aparc+aseg.mgz"
if [[ ! -f "$APARC_ASEG" ]]; then
    _log ERROR "FreeSurfer parcellation not found: ${APARC_ASEG}"
    exit 1
fi

# DESIGNER brain mask
BRAIN_MASK=""
DESIGNER_SCRATCH="${WORK}/designer_scratch"
if [[ -d "$DESIGNER_SCRATCH" ]]; then
    BRAIN_MASK=$(find "$DESIGNER_SCRATCH" \
        -name "brain_mask*" -o -name "*brain*mask*" 2>/dev/null | \
        head -1 || true)
fi

if [[ -z "$BRAIN_MASK" || ! -f "$BRAIN_MASK" ]]; then
    _log ERROR "DESIGNER brain mask not found in ${DESIGNER_SCRATCH}"
    _log ERROR "Run stage 05 (designer) before stage 09"
    exit 1
fi

# FA map from stage 06 (for mean-FA connectome)
FA_MAP="${WORK}/tmi/fa_dti.nii"
if [[ ! -f "$FA_MAP" ]]; then
    _log WARN "FA map not found: ${FA_MAP} — mean-FA connectome will be skipped"
fi

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

TRACT_DIR="${WORK}/tractography"
mkdir -p "$TRACT_DIR"

WMFOD="${TRACT_DIR}/wmfod.mif"
GMFOD="${TRACT_DIR}/gm.mif"
CSFFOD="${TRACT_DIR}/csf.mif"
WMFOD_NORM="${TRACT_DIR}/wmfod_norm.mif"
GMFOD_NORM="${TRACT_DIR}/gm_norm.mif"
CSFFOD_NORM="${TRACT_DIR}/csf_norm.mif"
FIVETT="${TRACT_DIR}/5tt.mif"
GMWMI="${TRACT_DIR}/gmwmi.mif"
TRACKS="${TRACT_DIR}/tracks_10M.tck"
SIFT2_WEIGHTS="${TRACT_DIR}/sift2_weights.txt"
PARCELLATION="${TRACT_DIR}/parcellation.mif"
CONNECTOME_COUNT="${TRACT_DIR}/connectome_count.csv"
CONNECTOME_FA="${TRACT_DIR}/connectome_fa.csv"
CONNECTOME_LENGTH="${TRACT_DIR}/connectome_length.csv"

# MRtrix3Tissue bin — for ss3t_csd_beta1 (called by explicit path)
# Locate ss3t_csd_beta1 — search common installation paths
SS3T_CMD="${SS3T_CSD:-}"
if [[ -z "$SS3T_CMD" || ! -f "$SS3T_CMD" ]]; then
    for _candidate in         "${MRTRIX3TISSUE_HOME:-}/bin/ss3t_csd_beta1"         "${HOME}/MRtrix3Tissue/bin/ss3t_csd_beta1"         "${HOME}/mrtrix3tissue/bin/ss3t_csd_beta1"         "/opt/mrtrix3tissue/bin/ss3t_csd_beta1"; do
        if [[ -f "$_candidate" ]]; then
            SS3T_CMD="$_candidate"
            break
        fi
    done
fi

if [[ -z "$SS3T_CMD" || ! -f "$SS3T_CMD" ]]; then
    _log ERROR "ss3t_csd_beta1 not found"
    _log ERROR "Searched: MRTRIX3TISSUE_HOME, ~/MRtrix3Tissue/, ~/mrtrix3tissue/, /opt/mrtrix3tissue/"
    _log ERROR "Install: git clone https://github.com/3Tissue/MRtrix3Tissue.git ~/MRtrix3Tissue"
    _log ERROR "         cp ~/mrtrix3/bin/mrtrix3.py ~/MRtrix3Tissue/bin/mrtrix3.py"
    exit 1
fi
_log INFO "ss3t_csd_beta1: ${SS3T_CMD}"

N_THREADS="${OMP_NUM_THREADS:-4}"

# ---------------------------------------------------------------------------
# Step 2: SS3T-CSD FOD estimation
# ---------------------------------------------------------------------------
# Uses group-averaged response functions — critical for quantitative
# comparisons across subjects. Ordering is strict: WM GM CSF.

if [[ ! -f "$WMFOD" ]]; then
    _log INFO "Step 2: SS3T-CSD FOD estimation (group responses)"

    env "PYTHONPATH=${MRTRIX_PYTHON_PATH}" \
        "${SS3T_CMD}" \
            "$MIF_FINAL" \
            "$GROUP_WM"  "$WMFOD" \
            "$GROUP_GM"  "$GMFOD" \
            "$GROUP_CSF" "$CSFFOD" \
            -mask    "$BRAIN_MASK" \
            -nthreads "$N_THREADS" \
            -force

    _log OK "  WM FOD: ${WMFOD##*/}"
    _log OK "  GM:     ${GMFOD##*/}"
    _log OK "  CSF:    ${CSFFOD##*/}"
else
    _log INFO "Step 2: FODs already estimated, skipping"
fi

# ---------------------------------------------------------------------------
# Step 3: Multi-tissue intensity normalisation
# ---------------------------------------------------------------------------
# mtnormalise corrects bias fields using the 3-tissue compartments and
# performs global intensity normalisation — making absolute FOD amplitudes
# comparable across subjects. Essential for group studies.

if [[ ! -f "$WMFOD_NORM" ]]; then
    _log INFO "Step 3: Multi-tissue intensity normalisation (mtnormalise)"

    mtnormalise \
        "$WMFOD"  "$WMFOD_NORM" \
        "$GMFOD"  "$GMFOD_NORM" \
        "$CSFFOD" "$CSFFOD_NORM" \
        -mask "$BRAIN_MASK" \
        -nthreads "$N_THREADS" \
        -force

    _log OK "  Normalised WM FOD: ${WMFOD_NORM##*/}"
else
    _log INFO "Step 3: Normalisation already done, skipping"
fi

# ---------------------------------------------------------------------------
# Step 4: 5-tissue-type image (5ttgen hsvs)
# ---------------------------------------------------------------------------
# hsvs (Hybrid Surface and Volume Segmentation) uses FreeSurfer surfaces
# for accurate cortical representation — better than 5ttgen fsl for cortical
# seeding and ACT termination at the GM/WM boundary.

if [[ ! -f "$FIVETT" ]]; then
    _log INFO "Step 4: Generating 5TT image (5ttgen hsvs, FreeSurfer)"

    env "PYTHONPATH=${MRTRIX_PYTHON_PATH}" \
        5ttgen hsvs \
            "$FS_SUBJECT_DIR" \
            "$FIVETT" \
            -nthreads "$N_THREADS" \
            -force

    _log OK "  5TT: ${FIVETT##*/}"
else
    _log INFO "Step 4: 5TT already exists, skipping"
fi

# ---------------------------------------------------------------------------
# Step 5: GM/WM interface seeding image
# ---------------------------------------------------------------------------

if [[ ! -f "$GMWMI" ]]; then
    _log INFO "Step 5: Generating GM/WM interface image (5tt2gmwmi)"

    5tt2gmwmi "$FIVETT" "$GMWMI" -force

    _log OK "  GMWMI: ${GMWMI##*/}"
else
    _log INFO "Step 5: GMWMI already exists, skipping"
fi

# ---------------------------------------------------------------------------
# Step 6: Probabilistic tractography (tckgen iFOD2 + ACT)
# ---------------------------------------------------------------------------
# -seed_gmwmi:  seeds from the GM/WM interface — most anatomically precise
# -act:         Anatomically Constrained Tractography using 5TT image
# -backtrack:   allows streamlines to backtrack rather than terminate on failure
# -crop_at_gmwmi: crops streamline endpoints to the GMWMI surface
# -select 10M:  target 10 million accepted streamlines
# -cutoff 0.07: FOD amplitude threshold — conservative for b=1000 data

N_STREAMLINES="${DWIFORGE_N_STREAMLINES:-10000000}"

if [[ ! -f "$TRACKS" ]]; then
    _log INFO "Step 6: Tractography (iFOD2, ACT, ${N_STREAMLINES} streamlines)"
    _log INFO "  This step takes 30-90 min depending on hardware"

    tckgen \
        "$WMFOD_NORM" \
        "$TRACKS" \
        -act         "$FIVETT" \
        -seed_gmwmi  "$GMWMI" \
        -backtrack \
        -crop_at_gmwmi \
        -select      "$N_STREAMLINES" \
        -cutoff      "${DWIFORGE_TCK_CUTOFF:-0.07}" \
        -nthreads    "$N_THREADS" \
        -force

    N_ACTUAL=$(tckinfo "$TRACKS" 2>/dev/null | grep "count" | awk '{print $3}' || echo "unknown")
    _log OK "  Tractogram: ${TRACKS##*/} (${N_ACTUAL} streamlines)"
else
    _log INFO "Step 6: Tractogram already exists, skipping"
fi

# ---------------------------------------------------------------------------
# Step 7: SIFT2 streamline weight estimation
# ---------------------------------------------------------------------------
# tcksift2 produces a weights file (one float per streamline), NOT a filtered
# tractogram. ALL downstream tck2connectome and tcksample calls must include
# -tck_weights_in ${SIFT2_WEIGHTS} or the weights are silently ignored.

if [[ ! -f "$SIFT2_WEIGHTS" ]]; then
    _log INFO "Step 7: SIFT2 streamline weighting"

    env "PYTHONPATH=${MRTRIX_PYTHON_PATH}" \
        tcksift2 \
            "$TRACKS" \
            "$WMFOD_NORM" \
            "$SIFT2_WEIGHTS" \
            -act       "$FIVETT" \
            -nthreads  "$N_THREADS" \
            -force

    _log OK "  SIFT2 weights: ${SIFT2_WEIGHTS##*/}"
else
    _log INFO "Step 7: SIFT2 weights already exist, skipping"
fi

# ---------------------------------------------------------------------------
# Step 8: Parcellation — FreeSurfer DK atlas → connectome labels
# ---------------------------------------------------------------------------
# labelconvert maps FreeSurfer's aparc+aseg label scheme to a compact
# 0-N integer scheme required by tck2connectome.
# The FreeSurfer colour LUT and the MRtrix3 DK connectome config are required.

FS_LUT="${FREESURFER_HOME}/FreeSurferColorLUT.txt"
MRTRIX_DK_LUT=$(find "${MRTRIX_PYTHON_PATH%/lib}"/share/mrtrix3 \
    -name "fs_default.txt" 2>/dev/null | head -1 || true)

# Fallback search locations
if [[ -z "$MRTRIX_DK_LUT" ]]; then
    for search_path in \
        /opt/mrtrix3/share/mrtrix3 \
        /usr/share/mrtrix3 \
        /usr/local/share/mrtrix3; do
        candidate="${search_path}/labelconvert/fs_default.txt"
        [[ -f "$candidate" ]] && MRTRIX_DK_LUT="$candidate" && break
    done
fi

if [[ ! -f "${FS_LUT:-}" ]]; then
    _log ERROR "FreeSurfer colour LUT not found: ${FS_LUT:-\$FREESURFER_HOME/FreeSurferColorLUT.txt}"
    exit 1
fi

if [[ -z "$MRTRIX_DK_LUT" || ! -f "$MRTRIX_DK_LUT" ]]; then
    _log ERROR "MRtrix3 DK label config (fs_default.txt) not found"
    _log ERROR "Search paths: \$MRTRIX_PYTHON_PATH/../share/mrtrix3/labelconvert/"
    exit 1
fi

if [[ ! -f "$PARCELLATION" ]]; then
    _log INFO "Step 8: Converting FreeSurfer parcellation to connectome labels"
    _log INFO "  aparc+aseg: ${APARC_ASEG##*/}"
    _log INFO "  LUT:        ${FS_LUT##*/}"
    _log INFO "  DK config:  ${MRTRIX_DK_LUT##*/}"

    # Step 8a: Register aparc+aseg from T1w space to DWI space
    # The tractogram is in DWI space; parcellation must match.
    # b0_to_t1w.mat maps DWI→T1w; invert it to get T1w→DWI.
    B0_TO_T1W_MAT="${WORK}/b0_to_t1w.mat"
    T1W_TO_B0_MAT="${WORK}/t1w_to_b0.mat"
    APARC_NII="${WORK}/tractography/aparc+aseg.nii.gz"
    APARC_DWI="${WORK}/tractography/aparc+aseg_dwi.nii.gz"
    B0_MEAN="${WORK}/b0_mean.nii.gz"

    if [[ ! -f "$B0_TO_T1W_MAT" ]]; then
        _log ERROR "b0_to_t1w.mat not found — run stage 06 (tensor-fitting) first"
        exit 1
    fi

    # Invert the transform
    convert_xfm -omat "$T1W_TO_B0_MAT" -inverse "$B0_TO_T1W_MAT"
    _log INFO "  Transform inverted: t1w_to_b0.mat"

    # Convert aparc+aseg.mgz to NIfTI for flirt
    mrconvert "$APARC_ASEG" "$APARC_NII" -quiet -force

    # Apply inverse transform — nearest neighbour interpolation for label image
    flirt         -in      "$APARC_NII"         -ref     "$B0_MEAN"         -applyxfm         -init    "$T1W_TO_B0_MAT"         -interp  nearestneighbour         -out     "$APARC_DWI"
    _log OK "  Parcellation registered to DWI space: ${APARC_DWI##*/}"

    # Step 8b: labelconvert on DWI-space parcellation
    labelconvert \
        "$APARC_DWI" \
        "$FS_LUT" \
        "$MRTRIX_DK_LUT" \
        "$PARCELLATION" \
        -force

    N_LABELS=$(mrinfo "$PARCELLATION" -max 2>/dev/null | awk '{printf "%d", $1}' || echo "unknown")
    _log OK "  Parcellation: ${PARCELLATION##*/} (${N_LABELS} regions)"
else
    _log INFO "Step 8: Parcellation already exists, skipping"
fi

# ---------------------------------------------------------------------------
# Step 9: Connectome — SIFT2-weighted streamline count
# ---------------------------------------------------------------------------
# -tck_weights_in: REQUIRED — without this SIFT2 weights are silently ignored
# -symmetric:      make matrix symmetric (undirected connectivity)
# -zero_diagonal:  no self-connections

if [[ ! -f "$CONNECTOME_COUNT" ]]; then
    _log INFO "Step 9: Building SIFT2-weighted streamline count connectome"

    tck2connectome \
        "$TRACKS" \
        "$PARCELLATION" \
        "$CONNECTOME_COUNT" \
        -tck_weights_in  "$SIFT2_WEIGHTS" \
        -symmetric \
        -zero_diagonal \
        -out_assignments "${TRACT_DIR}/assignments.txt" \
        -nthreads        "$N_THREADS" \
        -force

    _log OK "  Count connectome: ${CONNECTOME_COUNT##*/}"
else
    _log INFO "Step 9: Count connectome already exists, skipping"
fi

# ---------------------------------------------------------------------------
# Step 10: Connectome — mean FA along streamlines
# ---------------------------------------------------------------------------
# tcksample samples the FA map along each streamline and computes per-
# streamline mean FA. tck2connectome then averages per-edge using SIFT2
# weights for the final NxN mean-FA matrix.

if [[ -f "$FA_MAP" && ! -f "$CONNECTOME_FA" ]]; then
    _log INFO "Step 10: Building mean-FA connectome"

    FA_SAMPLE="${TRACT_DIR}/fa_per_streamline.txt"

    # Convert FA NIfTI to .mif for mrconvert compatibility
    FA_MIF="${TRACT_DIR}/fa.mif"
    if [[ ! -f "$FA_MIF" ]]; then
        mrconvert "$FA_MAP" "$FA_MIF" -quiet -force
    fi

    # Sample FA along each streamline (mean per streamline)
    # tcksample samples the FA value along each streamline.
    # Note: -tck_weights_in is NOT valid for tcksample — weights are
    # only applied in the subsequent tck2connectome call.
    tcksample \
        "$TRACKS" \
        "$FA_MIF" \
        "$FA_SAMPLE" \
        -stat_tck mean \
        -nthreads "$N_THREADS" \
        -force

    # Build connectome using sampled FA values as edge weights
    tck2connectome \
        "$TRACKS" \
        "$PARCELLATION" \
        "$CONNECTOME_FA" \
        -scale_file      "$FA_SAMPLE" \
        -stat_edge       mean \
        -tck_weights_in  "$SIFT2_WEIGHTS" \
        -symmetric \
        -zero_diagonal \
        -nthreads        "$N_THREADS" \
        -force

    _log OK "  FA connectome: ${CONNECTOME_FA##*/}"

    # Also build mean streamline length connectome (useful for normalisation)
    tck2connectome \
        "$TRACKS" \
        "$PARCELLATION" \
        "$CONNECTOME_LENGTH" \
        -scale_length \
        -stat_edge       mean \
        -tck_weights_in  "$SIFT2_WEIGHTS" \
        -symmetric \
        -zero_diagonal \
        -nthreads        "$N_THREADS" \
        -force

    _log OK "  Length connectome: ${CONNECTOME_LENGTH##*/}"

elif [[ ! -f "$FA_MAP" ]]; then
    _log WARN "Step 10: FA map not found — skipping mean-FA connectome"
else
    _log INFO "Step 10: FA connectome already exists, skipping"
fi

# ---------------------------------------------------------------------------
# Verify outputs
# ---------------------------------------------------------------------------

REQUIRED=(
    "$WMFOD_NORM"
    "$FIVETT"
    "$TRACKS"
    "$SIFT2_WEIGHTS"
    "$PARCELLATION"
    "$CONNECTOME_COUNT"
)
MISSING=()
for f in "${REQUIRED[@]}"; do
    [[ ! -f "$f" ]] && MISSING+=("${f##*/}")
done

if [[ "${#MISSING[@]}" -gt 0 ]]; then
    _log WARN "Some expected outputs missing: ${MISSING[*]}"
fi

# ---------------------------------------------------------------------------
# Update capability profile
# ---------------------------------------------------------------------------

"${PYTHON_EXECUTABLE:-python3}" - << PYEOF
import json, os

cap_path = '${CAPABILITY_JSON}'
with open(cap_path) as f:
    d = json.load(f)

td = '${TRACT_DIR}'
d['tractography'] = {
    'status':             'complete',
    'output_dir':         td,
    'fod_method':         'ss3t_csd (msmt_csd with dhollander group responses)',
    'n_streamlines':      ${N_STREAMLINES},
    'sift2_weighted':     True,
    'wmfod_norm':         '${WMFOD_NORM}',
    'tracks':             '${TRACKS}',
    'sift2_weights':      '${SIFT2_WEIGHTS}',
    'parcellation':       '${PARCELLATION}',
    'parcellation_atlas': 'Desikan-Killiany (FreeSurfer aparc+aseg)',
    'connectome_count':   '${CONNECTOME_COUNT}',
    'connectome_fa':      '${CONNECTOME_FA}' if os.path.exists('${CONNECTOME_FA}') else None,
    'connectome_length':  '${CONNECTOME_LENGTH}' if os.path.exists('${CONNECTOME_LENGTH}') else None,
}

with open(cap_path, 'w') as f:
    json.dump(d, f, indent=2)
print('capability.json updated with tractography outputs')
PYEOF

_log OK "Stage 09 complete"
_log OK "  Tractogram:       ${TRACKS##*/}"
_log OK "  Count connectome: ${CONNECTOME_COUNT##*/}"
[[ -f "$CONNECTOME_FA" ]] && _log OK "  FA connectome:    ${CONNECTOME_FA##*/}"

log_stage_end "09_tractography" "$SUB"
