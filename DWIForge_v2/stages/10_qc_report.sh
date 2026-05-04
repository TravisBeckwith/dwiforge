#!/usr/bin/env bash
# stages/10_qc_report.sh — QC report generation
# =============================================================================
# Generates a per-subject PDF QC report summarising all pipeline stages,
# key metrics, warnings, and image slices for visual inspection.
#
# The report includes:
#   - Stage completion status (pass / skip / fail / warn)
#   - Acquisition summary (b-values, directions, PE direction, partial Fourier)
#   - EPI correction method and SDC status
#   - DTI metrics: FA / MD / AD / RD mean ± SD in WM mask
#   - NODDI metrics: NDI / ODI / ISOVF mean ± SD in WM mask
#   - Connectome summary: N nodes, density, mean strength
#   - Image slices: b0, FA, MD, NODDI maps, 5TT, FODs
#   - Warning log: all flagged issues from capability profile
#
# Output:
#   DIR_LOGS/<sub>/qc_report.pdf
#
# Delegates to python/qc_report.py for all rendering.
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
log_stage_start "10_qc_report" "$SUB"

if [[ ! -f "$CAPABILITY_JSON" ]]; then
    _log ERROR "capability.json not found: ${CAPABILITY_JSON}"
    _log ERROR "Run stage 00 (qc_bids) before stage 10"
    exit 1
fi

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

REPORT_PDF="${LOGS}/qc_report.pdf"

if [[ -f "$REPORT_PDF" && "${DWIFORGE_FORCE_QC:-false}" != "true" ]]; then
    _log INFO "QC report already exists: ${REPORT_PDF##*/}"
    _log INFO "Set DWIFORGE_FORCE_QC=true to regenerate"
    log_stage_end "10_qc_report" "$SUB"
    exit 0
fi

# ---------------------------------------------------------------------------
# Generate report
# ---------------------------------------------------------------------------

_log INFO "Generating QC report for ${SUB}"

# Use DWIFORGE_DEPS_DIR for nibabel/matplotlib (our deps, not DESIGNER's)
env "PYTHONPATH=${DWIFORGE_DEPS_DIR}:${PYTHONPATH:-}" \
    "${PYTHON_EXECUTABLE:-python3}" \
    "${DWIFORGE_ROOT}/python/qc_report.py" \
    --subject         "$SUB" \
    --capability_json "$CAPABILITY_JSON" \
    --work_dir        "$WORK" \
    --output          "$REPORT_PDF" \
    --pipeline_version "${DWIFORGE_VERSION:-2.0}"

if [[ ! -f "$REPORT_PDF" ]]; then
    _log ERROR "QC report not generated: ${REPORT_PDF}"
    exit 1
fi

SIZE_KB=$(du -k "$REPORT_PDF" | cut -f1)
_log OK "QC report: ${REPORT_PDF##*/} (${SIZE_KB} KB)"

log_stage_end "10_qc_report" "$SUB"
