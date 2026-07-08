#!/usr/bin/env bash
# stages/11_connectome_stats.sh — Compositional (CLR) transform of connectome
# =============================================================================
# Adds a derived, group-analysis-ready representation of the SIFT2-weighted
# streamline-count connectome. Structural connectome edge weights are
# compositional data (each node's row of connections is a set of parts
# whose *relative* magnitudes are meaningful, not their absolute scale --
# see python/connectome_stats.py docstring for the full rationale and
# citations). Running correlation, PCA, regression, or group comparisons
# directly on raw streamline counts violates the compositional constraint
# and can introduce spurious structure. This stage does not replace the
# raw matrix (kept for QC/visualisation in Stage 10) -- it adds a CLR
# matrix alongside it for use in any downstream group-level statistics.
#
# Prerequisites:
#   Stage 09: tractography — connectome_count.csv must exist
#
# Steps:
#   1. Locate connectome_count.csv from Stage 09
#   2. Run connectome_stats.py — Bayesian-multiplicative zero replacement
#      followed by row-wise CLR transform
#   3. Update capability profile
#
# Outputs (in DIR_WORK/<sub>/tractography/):
#   connectome_count_clr.csv              — CLR-transformed connectome
#   connectome_count_clr_diagnostics.json — zero fractions, disconnected nodes
#
# Note:
#   This stage is advisory, not fatal — a failure here does not block
#   downstream stages, since the raw connectome remains fully usable for
#   QC and for any non-compositional analysis method.
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
log_stage_start "11_connectome_stats" "$SUB"

# ---------------------------------------------------------------------------
# Step 1: Locate Stage 09 output
# ---------------------------------------------------------------------------

TRACT_DIR="${WORK}/tractography"
CONNECTOME_COUNT="${TRACT_DIR}/connectome_count.csv"
CONNECTOME_CLR="${TRACT_DIR}/connectome_count_clr.csv"

if [[ ! -f "$CONNECTOME_COUNT" ]]; then
    _log ERROR "connectome_count.csv not found: ${CONNECTOME_COUNT}"
    _log ERROR "Run stage 09 (tractography) before stage 11"
    exit 1
fi

MODE="${DWIFORGE_CONNECTOME_STATS_MODE:-matrix}"
DELTA_FRAC="${DWIFORGE_CONNECTOME_STATS_DELTA_FRAC:-0.5}"

# ---------------------------------------------------------------------------
# Step 2: CLR transform
# ---------------------------------------------------------------------------

if [[ ! -f "$CONNECTOME_CLR" ]]; then
    _log INFO "Step 2: CLR transform (mode=${MODE}, delta_frac=${DELTA_FRAC})"

    # Invoke the helper directly (no output pipe). The orchestrator's
    # _run_stage already routes this script's stdout/stderr through the
    # per-stage logger, so piping through a `while read` loop here would
    # double-prefix every line AND mask the Python exit code (the loop's
    # exit status would replace Python's). Check the real exit code.
    if ! "${PYTHON_EXECUTABLE:-python3}" \
            "${DWIFORGE_ROOT}/python/connectome_stats.py" \
            --connectome      "$CONNECTOME_COUNT" \
            --output-dir      "$TRACT_DIR" \
            --mode            "$MODE" \
            --delta-frac      "$DELTA_FRAC" \
            --capability_json "$CAPABILITY_JSON"; then
        _log ERROR "connectome_stats.py failed (see output above)"
        exit 1
    fi

    if [[ ! -f "$CONNECTOME_CLR" ]]; then
        _log ERROR "connectome_stats.py did not produce ${CONNECTOME_CLR##*/}"
        exit 1
    fi

    _log OK "  CLR connectome: ${CONNECTOME_CLR##*/}"
else
    _log INFO "Step 2: CLR connectome already exists, skipping"
fi

_log OK "Stage 11 complete"
_log OK "  CLR connectome: ${CONNECTOME_CLR##*/}"

log_stage_end "11_connectome_stats" "$SUB"
