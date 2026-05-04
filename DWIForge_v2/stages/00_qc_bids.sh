#!/usr/bin/env bash
# stages/00_qc_bids.sh — BIDS Validation and Capability Profiling
#
# Called by dwiforge.sh orchestrator:
#   bash stages/00_qc_bids.sh <subject_id>
#
# Reads all configuration from DWIFORGE_* environment variables.
# Writes:
#   DIR_LOGS/<sub>/capability.json   — machine-readable profile (read by all stages)
#   DIR_QC/<sub>/qc_bids.txt         — human-readable summary
#
# Exit codes:
#   0  QC passed — subject may proceed
#   1  Critical failure — subject aborted, no further stages run
#
# This is a FATAL stage. Failure aborts the subject immediately.

set -euo pipefail

STAGE_NAME="00_qc_bids"
SUB="${1:?Usage: $0 <subject_id>}"

# ---------------------------------------------------------------------------
# Source libraries
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DWIFORGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${DWIFORGE_ROOT}/lib/logging.sh"
source "${DWIFORGE_ROOT}/lib/utils.sh"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LOG_SUB="${DWIFORGE_DIR_LOGS}/${SUB}"
QC_SUB="${DWIFORGE_DIR_QC}/${SUB}"
CAPABILITY_JSON="${LOG_SUB}/capability.json"
QC_SUMMARY="${QC_SUB}/qc_bids.txt"
QC_SCRIPT="${DWIFORGE_ROOT}/python/qc_bids.py"

mkdir -p "${LOG_SUB}" "${QC_SUB}"
export DWIFORGE_LOG_FILE="${LOG_SUB}/${STAGE_NAME}.log"

_log() { log_sub "$1" "$SUB" "${*:2}"; }

log_stage_start "$STAGE_NAME" "$SUB"

# ---------------------------------------------------------------------------
# Run QC analyser
# ---------------------------------------------------------------------------

_log INFO "Running BIDS QC analysis..."

QC_ARGS=(
    "$SUB"
    "${DWIFORGE_DIR_SOURCE}"
    "${CAPABILITY_JSON}"
    --b0-threshold "${DWIFORGE_B0_THRESHOLD:-50}"
    --noddi-min-directions "${DWIFORGE_NODDI_MIN_DIRECTIONS:-30}"
    --noddi-high-directions "${DWIFORGE_NODDI_HIGH_DIRECTIONS:-60}"
)

# NDC is opt-in: slower but adds a useful quality metric
if [ "${DWIFORGE_QC_RUN_NDC:-false}" = "true" ]; then
    QC_ARGS+=(--run-ndc)
    _log INFO "NDC quality metric enabled"
fi

"${PYTHON_EXECUTABLE:-python3}" "$QC_SCRIPT" "${QC_ARGS[@]}"
QC_EXIT=$?

# ---------------------------------------------------------------------------
# Read results and act on them
# ---------------------------------------------------------------------------

if [ ! -f "$CAPABILITY_JSON" ]; then
    _log ERROR "Capability profile not written — QC script may have crashed"
    exit 1
fi

# Extract key fields from the JSON using Python (avoids jq dependency)
read_cap() {
    "${PYTHON_EXECUTABLE:-python3}" - <<PYEOF
import json, sys
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(d.get('$1', '$2'))
PYEOF
}

read_cap_nested() {
    "${PYTHON_EXECUTABLE:-python3}" - <<PYEOF
import json, sys
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
section = d.get('$1', {})
print(section.get('$2', '$3'))
PYEOF
}

QC_PASSED=$(read_cap "qc_passed" "false")
N_WARNINGS=$(
    "${PYTHON_EXECUTABLE:-python3}" - <<PYEOF
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(len(d.get('warnings', [])))
PYEOF
)
N_CRITICAL=$(
    "${PYTHON_EXECUTABLE:-python3}" - <<PYEOF
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
print(len(d.get('critical_failures', [])))
PYEOF
)

# Log warnings from the profile
if [ "${N_WARNINGS}" -gt 0 ]; then
    _log WARN "${N_WARNINGS} warning(s) in capability profile:"
    "${PYTHON_EXECUTABLE:-python3}" - <<PYEOF
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
for w in d.get('warnings', []):
    print(f"  • {w}")
PYEOF
fi

# Log critical failures
if [ "${N_CRITICAL}" -gt 0 ]; then
    _log ERROR "${N_CRITICAL} critical failure(s):"
    "${PYTHON_EXECUTABLE:-python3}" - <<PYEOF
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)
for c in d.get('critical_failures', []):
    print(f"  ✗ {c}")
PYEOF
fi

# ---------------------------------------------------------------------------
# Log capability summary
# ---------------------------------------------------------------------------

if [ "$QC_PASSED" = "True" ] || [ "$QC_PASSED" = "true" ]; then
    _log OK "QC passed — capability summary:"

    "${PYTHON_EXECUTABLE:-python3}" - <<PYEOF
import json
with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)

shells  = d.get('shells', {})
caps    = d.get('capabilities', {})
acq     = d.get('acquisition', {})

n_shells = shells.get('count', '?')
b_vals   = shells.get('b_values', [])
dirs     = shells.get('directions_per_shell', {})
single   = shells.get('is_single_shell', False)

print(f"  Shells:       {n_shells} ({'single' if single else 'multi'})"
      f"  b={b_vals}")
for b, n in dirs.items():
    print(f"    b={b:>5s}: {n} directions")

print()
tick = lambda v: '✓' if v else '✗'
conf = caps.get('noddi_confidence') or 'disabled'
approx = ' (single-shell approx)' if caps.get('noddi_single_shell_approx') else ''
print(f"  DTI:            {tick(caps.get('dti'))}")
print(f"  NODDI:          {tick(caps.get('noddi'))}  [{conf}{approx}]")
print(f"  DKI:            {tick(caps.get('dki'))}")
print(f"  CSD (SS):       {tick(caps.get('csd_single_shell'))}")
print(f"  MSMT-CSD:       {tick(caps.get('msmt_csd'))}")
print(f"  Tractography:   {tick(caps.get('tractography'))}")

print()
print(f"  Topup ready:    {tick(acq.get('topup_ready'))}")
print(f"  Synb0 possible: {tick(acq.get('synb0_possible'))}")
print(f"  T1w available:  {tick(acq.get('t1w_available'))}")
snr = d.get('quality', {}).get('snr_b0_estimate')
if snr:
    print(f"  SNR (b0 est):   {snr:.1f}")
PYEOF
fi

# ---------------------------------------------------------------------------
# Write human-readable QC summary to DIR_QC
# ---------------------------------------------------------------------------

"${PYTHON_EXECUTABLE:-python3}" - <<PYEOF
import json
from datetime import datetime

with open('${CAPABILITY_JSON}') as f:
    d = json.load(f)

shells  = d.get('shells', {})
caps    = d.get('capabilities', {})
acq     = d.get('acquisition', {})
quality = d.get('quality', {})
data    = d.get('data', {})

lines = [
    "dwiforge BIDS QC Report",
    "=" * 50,
    f"Subject:    {d.get('subject')}",
    f"Generated:  {d.get('generated')}",
    f"QC passed:  {d.get('qc_passed')}",
    "",
    "DATA",
    f"  DWI:        {data.get('dwi_file','')}",
    f"  Volumes:    {data.get('n_volumes','?')}  "
    f"(b0: {data.get('n_b0','?')}, DWI: {data.get('n_dwi','?')})",
    f"  Matrix:     {data.get('matrix','?')}",
    f"  Voxel size: {data.get('voxel_size_mm','?')} mm",
    "",
    "SHELLS",
    f"  Count:      {shells.get('count','?')}",
    f"  b-values:   {shells.get('b_values',[])}",
]
for b, n in shells.get('directions_per_shell', {}).items():
    lines.append(f"    b={b}: {n} directions")

lines += [
    "",
    "CAPABILITIES",
]
tick = lambda v: "YES" if v else "NO"
conf = caps.get('noddi_confidence') or 'disabled'
approx = " (single-shell approximation)" if caps.get('noddi_single_shell_approx') else ""
lines += [
    f"  DTI:            {tick(caps.get('dti'))}",
    f"  NODDI:          {tick(caps.get('noddi'))}  [{conf}{approx}]",
    f"  DKI:            {tick(caps.get('dki'))}",
    f"  CSD (SS):       {tick(caps.get('csd_single_shell'))}",
    f"  MSMT-CSD:       {tick(caps.get('msmt_csd'))}",
    f"  Tractography:   {tick(caps.get('tractography'))}",
    "",
    "ACQUISITION",
    f"  Reverse PE:     {tick(acq.get('reverse_pe_available'))}",
    f"  Topup ready:    {tick(acq.get('topup_ready'))}",
    f"  Synb0 possible: {tick(acq.get('synb0_possible'))}",
    f"  T1w:            {tick(acq.get('t1w_available'))}",
    f"  Phase enc dir:  {acq.get('phase_encoding_direction','unknown')}",
    "",
    "QUALITY",
    f"  SNR estimate:   {quality.get('snr_b0_estimate','N/A')}",
    f"  Outlier vols:   {quality.get('n_outlier_volumes',0)}",
]
if quality.get('outlier_volume_indices'):
    lines.append(f"    Indices: {quality['outlier_volume_indices']}")
if quality.get('gradient_issues'):
    lines.append("  Gradient issues:")
    for iss in quality['gradient_issues']:
        lines.append(f"    • {iss}")

if d.get('warnings'):
    lines += ["", "WARNINGS"]
    for w in d['warnings']:
        lines.append(f"  • {w}")

if d.get('critical_failures'):
    lines += ["", "CRITICAL FAILURES"]
    for c in d['critical_failures']:
        lines.append(f"  ✗ {c}")

with open('${QC_SUMMARY}', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print("QC summary written: ${QC_SUMMARY}")
PYEOF

# ---------------------------------------------------------------------------
# Final outcome
# ---------------------------------------------------------------------------

if [ "$QC_PASSED" != "True" ] && [ "$QC_PASSED" != "true" ]; then
    _log ERROR "BIDS QC FAILED — subject will not be processed"
    _log ERROR "Review: ${CAPABILITY_JSON}"
    _log ERROR "Summary: ${QC_SUMMARY}"
    exit 1
fi

log_stage_end "$STAGE_NAME" "$SUB"
exit 0
