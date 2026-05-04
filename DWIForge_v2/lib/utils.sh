#!/usr/bin/env bash
# lib/utils.sh — dwiforge utility functions
#
# Provides:
#   dirs_init subject              Create all per-subject directories
#   dirs_validate                  Check all root dirs exist/are writable
#   disk_check_all subject         Check free space on all locations
#   disk_free_gb path              Print free GB on the filesystem of path
#   cleanup_subject subject tier   Remove intermediates up to tier N
#   output_sanity_check sub file min_mb   Verify output exists and has size
#   require_tool name [version_cmd]       Abort if tool not found
#   require_python_module module          Abort if Python module not importable

# ---------------------------------------------------------------------------
# Directory initialisation
# ---------------------------------------------------------------------------

# All directories that must exist for a subject run.
# Called once per subject by the orchestrator before stage dispatch.
dirs_init() {
    local sub="$1"

    local dirs=(
        "${DWIFORGE_DIR_WORK}/${sub}/dwi"
        "${DWIFORGE_DIR_WORK}/${sub}/anat"
        "${DWIFORGE_DIR_WORK}/${sub}/reg"
        "${DWIFORGE_DIR_WORK}/${sub}/tmp"
        "${DWIFORGE_DIR_WORK}/${sub}/checkpoints"
        "${DWIFORGE_DIR_OUTPUT}/${sub}/dti"
        "${DWIFORGE_DIR_OUTPUT}/${sub}/noddi"
        "${DWIFORGE_DIR_OUTPUT}/${sub}/tracts"
        "${DWIFORGE_DIR_OUTPUT}/${sub}/connectomes"
        "${DWIFORGE_DIR_LOGS}/${sub}"
        "${DWIFORGE_DIR_QC}/${sub}"
    )

    for d in "${dirs[@]}"; do
        if ! mkdir -p "$d" 2>/dev/null; then
            log "ERROR" "Cannot create directory: ${d}"
            return 1
        fi
    done

    # In SLURM worker mode, isolate work under the job ID to prevent
    # collisions between array tasks for the same subject.
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        local slurm_work="${DWIFORGE_DIR_WORK}/${sub}/job-${SLURM_JOB_ID}"
        mkdir -p "${slurm_work}/"{dwi,anat,reg,tmp}
        export DWIFORGE_SUB_WORK="${slurm_work}"
    else
        export DWIFORGE_SUB_WORK="${DWIFORGE_DIR_WORK}/${sub}"
    fi

    log_sub "DEBUG" "$sub" "Directories initialised (work: ${DWIFORGE_SUB_WORK})"
}

# Validate that all configured root directories exist and are writable.
# Called once at pipeline startup, not per-subject.
dirs_validate() {
    local ok=true

    # Source must exist and be readable
    if [[ ! -d "${DWIFORGE_DIR_SOURCE}" ]]; then
        log "ERROR" "Source directory does not exist: ${DWIFORGE_DIR_SOURCE}"
        ok=false
    elif [[ ! -r "${DWIFORGE_DIR_SOURCE}" ]]; then
        log "ERROR" "Source directory is not readable: ${DWIFORGE_DIR_SOURCE}"
        ok=false
    fi

    # All write locations: create if missing, check writability
    local write_dirs=(
        DWIFORGE_DIR_WORK
        DWIFORGE_DIR_OUTPUT
        DWIFORGE_DIR_FREESURFER
        DWIFORGE_DIR_LOGS
        DWIFORGE_DIR_QC
    )
    for var in "${write_dirs[@]}"; do
        local path="${!var}"
        if [[ -z "$path" ]]; then
            log "ERROR" "${var} is not set"
            ok=false
            continue
        fi
        if ! mkdir -p "$path" 2>/dev/null; then
            log "ERROR" "Cannot create ${var}: ${path}"
            ok=false
            continue
        fi
        if [[ ! -w "$path" ]]; then
            log "ERROR" "${var} is not writable: ${path}"
            ok=false
        fi
    done

    [[ "$ok" == true ]]
}

# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------

# Print free space in GB on the filesystem containing 'path'.
disk_free_gb() {
    local path="$1"
    # df -P gives POSIX output; awk extracts 'Available' in 1K blocks
    local avail_kb
    avail_kb=$(df -Pk "$path" 2>/dev/null | awk 'NR==2{print $4}')
    if [[ -z "$avail_kb" ]]; then
        echo "0"
        return 1
    fi
    # bc not available everywhere; use awk for division
    awk "BEGIN{printf \"%.1f\", ${avail_kb}/1048576}"
}

# Check all storage locations against configured minimums.
# Logs a warning for any location below threshold; does not abort.
# Returns 1 if any location is critically low (below half of threshold).
disk_check_all() {
    local sub="${1:-}"
    local prefix="${sub:+[${sub}] }"
    local critical=false

    _disk_check_one() {
        local label="$1"
        local path="$2"
        local min_gb="$3"

        local free
        free=$(disk_free_gb "$path")
        local half_min
        half_min=$(awk "BEGIN{printf \"%.1f\", ${min_gb}/2}")

        if awk "BEGIN{exit (${free} >= ${min_gb}) ? 0 : 1}"; then
            log "DEBUG" "${prefix}Disk OK — ${label}: ${free} GB free (min: ${min_gb} GB)"
        elif awk "BEGIN{exit (${free} >= ${half_min}) ? 0 : 1}"; then
            log "WARN" "${prefix}Low disk space — ${label}: ${free} GB free (min: ${min_gb} GB): ${path}"
        else
            log "WARN" "${prefix}Critically low disk — ${label}: ${free} GB free (min: ${min_gb} GB): ${path}"
            critical=true
        fi
    }

    _disk_check_one "work"       "${DWIFORGE_DIR_WORK}"       "${DWIFORGE_MIN_FREE_GB_WORK}"
    _disk_check_one "output"     "${DWIFORGE_DIR_OUTPUT}"     "${DWIFORGE_MIN_FREE_GB_OUTPUT}"
    _disk_check_one "freesurfer" "${DWIFORGE_DIR_FREESURFER}" "${DWIFORGE_MIN_FREE_GB_FREESURFER}"

    [[ "$critical" == false ]]
}

# ---------------------------------------------------------------------------
# Cleanup tiers
# ---------------------------------------------------------------------------
# Tiers are cumulative: tier 2 removes everything tier 1 removes, plus more.
# Cleanup only runs if DWIFORGE_CLEANUP_TIER > 0.
# Each deletion is guarded: the file must exist AND the corresponding output
# must exist and pass a minimum size check before any intermediate is removed.

cleanup_subject() {
    local sub="$1"
    local tier="${2:-${DWIFORGE_CLEANUP_TIER:-0}}"
    local work="${DWIFORGE_DIR_WORK}/${sub}"

    if [[ "$tier" -le 0 ]]; then
        log_sub "DEBUG" "$sub" "Cleanup skipped (tier 0 — keep all)"
        return 0
    fi

    log_sub "INFO" "$sub" "Running cleanup tier ${tier}"

    # --- Tier 1: tmp/ and raw .mif intermediates ---
    if [[ "$tier" -ge 1 ]]; then
        # tmp/ is always safe to remove
        if [[ -d "${work}/tmp" ]]; then
            rm -rf "${work}/tmp"
            log_sub "DEBUG" "$sub" "Removed tmp/"
        fi
        # Raw MRtrix .mif files — only if dwi_final exists
        if _output_exists_sub "$sub" "dwi_final"; then
            find "${work}/dwi" -name "*.mif" \
                ! -name "dwi_final*" \
                -delete 2>/dev/null && \
            log_sub "DEBUG" "$sub" "Removed intermediate .mif files"
        fi
    fi

    # --- Tier 2: eddy/bias/registration intermediates ---
    if [[ "$tier" -ge 2 ]]; then
        local tier2_patterns=(
            "dwi_denoised.*"
            "dwi_degibbs.*"
            "dwi_eddy.*"
            "dwi_biascorr.*"
            "dwi_for_ml.*"
            "dwi_ml.*"
        )
        for pat in "${tier2_patterns[@]}"; do
            find "${work}/dwi" -name "$pat" -delete 2>/dev/null || true
        done
        # Registration field maps — only if outputs exist
        if _output_exists_sub "$sub" "fa"; then
            find "${work}/reg" -name "field_map*" -delete 2>/dev/null || true
            find "${work}/anat" -name "T1w_reg*"  -delete 2>/dev/null || true
        fi
        log_sub "DEBUG" "$sub" "Tier 2 intermediates removed"
    fi

    # --- Tier 3: raw tractograms ---
    if [[ "$tier" -ge 3 ]]; then
        local tracts_dir="${DWIFORGE_DIR_OUTPUT}/${sub}/tracts"
        # Only remove raw .tck if a SIFT-filtered version exists
        if ls "${tracts_dir}"/*_sift.tck 2>/dev/null | grep -q .; then
            find "${tracts_dir}" -name "*.tck" \
                ! -name "*_sift*" \
                -delete 2>/dev/null && \
            log_sub "DEBUG" "$sub" "Removed raw tractograms (SIFT versions retained)"
        else
            log_sub "WARN" "$sub" "Tier 3: no SIFT tractograms found, skipping raw .tck removal"
        fi
    fi

    # --- Tier 4: all work intermediates ---
    if [[ "$tier" -ge 4 ]]; then
        # Keep checkpoints so --resume still works
        find "${work}" -mindepth 1 \
            ! -path "${work}/checkpoints" \
            ! -path "${work}/checkpoints/*" \
            -delete 2>/dev/null
        log_sub "DEBUG" "$sub" "All work intermediates removed (checkpoints retained)"
    fi

    log_sub "OK" "$sub" "Cleanup tier ${tier} complete"
}

# Check that the named output exists and is non-empty for a subject
_output_exists_sub() {
    local sub="$1"
    local name="$2"
    # Looks for any file matching *_<name>.nii.gz in the output tree
    find "${DWIFORGE_DIR_OUTPUT}/${sub}" \
        -name "*_${name}.nii.gz" -size +0c 2>/dev/null | grep -q .
}

# ---------------------------------------------------------------------------
# Output sanity checks
# ---------------------------------------------------------------------------

# Verify a specific output file exists and meets a minimum size in MB.
# Usage: output_sanity_check sub /path/to/file.nii.gz 5
output_sanity_check() {
    local sub="$1"
    local filepath="$2"
    local min_mb="${3:-1}"

    if [[ ! -f "$filepath" ]]; then
        log_sub "WARN" "$sub" "Expected output not found: ${filepath}"
        return 1
    fi

    local size_kb
    size_kb=$(du -k "$filepath" 2>/dev/null | cut -f1)
    local min_kb=$(( min_mb * 1024 ))

    if [[ -z "$size_kb" || "$size_kb" -lt "$min_kb" ]]; then
        log_sub "WARN" "$sub" \
            "Output suspiciously small (${size_kb}KB < ${min_mb}MB): ${filepath}"
        return 1
    fi

    return 0
}

# ---------------------------------------------------------------------------
# Tool availability checks
# ---------------------------------------------------------------------------

# Abort with a clear message if a required tool is not in PATH.
require_tool() {
    local tool="$1"
    local version_cmd="${2:-}"

    if ! command -v "$tool" >/dev/null 2>&1; then
        log "ERROR" "Required tool not found: ${tool}"
        log "ERROR" "Ensure it is installed and in PATH."
        return 1
    fi

    if [[ -n "$version_cmd" ]]; then
        local version
        version=$(eval "$version_cmd" 2>/dev/null | head -1 || echo "unknown")
        log "DEBUG" "Tool found: ${tool} — ${version}"
    else
        log "DEBUG" "Tool found: ${tool}"
    fi
}

# Warn (non-fatal) if a Python module is not importable.
require_python_module() {
    local module="$1"
    local pip_name="${2:-$1}"

    if ! "${PYTHON_EXECUTABLE:-python3}" -c "import ${module}" >/dev/null 2>&1; then
        log "WARN" "Python module not available: ${module}"
        log "WARN" "Install with: pip install ${pip_name}"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

# Find all sub-* directories in DIR_SOURCE that have DWI data.
# Prints one subject ID per line.
discover_subjects() {
    find "${DWIFORGE_DIR_SOURCE}" \
        -maxdepth 1 \
        -type d \
        -name 'sub-*' \
        -exec test -d '{}/dwi' \; \
        -printf '%f\n' 2>/dev/null | sort
}

# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

# Resolve the absolute path of the dwiforge installation root,
# regardless of where the calling script lives.
dwiforge_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    # Walk up until we find a dwiforge.toml or lib/ directory
    local dir="$script_dir"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "${dir}/dwiforge.toml" ]] || [[ -d "${dir}/lib" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    # Fall back to script directory
    echo "$script_dir"
}

# Rotate a log file if it exceeds a size limit (default 50MB)
rotate_log() {
    local logfile="$1"
    local max_mb="${2:-50}"
    if [[ -f "$logfile" ]]; then
        local size_mb
        size_mb=$(du -m "$logfile" 2>/dev/null | cut -f1)
        if [[ "${size_mb:-0}" -ge "$max_mb" ]]; then
            mv "$logfile" "${logfile}.$(date +%Y%m%d_%H%M%S).gz" 2>/dev/null || true
            gzip -f "${logfile}.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
        fi
    fi
}
