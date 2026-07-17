#!/usr/bin/env bash
# dwiforge.sh — DWI Processing Pipeline Orchestrator
# Version 2.0
#
# Usage:
#   ./dwiforge.sh [options]
#   ./dwiforge.sh --slurm [options]          Submit SLURM job array
#   ./dwiforge.sh --slurm-worker <sub-ID>    Run single subject (SLURM worker)
#
# See --help for full option reference.
#
# Storage locations are configured in dwiforge.toml.
# Config file lookup: --config flag > ./dwiforge.toml > ~/.config/dwiforge/dwiforge.toml

set -euo pipefail

DWIFORGE_VERSION="2.0"
DWIFORGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Source library files
# ---------------------------------------------------------------------------

source "${DWIFORGE_ROOT}/lib/logging.sh"
source "${DWIFORGE_ROOT}/lib/utils.sh"
source "${DWIFORGE_ROOT}/lib/env_setup.sh"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_usage() {
    cat <<EOF
dwiforge v${DWIFORGE_VERSION} — DWI Processing Pipeline

USAGE
  $(basename "$0") [options]
  $(basename "$0") --slurm [options]
  $(basename "$0") --slurm-worker <sub-ID> --config <path> [options]

SUBJECT SELECTION
  --subject <sub-ID>        Process a single subject
  --subjects <a> <b> ...    Process specific subjects (space-separated)
  --subjects-file <path>    Read subject IDs from file (one per line)
  --exclude <sub-ID>        Exclude a subject (repeatable)

CONFIGURATION
  --config <path>           Explicit config file (default: auto-discover)
  --source <path>           Override [paths].source
  --work <path>             Override [paths].work
  --output <path>           Override [paths].output
  --freesurfer <path>       Override [paths].freesurfer
  --logs <path>             Override [paths].logs
  --qc <path>               Override [paths].qc

STAGE CONTROL
  --skip-stage <name>       Skip a stage (repeatable).
                            Names: qc-bids recon-all preprocessing t1w-prep
                                   epi-correction designer tensor-fitting noddi
                                   response-functions tractography qc-report
                                   connectome-stats
  --only-stage <name>       Run only this stage (repeatable)
  --rerun-stage <name>      Force rerun even if checkpoint exists (repeatable)
  --resume                  Skip subjects/stages with existing checkpoints (default)
  --rerun                   Ignore all checkpoints, reprocess everything

PROCESSING OPTIONS
  --ml-method <method>      Registration method: auto synthmorph ants
  --ml-quick-mode           Fast scipy registration (default)
  --ml-full-mode            PyTorch dense field registration
  --cleanup <tier>          Remove intermediates: 0=keep 1 2 3 4=all (default: 0)
  --no-gpu                  Disable GPU even if available
  --threads <n>             OpenMP threads (default: all cores)
  --parallel <n>            Subjects in parallel (default: 1)
  --dry-run                 Print what would run without executing

SLURM OPTIONS
  --slurm                   Submit a SLURM job array instead of running locally
  --slurm-worker <sub-ID>   Internal: run as a SLURM array worker for one subject

INFORMATION
  -h, --help                Show this help
  --version                 Show version
  --show-config             Show resolved configuration and exit
  --check-space             Check disk space on all locations and exit

EXAMPLES
  # First run — everything in one place:
  $(basename "$0") --source /data/BIDS

  # Your 3-drive setup:
  $(basename "$0") --config ~/projects/my_study/dwiforge.toml

  # Single subject, force rerun of tensor fitting:
  $(basename "$0") --subject sub-001 --rerun-stage tensor-fitting

  # Submit all subjects to SLURM:
  $(basename "$0") --config dwiforge.toml --slurm
EOF
}

# Defaults (overridden by config then CLI)
declare -a ARG_SUBJECTS=()
declare -a ARG_EXCLUDE=()
declare -a ARG_SKIP_STAGES=()
declare -a ARG_ONLY_STAGES=()
declare -a ARG_RERUN_STAGES=()
ARG_CONFIG=""
ARG_SOURCE=""
ARG_WORK=""
ARG_OUTPUT=""
ARG_FREESURFER=""
ARG_LOGS=""
ARG_QC=""
ARG_SUBJECTS_FILE=""
ARG_ML_METHOD=""
ARG_CLEANUP=""
ARG_THREADS=""
ARG_PARALLEL=""
ARG_ML_QUICK=""
ARG_NO_GPU=false
ARG_DRY_RUN=false
ARG_RESUME=true       # default: skip completed
ARG_RERUN=false
ARG_SLURM=false
ARG_SLURM_WORKER=""
ARG_SHOW_CONFIG=false
ARG_CHECK_SPACE=false

_parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)          _usage; exit 0 ;;
            --version)          echo "dwiforge ${DWIFORGE_VERSION}"; exit 0 ;;
            --show-config)      ARG_SHOW_CONFIG=true; shift ;;
            --check-space)      ARG_CHECK_SPACE=true; shift ;;
            --config)           ARG_CONFIG="$2"; shift 2 ;;
            --source)           ARG_SOURCE="$2"; shift 2 ;;
            --work)             ARG_WORK="$2"; shift 2 ;;
            --output)           ARG_OUTPUT="$2"; shift 2 ;;
            --freesurfer)       ARG_FREESURFER="$2"; shift 2 ;;
            --logs)             ARG_LOGS="$2"; shift 2 ;;
            --qc)               ARG_QC="$2"; shift 2 ;;
            --subject)          ARG_SUBJECTS+=("$2"); shift 2 ;;
            --subjects)         shift
                                while [[ $# -gt 0 && "$1" != --* ]]; do
                                    ARG_SUBJECTS+=("$1"); shift
                                done ;;
            --subjects-file)    ARG_SUBJECTS_FILE="$2"; shift 2 ;;
            --exclude)          ARG_EXCLUDE+=("$2"); shift 2 ;;
            --skip-stage)       ARG_SKIP_STAGES+=("$2"); shift 2 ;;
            --only-stage)       ARG_ONLY_STAGES+=("$2"); shift 2 ;;
            --rerun-stage)      ARG_RERUN_STAGES+=("$2"); shift 2 ;;
            --resume)           ARG_RESUME=true; shift ;;
            --rerun)            ARG_RERUN=true; ARG_RESUME=false; shift ;;
            --ml-method)        ARG_ML_METHOD="$2"; shift 2 ;;
            --ml-quick-mode)    ARG_ML_QUICK=true; shift ;;
            --ml-full-mode)     ARG_ML_QUICK=false; shift ;;
            --cleanup)          ARG_CLEANUP="$2"; shift 2 ;;
            --no-gpu)           ARG_NO_GPU=true; shift ;;
            --threads)          ARG_THREADS="$2"; shift 2 ;;
            --parallel)         ARG_PARALLEL="$2"; shift 2 ;;
            --dry-run)          ARG_DRY_RUN=true; shift ;;
            --slurm)            ARG_SLURM=true; shift ;;
            --slurm-worker)     ARG_SLURM_WORKER="$2"; shift 2 ;;
            *)
                log "ERROR" "Unknown argument: $1"
                _usage; exit 1 ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------

_load_config() {
    local parse_args=(
        "${PYTHON_EXECUTABLE}" "${DWIFORGE_ROOT}/scripts/parse_config.py"
    )
    [[ -n "$ARG_CONFIG" ]]      && parse_args+=(--config "$ARG_CONFIG")
    [[ -n "$ARG_SOURCE" ]]      && parse_args+=(--source "$ARG_SOURCE")
    [[ -n "$ARG_WORK" ]]        && parse_args+=(--work "$ARG_WORK")
    [[ -n "$ARG_OUTPUT" ]]      && parse_args+=(--output "$ARG_OUTPUT")
    [[ -n "$ARG_FREESURFER" ]]  && parse_args+=(--freesurfer "$ARG_FREESURFER")
    [[ -n "$ARG_LOGS" ]]        && parse_args+=(--logs "$ARG_LOGS")
    [[ -n "$ARG_QC" ]]          && parse_args+=(--qc "$ARG_QC")
    [[ -n "$ARG_ML_METHOD" ]]   && parse_args+=(--ml-method "$ARG_ML_METHOD")
    [[ -n "$ARG_CLEANUP" ]]     && parse_args+=(--cleanup-tier "$ARG_CLEANUP")
    [[ -n "$ARG_THREADS" ]]     && parse_args+=(--omp-threads "$ARG_THREADS")
    [[ -n "$ARG_PARALLEL" ]]    && parse_args+=(--parallel-subjects "$ARG_PARALLEL")
    [[ "$ARG_NO_GPU" == true ]] && parse_args+=(--no-gpu)
    [[ "$ARG_CHECK_SPACE" == true ]] && parse_args+=(--check-space)

    # Evaluate the exports — this populates all DWIFORGE_* variables
    eval "$("${parse_args[@]}")"
}

_show_config() {
    cat <<EOF
dwiforge v${DWIFORGE_VERSION} — Resolved Configuration
═══════════════════════════════════════════════════════
Config file:    ${DWIFORGE_CONFIG:-not found — using defaults}

Paths:
  source:       ${DWIFORGE_DIR_SOURCE}
  work:         ${DWIFORGE_DIR_WORK}
  output:       ${DWIFORGE_DIR_OUTPUT}
  freesurfer:   ${DWIFORGE_DIR_FREESURFER}
  logs:         ${DWIFORGE_DIR_LOGS}
  qc:           ${DWIFORGE_DIR_QC}

Options:
  cleanup tier: ${DWIFORGE_CLEANUP_TIER}
  ml method:    ${DWIFORGE_ML_METHOD}
  quick mode:   ${DWIFORGE_ML_QUICK_MODE}
  gpu:          ${DWIFORGE_USE_GPU}
  threads:      ${DWIFORGE_OMP_THREADS}
  parallel:     ${DWIFORGE_PARALLEL_SUBJECTS}

Stages enabled:
  preprocessing:  ${DWIFORGE_RUN_PREPROCESSING}
  registration:   ${DWIFORGE_RUN_REGISTRATION}
  refinement:     ${DWIFORGE_RUN_REFINEMENT}
  tensor fitting: ${DWIFORGE_RUN_TENSOR_FITTING}
  noddi:          ${DWIFORGE_RUN_NODDI}
  connectivity:   ${DWIFORGE_RUN_CONNECTIVITY}
  qc report:      ${DWIFORGE_RUN_QC_REPORT}
  connectome stats: ${DWIFORGE_RUN_CONNECTOME_STATS:-true}

Runtime:
  GPU available:      ${GPU_AVAILABLE:-unknown}
  PyTorch available:  ${PYTORCH_AVAILABLE:-unknown}
  AMICO available:    ${AMICO_AVAILABLE:-unknown}
  SynthMorph:         ${SYNTHMORPH_AVAILABLE:-unknown}
  Python:             ${PYTHON_EXECUTABLE}
═══════════════════════════════════════════════════════
EOF
}

# ---------------------------------------------------------------------------
# Subject list resolution
# ---------------------------------------------------------------------------

_build_subject_list() {
    declare -ga SUBJECTS=()

    # Explicit list from CLI
    if [[ ${#ARG_SUBJECTS[@]} -gt 0 ]]; then
        SUBJECTS=("${ARG_SUBJECTS[@]}")
    # From file
    elif [[ -n "$ARG_SUBJECTS_FILE" ]]; then
        if [[ ! -f "$ARG_SUBJECTS_FILE" ]]; then
            log "ERROR" "Subjects file not found: ${ARG_SUBJECTS_FILE}"
            exit 1
        fi
        mapfile -t SUBJECTS < <(grep -v '^\s*#' "$ARG_SUBJECTS_FILE" | grep -v '^\s*$')
    # Auto-discover from BIDS source
    else
        mapfile -t SUBJECTS < <(discover_subjects)
        if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
            log "ERROR" "No sub-*/dwi/ directories found under ${DWIFORGE_DIR_SOURCE}"
            exit 1
        fi
        log "INFO" "Discovered ${#SUBJECTS[@]} subjects"
    fi

    # Apply exclusions
    if [[ ${#ARG_EXCLUDE[@]} -gt 0 ]]; then
        local filtered=()
        for sub in "${SUBJECTS[@]}"; do
            local excluded=false
            for excl in "${ARG_EXCLUDE[@]}"; do
                [[ "$sub" == "$excl" ]] && excluded=true && break
            done
            [[ "$excluded" == false ]] && filtered+=("$sub")
        done
        local n_excluded=$(( ${#SUBJECTS[@]} - ${#filtered[@]} ))
        SUBJECTS=("${filtered[@]}")
        log "INFO" "Excluded ${n_excluded} subject(s); ${#SUBJECTS[@]} remaining"
    fi
}

# ---------------------------------------------------------------------------
# Stage dispatch
# ---------------------------------------------------------------------------

# Ordered stage list — defines canonical names and script mappings
# Stage 08 (response-functions) runs per-subject, then the orchestrator
# calls _run_responsemean before any subject proceeds to stage 09.
declare -A STAGE_SCRIPTS=(
    [qc-bids]="stages/00_qc_bids.sh"
    [recon-all]="stages/01_recon_all.sh"
    [preprocessing]="stages/02_preprocessing.sh"
    [t1w-prep]="stages/03_t1w_prep.sh"
    [epi-correction]="stages/04_epi_correction.sh"
    [designer]="stages/05_designer.sh"
    [tensor-fitting]="stages/06_tensor_fitting.sh"
    [noddi]="stages/07_noddi.sh"
    [response-functions]="stages/08_response_functions.sh"
    [tractography]="stages/09_tractography.sh"
    [qc-report]="stages/10_qc_report.sh"
    [connectome-stats]="stages/11_connectome_stats.sh"
)
STAGE_ORDER=(
    qc-bids
    recon-all
    preprocessing
    t1w-prep
    epi-correction
    designer
    tensor-fitting
    noddi
    response-functions
    tractography
    qc-report
    connectome-stats
)

# Split at the group barrier (response-functions -> responsemean -> tractography).
# Used for two-pass local execution when multiple subjects share group responses:
# every subject must finish PHASE1 before responsemean can run, and no subject
# can start PHASE2 until responsemean has produced group response functions.
STAGE_ORDER_PHASE1=("${STAGE_ORDER[@]:0:9}")   # qc-bids .. response-functions
STAGE_ORDER_PHASE2=("${STAGE_ORDER[@]:9}")     # tractography .. connectome-stats

# Does this run touch any stage that depends on group responses?
_phase2_enabled() {
    local stage
    for stage in "${STAGE_ORDER_PHASE2[@]}"; do
        _stage_enabled "$stage" && return 0
    done
    return 1
}

# Map stage name to DWIFORGE_RUN_* config variable
declare -A STAGE_CONFIG_VAR=(
    [qc-bids]="DWIFORGE_RUN_QC_BIDS"
    [recon-all]="DWIFORGE_RUN_RECON_ALL"
    [preprocessing]="DWIFORGE_RUN_PREPROCESSING"
    [t1w-prep]="DWIFORGE_RUN_T1W_PREP"
    [epi-correction]="DWIFORGE_RUN_EPI_CORRECTION"
    [designer]="DWIFORGE_RUN_DESIGNER"
    [tensor-fitting]="DWIFORGE_RUN_TENSOR_FITTING"
    [noddi]="DWIFORGE_RUN_NODDI"
    [response-functions]="DWIFORGE_RUN_RESPONSE_FUNCTIONS"
    [tractography]="DWIFORGE_RUN_TRACTOGRAPHY"
    [qc-report]="DWIFORGE_RUN_QC_REPORT"
    [connectome-stats]="DWIFORGE_RUN_CONNECTOME_STATS"
)

# ---------------------------------------------------------------------------
# Group barrier: responsemean
# ---------------------------------------------------------------------------
# Called by the orchestrator after all subjects complete stage 08.
# Averages per-subject response functions into group response functions
# used by stage 09 (tractography / ss3t_csd_beta1).

_run_responsemean() {
    local group_resp_dir="${DWIFORGE_DIR_WORK}/group/responses"
    local group_avg_dir="${DWIFORGE_DIR_WORK}/group"
    local log_file="${DWIFORGE_DIR_LOGS}/responsemean.log"

    mkdir -p "$group_avg_dir"

    log "INFO" "Running responsemean across all subjects"

    # Collect per-subject response files
    local wm_files=()  gm_files=()  csf_files=()
    while IFS= read -r -d '' f; do
        wm_files+=("$f")
    done < <(find "$group_resp_dir" -name "response_wm.txt" -print0 | sort -z)

    while IFS= read -r -d '' f; do
        gm_files+=("$f")
    done < <(find "$group_resp_dir" -name "response_gm.txt" -print0 | sort -z)

    while IFS= read -r -d '' f; do
        csf_files+=("$f")
    done < <(find "$group_resp_dir" -name "response_csf.txt" -print0 | sort -z)

    local n_wm=${#wm_files[@]}
    log "INFO" "  Found ${n_wm} WM response files"

    if [[ "$n_wm" -eq 0 ]]; then
        log "ERROR" "No response functions found in ${group_resp_dir}"
        log "ERROR" "Run stage 08 for all subjects before calling responsemean"
        return 1
    fi

    # Run responsemean for each tissue type.
    # responsemean is in standard MRtrix3 3.0.8+ — use it directly.
    # Fall back to MRtrix3Tissue only if not found on PATH.
    local rm_bin
    rm_bin=$(command -v responsemean 2>/dev/null ||              echo "${MRTRIX3TISSUE_HOME:-/opt/mrtrix3tissue}/bin/responsemean")
    log "INFO" "  Using responsemean: ${rm_bin}"

    for tissue in wm gm csf; do
        local -n files_ref="${tissue}_files"
        env "PYTHONPATH=${MRTRIX_PYTHON_PATH}"             "${rm_bin}"             "${files_ref[@]}"             "${group_avg_dir}/group_response_${tissue}.txt"             -force
        log "OK" "  Group ${tissue^^} response: group_response_${tissue}.txt"
    done

    log "OK" "responsemean complete — group responses in ${group_avg_dir}"

    # Write sentinel so stage 09 knows group responses are ready
    touch "${group_avg_dir}/responsemean.done"
    return 0
}

_stage_enabled() {
    local stage="$1"

    # --only-stage overrides everything
    if [[ ${#ARG_ONLY_STAGES[@]} -gt 0 ]]; then
        local found=false
        for s in "${ARG_ONLY_STAGES[@]}"; do
            [[ "$s" == "$stage" ]] && found=true && break
        done
        [[ "$found" == true ]]
        return
    fi

    # --skip-stage
    for s in "${ARG_SKIP_STAGES[@]}"; do
        [[ "$s" == "$stage" ]] && return 1
    done

    # Config flag
    local cfg_var="${STAGE_CONFIG_VAR[$stage]:-}"
    if [[ -n "$cfg_var" && "${!cfg_var:-true}" == "false" ]]; then
        return 1
    fi

    return 0
}

_stage_needs_run() {
    local stage="$1"
    local sub="$2"

    # --rerun clears all checkpoints
    if [[ "$ARG_RERUN" == true ]]; then
        return 0
    fi

    # --rerun-stage for this specific stage
    for s in "${ARG_RERUN_STAGES[@]}"; do
        if [[ "$s" == "$stage" ]]; then
            checkpoint_clear "$sub" "$stage"
            return 0
        fi
    done

    # Skip if checkpoint exists and --resume is active (default)
    if [[ "$ARG_RESUME" == true ]] && checkpoint_check "$sub" "$stage"; then
        return 1  # already done
    fi

    return 0  # needs to run
}

_run_stage() {
    local stage="$1"
    local sub="$2"
    local script="${DWIFORGE_ROOT}/${STAGE_SCRIPTS[$stage]}"

    if [[ ! -f "$script" ]]; then
        log_sub "WARN" "$sub" "Stage script not found: ${script} — skipping"
        return 0
    fi

    if [[ "$ARG_DRY_RUN" == true ]]; then
        log_sub "INFO" "$sub" "[dry-run] Would run: ${stage}"
        return 0
    fi

    local log_file="${DWIFORGE_DIR_LOGS}/${sub}/${stage}.log"
    local timer
    timer=$(timer_start)

    log_stage_start "$stage" "$sub"

    # Each stage script is executed as a subprocess.
    # It reads all configuration from DWIFORGE_* env vars.
    # stdout+stderr go to the stage log file AND the terminal.
    if bash "$script" "$sub" 2>&1 | tee -a "$log_file" | \
        while IFS= read -r line; do
            log "INFO" "[${sub}][${stage}] ${line}"
        done; then
        local elapsed
        elapsed=$(timer_elapsed "$timer")
        checkpoint_set "$sub" "$stage"
        log_stage_end "$stage" "$sub" "$elapsed"
        return 0
    else
        local rc=${PIPESTATUS[0]}
        log_sub "ERROR" "$sub" "Stage failed: ${stage} (exit ${rc})"
        log_sub "ERROR" "$sub" "Log: ${log_file}"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Single-subject processing
# ---------------------------------------------------------------------------

_process_subject() {
    local sub="$1"
    # phase: full (default) runs every stage; phase1/phase2 run only the
    # stages on that side of the response-functions group barrier. Callers
    # that need the barrier (multiple local subjects with any post-barrier
    # stage enabled) run phase1 for every subject, call _run_responsemean
    # once, then run phase2 for every subject that survived phase1. See
    # main()'s local-processing section.
    local phase="${2:-full}"

    local stage_list=()
    case "$phase" in
        phase1) stage_list=("${STAGE_ORDER_PHASE1[@]}") ;;
        phase2) stage_list=("${STAGE_ORDER_PHASE2[@]}") ;;
        full)   stage_list=("${STAGE_ORDER[@]}") ;;
        *)      log_sub "ERROR" "$sub" "Internal error: unknown phase '${phase}'"; return 1 ;;
    esac

    # Per-subject log file (all stages for this subject). The containing
    # directory must exist before anything is logged — dirs_init() would
    # normally create it, but that happens later, and logging to a
    # nonexistent path fails. In sequential mode that failure was silently
    # swallowed (calling _process_subject as an `if` condition suspends
    # errexit for the whole call), but backgrounding it for parallel
    # execution does not get that same accidental protection, so create the
    # directory explicitly up front rather than relying on call-site quirks.
    local sub_log="${DWIFORGE_DIR_LOGS}/${sub}/subject.log"
    mkdir -p "${DWIFORGE_DIR_LOGS}/${sub}"
    export DWIFORGE_LOG_FILE="$sub_log"
    # Only rotate at the start of a subject's first phase, so phase2 doesn't
    # truncate the log phase1 just wrote.
    [[ "$phase" != "phase2" ]] && rotate_log "$sub_log"

    log_sub "INFO" "$sub" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [[ "$phase" == "phase2" ]]; then
        log_sub "INFO" "$sub" "Resuming subject processing (post-barrier stages)"
    else
        log_sub "INFO" "$sub" "Starting subject processing"
    fi

    # Validate source data exists (skip re-check on phase2 — already passed)
    if [[ "$phase" != "phase2" ]]; then
        if [[ ! -d "${DWIFORGE_DIR_SOURCE}/${sub}/dwi" ]]; then
            log_sub "ERROR" "$sub" "No DWI directory found: ${DWIFORGE_DIR_SOURCE}/${sub}/dwi"
            return 1
        fi

        # Create directories
        dirs_init "$sub" || return 1

        # Check disk space (warns but does not abort)
        disk_check_all "$sub"

        # Clear checkpoints if --rerun
        [[ "$ARG_RERUN" == true ]] && checkpoint_clear_all "$sub"
    fi

    # Run stages in order
    local n_run=0 n_skip=0 stage_failed=false

    for stage in "${stage_list[@]}"; do
        if ! _stage_enabled "$stage"; then
            log_sub "DEBUG" "$sub" "Stage disabled: ${stage}"
            continue
        fi

        if ! _stage_needs_run "$stage" "$sub"; then
            log_stage_skip "$stage" "$sub"
            (( n_skip++ )) || true
            continue
        fi

        if ! _run_stage "$stage" "$sub"; then
            stage_failed=true
            # Fatal stages: abort subject immediately
            if [[ "$stage" =~ ^(preprocessing|designer|epi-correction)$ ]]; then
                log_sub "ERROR" "$sub" "Fatal stage failed — aborting subject"
                break
            fi
            # Advisory stages: log and continue
            log_sub "WARN" "$sub" "Stage ${stage} failed — continuing with remaining stages"
        else
            (( n_run++ )) || true
        fi
    done

    # Cleanup only once the subject has finished its final requested phase —
    # phase1 output (e.g. dwi_preprocessed.mif, per-subject responses) is
    # still needed by phase2, so don't clean up until phase2 (or a full,
    # single-pass run) completes.
    if [[ "$phase" != "phase1" ]]; then
        cleanup_subject "$sub" "${DWIFORGE_CLEANUP_TIER:-0}"
    fi

    unset DWIFORGE_LOG_FILE

    if [[ "$stage_failed" == true ]]; then
        log_sub "WARN" "$sub" "Completed with failures (${n_run} stages run, ${n_skip} skipped)"
        return 1
    fi

    log_sub "OK" "$sub" "All stages complete (${n_run} run, ${n_skip} skipped)"
    return 0
}

# ---------------------------------------------------------------------------
# Batch execution — runs a list of subjects through one phase, sequentially
# or in parallel, with correct per-subject success/failure accounting.
# ---------------------------------------------------------------------------
# Populates the caller's BATCH_OK / BATCH_FAILED arrays (declare -a in the
# caller before invoking). Does not touch n_success/n_fail/failed_subjects
# directly so it can be called more than once per run (phase1, then phase2).
#
# Parallel execution runs in fixed-size batches of ${parallel} subjects:
# each batch is fully launched, then every PID in that batch is waited on
# individually so its real exit code is captured (bash has no portable way
# to learn *which* job `wait -n` just reaped, so a rolling window can't
# attribute exit codes to subjects — fixed batches sidestep that).
_run_subject_batch() {
    local phase="$1"; shift
    local -a subjects=("$@")

    BATCH_OK=()
    BATCH_FAILED=()

    local parallel="${DWIFORGE_PARALLEL_SUBJECTS:-1}"
    local n=${#subjects[@]}

    if [[ "$n" -eq 0 ]]; then
        return 0
    fi

    if [[ "$parallel" -gt 1 ]]; then
        local i=0
        while [[ $i -lt $n ]]; do
            local -a batch_pids=()
            declare -A batch_subs=()
            local batch_end=$(( i + parallel > n ? n : i + parallel ))
            for (( ; i < batch_end; i++ )); do
                local sub="${subjects[$i]}"
                _process_subject "$sub" "$phase" &
                local pid=$!
                batch_pids+=("$pid")
                batch_subs["$pid"]="$sub"
            done
            local pid
            for pid in "${batch_pids[@]}"; do
                if wait "$pid"; then
                    BATCH_OK+=("${batch_subs[$pid]}")
                else
                    BATCH_FAILED+=("${batch_subs[$pid]}")
                fi
            done
        done
    else
        local sub
        for sub in "${subjects[@]}"; do
            if _process_subject "$sub" "$phase"; then
                BATCH_OK+=("$sub")
            else
                BATCH_FAILED+=("$sub")
            fi
        done
    fi
}

# ---------------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------------

_submit_slurm() {
    local config_path="${DWIFORGE_CONFIG:-}"
    if [[ -z "$config_path" ]]; then
        # Write a resolved config to a temp file so workers get consistent paths
        config_path="${DWIFORGE_DIR_LOGS}/dwiforge_resolved.toml"
        log "INFO" "Writing resolved config for SLURM workers: ${config_path}"
        mkdir -p "$(dirname "$config_path")"
        "${PYTHON_EXECUTABLE}" "${DWIFORGE_ROOT}/scripts/write_resolved_config.py" \
            > "$config_path"
    fi

    local array_spec
    array_spec="1-${#SUBJECTS[@]}"
    if [[ "${DWIFORGE_SLURM_MAX_SIMULTANEOUS:-0}" -gt 0 ]]; then
        array_spec="${array_spec}%${DWIFORGE_SLURM_MAX_SIMULTANEOUS}"
    fi

    # Write subjects list for worker lookup by SLURM_ARRAY_TASK_ID
    local subjects_file="${DWIFORGE_DIR_LOGS}/slurm_subjects.txt"
    printf '%s\n' "${SUBJECTS[@]}" > "$subjects_file"

    local script_path="${DWIFORGE_DIR_LOGS}/slurm_submit.sh"
    {
        echo "#!/usr/bin/env bash"
        echo "#SBATCH --job-name=dwiforge"
        echo "#SBATCH --array=${array_spec}"
        echo "#SBATCH --partition=${DWIFORGE_SLURM_PARTITION}"
        echo "#SBATCH --mem=${DWIFORGE_SLURM_MEM_GB}G"
        echo "#SBATCH --cpus-per-task=${DWIFORGE_SLURM_CPUS_PER_TASK}"
        echo "#SBATCH --time=${DWIFORGE_SLURM_TIME_LIMIT}"
        echo "#SBATCH --output=${DWIFORGE_DIR_LOGS}/slurm_%A_%a.out"
        echo "#SBATCH --error=${DWIFORGE_DIR_LOGS}/slurm_%A_%a.err"
        if [[ -n "${DWIFORGE_SLURM_ACCOUNT:-}" ]]; then
            echo "#SBATCH --account=${DWIFORGE_SLURM_ACCOUNT}"
        fi
        if [[ -n "${DWIFORGE_SLURM_EXTRA_DIRECTIVES:-}" ]]; then
            echo "${DWIFORGE_SLURM_EXTRA_DIRECTIVES}"
        fi
        echo ""
        echo "set -euo pipefail"
        echo ""
        echo "# Look up subject for this array task"
        echo "SUB=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" '${subjects_file}')"
        echo "[[ -z \"\$SUB\" ]] && echo 'No subject for task \${SLURM_ARRAY_TASK_ID}' && exit 1"
        echo ""
        echo "exec '${DWIFORGE_ROOT}/dwiforge.sh' \\"
        echo "    --slurm-worker \"\$SUB\" \\"
        echo "    --config '${config_path}'"
    } > "$script_path"
    chmod +x "$script_path"

    if [[ "$ARG_DRY_RUN" == true ]]; then
        log "INFO" "[dry-run] Would submit: ${script_path}"
        log "INFO" "[dry-run] Array: ${array_spec} (${#SUBJECTS[@]} subjects)"
        cat "$script_path"
        return 0
    fi

    local job_id
    job_id=$(sbatch --parsable "$script_path")
    log "OK" "Submitted SLURM job array ${job_id} (${#SUBJECTS[@]} subjects)"
    log "INFO" "Monitor with: squeue -j ${job_id}"
    log "INFO" "Logs: ${DWIFORGE_DIR_LOGS}/slurm_${job_id}_*.out"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    _parse_args "$@"

    # Resolve Python before loading config (needed for parse_config.py)
    # Use a minimal resolution here; full setup happens after config load
    PYTHON_EXECUTABLE="$(command -v python3 2>/dev/null || echo python3)"
    export PYTHON_EXECUTABLE

    # Load and export all configuration
    _load_config

    # Full environment setup (FSL, FreeSurfer, GPU, etc.)
    setup_environment

    # Set up pipeline-level log
    export DWIFORGE_LOG_FILE="${DWIFORGE_DIR_LOGS}/pipeline.log"
    mkdir -p "${DWIFORGE_DIR_LOGS}"
    rotate_log "${DWIFORGE_LOG_FILE}"

    log "INFO" "dwiforge v${DWIFORGE_VERSION} starting"
    log "INFO" "Root: ${DWIFORGE_ROOT}"

    # Information-only modes
    if [[ "$ARG_SHOW_CONFIG" == true ]]; then
        _show_config
        exit 0
    fi
    if [[ "$ARG_CHECK_SPACE" == true ]]; then
        disk_check_all
        exit 0
    fi

    # Validate storage layout
    dirs_validate || exit 1

    # SLURM worker mode — single subject, called by sbatch
    if [[ -n "$ARG_SLURM_WORKER" ]]; then
        SUBJECTS=("$ARG_SLURM_WORKER")
        log "INFO" "SLURM worker mode: subject ${ARG_SLURM_WORKER} (job ${SLURM_JOB_ID:-local})"
        _process_subject "$ARG_SLURM_WORKER"
        exit $?
    fi

    # Build subject list
    _build_subject_list

    log "INFO" "Subjects to process: ${#SUBJECTS[@]}"
    [[ "$ARG_DRY_RUN" == true ]] && log "INFO" "DRY RUN — no files will be written"

    # SLURM submission mode
    if [[ "$ARG_SLURM" == true ]]; then
        _submit_slurm
        exit 0
    fi

    # Local processing
    local n_total=${#SUBJECTS[@]}
    local n_success=0 n_fail=0
    declare -a failed_subjects=()
    local parallel="${DWIFORGE_PARALLEL_SUBJECTS:-1}"

    if [[ "$parallel" -gt 1 ]]; then
        log "INFO" "Parallel mode: up to ${parallel} subjects simultaneously"
    fi

    declare -a BATCH_OK BATCH_FAILED

    if [[ "$n_total" -gt 1 ]] && _phase2_enabled; then
        # Group barrier required: stages 09-11 need group-averaged response
        # functions (stage 08, all subjects) before any subject can run them.
        # Running every subject through the full pipeline one at a time (the
        # old single-pass loop) would let subject 1 reach tractography before
        # subject 2 has even started — so this splits into two passes with
        # responsemean run once, in between, across all subjects.
        log "INFO" "Multiple subjects with post-response-functions stages enabled"
        log "INFO" "Running in two passes: (1) qc-bids..response-functions, responsemean, (2) tractography..connectome-stats"

        log "INFO" "── Pass 1/2: qc-bids .. response-functions ──"
        _run_subject_batch "phase1" "${SUBJECTS[@]}"
        local -a phase1_ok=("${BATCH_OK[@]}")
        for sub in "${BATCH_FAILED[@]}"; do
            (( n_fail++ )) || true
            failed_subjects+=("$sub")
        done

        if [[ ${#phase1_ok[@]} -eq 0 ]]; then
            log "ERROR" "No subjects completed pass 1 — skipping responsemean and pass 2"
        elif [[ "$ARG_DRY_RUN" == true ]]; then
            log "INFO" "[dry-run] Would run responsemean, then pass 2 for: ${phase1_ok[*]}"
            for sub in "${phase1_ok[@]}"; do
                (( n_success++ )) || true
            done
        else
            log "INFO" "All subjects that reached stage 08 completed it — running responsemean"
            if ! _run_responsemean; then
                log "ERROR" "responsemean failed — pass 2 (tractography onward) cannot run"
                for sub in "${phase1_ok[@]}"; do
                    (( n_fail++ )) || true
                    failed_subjects+=("$sub")
                done
            else
                log "INFO" "── Pass 2/2: tractography .. connectome-stats ──"
                _run_subject_batch "phase2" "${phase1_ok[@]}"
                for sub in "${BATCH_OK[@]}"; do
                    (( n_success++ )) || true
                done
                for sub in "${BATCH_FAILED[@]}"; do
                    (( n_fail++ )) || true
                    failed_subjects+=("$sub")
                done
            fi
        fi
    else
        # Single subject, or no post-barrier stage enabled — no group
        # dependency to coordinate, so run every stage in one pass per subject.
        _run_subject_batch "full" "${SUBJECTS[@]}"
        for sub in "${BATCH_OK[@]}"; do
            (( n_success++ )) || true
        done
        for sub in "${BATCH_FAILED[@]}"; do
            (( n_fail++ )) || true
            failed_subjects+=("$sub")
        done
    fi

    # Final summary
    log "INFO" "═══════════════════════════════════════"
    log "INFO" "Pipeline complete"
    log "INFO" "  Total:   ${n_total}"
    log "INFO" "  Success: ${n_success}"
    log "INFO" "  Failed:  ${n_fail}"
    if [[ ${#failed_subjects[@]} -gt 0 ]]; then
        log "WARN" "  Failed subjects: ${failed_subjects[*]}"
        log "WARN" "  Rerun with: --subjects ${failed_subjects[*]} --resume"
    fi
    log "INFO" "  Outputs: ${DWIFORGE_DIR_OUTPUT}"
    log "INFO" "  QC:      ${DWIFORGE_DIR_QC}"
    log "INFO" "  Logs:    ${DWIFORGE_DIR_LOGS}"
    log "INFO" "═══════════════════════════════════════"

    return $(( n_fail > 0 ? 1 : 0 ))
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    trap 'log "WARN" "Pipeline interrupted"; exit 130' INT TERM
    main "$@"
    exit $?
fi
