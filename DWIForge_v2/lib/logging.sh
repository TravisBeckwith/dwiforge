#!/usr/bin/env bash
# lib/logging.sh — dwiforge logging functions
#
# Source this file; do not execute it directly.
# Provides:
#   log LEVEL message ...
#   log_sub LEVEL subject message ...   (prefixes [sub-XXX])
#   checkpoint_set subject stage
#   checkpoint_check subject stage      (returns 0 if done, 1 if not)
#   checkpoint_clear subject stage
#
# Log levels: DEBUG INFO OK WARN ERROR ML
# Output:
#   - Always to ${DWIFORGE_LOG_FILE} if set
#   - To stderr for WARN/ERROR
#   - To stdout for all others (with color if terminal)

# ---------------------------------------------------------------------------
# Color codes — only when stdout is a terminal
# ---------------------------------------------------------------------------

if [[ -t 1 ]]; then
    _CLR_RESET='\033[0m'
    _CLR_GRAY='\033[0;90m'
    _CLR_GREEN='\033[0;32m'
    _CLR_YELLOW='\033[0;33m'
    _CLR_RED='\033[0;31m'
    _CLR_CYAN='\033[0;36m'
    _CLR_BLUE='\033[0;34m'
    _CLR_BOLD='\033[1m'
else
    _CLR_RESET=''
    _CLR_GRAY=''
    _CLR_GREEN=''
    _CLR_YELLOW=''
    _CLR_RED=''
    _CLR_CYAN=''
    _CLR_BLUE=''
    _CLR_BOLD=''
fi

# ---------------------------------------------------------------------------
# Core log function
# ---------------------------------------------------------------------------

log() {
    local level="${1:-INFO}"
    shift
    local message="$*"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"

    # Level formatting
    local label color fd
    case "$level" in
        DEBUG)  label="DEBUG" color="${_CLR_GRAY}"   fd=1 ;;
        INFO)   label="INFO " color=""               fd=1 ;;
        OK)     label="OK   " color="${_CLR_GREEN}"  fd=1 ;;
        WARN)   label="WARN " color="${_CLR_YELLOW}" fd=2 ;;
        ERROR)  label="ERROR" color="${_CLR_RED}"    fd=2 ;;
        ML)     label="ML   " color="${_CLR_CYAN}"   fd=1 ;;
        STAGE)  label="STAGE" color="${_CLR_BLUE}"   fd=1 ;;
        *)      label="$level" color=""              fd=1 ;;
    esac

    local formatted="${ts} [${label}] ${message}"

    # Terminal output with color
    if [[ -n "$color" ]]; then
        printf "${color}%s${_CLR_RESET}\n" "$formatted" >&$fd
    else
        printf "%s\n" "$formatted" >&$fd
    fi

    # File output (no color codes)
    if [[ -n "${DWIFORGE_LOG_FILE:-}" ]]; then
        printf "%s\n" "$formatted" >> "${DWIFORGE_LOG_FILE}"
    fi
}

# Convenience: log with subject prefix
log_sub() {
    local level="$1"
    local sub="$2"
    shift 2
    log "$level" "[${sub}] $*"
}

# ---------------------------------------------------------------------------
# Stage banner — printed at the start of each stage script
# ---------------------------------------------------------------------------

log_stage_start() {
    local stage_name="$1"
    local sub="${2:-}"
    local prefix="${sub:+[${sub}] }"
    log "STAGE" "${_CLR_BOLD}${prefix}━━ Starting: ${stage_name} ━━${_CLR_RESET}"
}

log_stage_end() {
    local stage_name="$1"
    local sub="${2:-}"
    local elapsed="${3:-}"
    local prefix="${sub:+[${sub}] }"
    local time_str="${elapsed:+ (${elapsed}s)}"
    log "OK" "${prefix}━━ Completed: ${stage_name}${time_str} ━━"
}

log_stage_skip() {
    local stage_name="$1"
    local sub="${2:-}"
    log "INFO" "${sub:+[${sub}] }Skipping ${stage_name} — checkpoint exists"
}

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
# Checkpoint files live at:
#   ${DWIFORGE_DIR_WORK}/<sub>/checkpoints/<stage>.done
# They record the completion timestamp, config hash, and dwiforge version
# so a --rerun-force can detect stale checkpoints if the config changed.

_checkpoint_dir() {
    local sub="$1"
    echo "${DWIFORGE_DIR_WORK}/${sub}/checkpoints"
}

_checkpoint_file() {
    local sub="$1"
    local stage="$2"
    echo "$(_checkpoint_dir "$sub")/${stage}.done"
}

checkpoint_set() {
    local sub="$1"
    local stage="$2"
    local dir
    dir="$(_checkpoint_dir "$sub")"
    mkdir -p "$dir"
    {
        echo "stage=${stage}"
        echo "subject=${sub}"
        echo "completed=$(date --iso-8601=seconds)"
        echo "dwiforge_version=${DWIFORGE_VERSION:-unknown}"
        echo "config=${DWIFORGE_CONFIG:-}"
    } > "$(_checkpoint_file "$sub" "$stage")"
    log_sub "DEBUG" "$sub" "Checkpoint set: ${stage}"
}

checkpoint_check() {
    local sub="$1"
    local stage="$2"
    [[ -f "$(_checkpoint_file "$sub" "$stage")" ]]
}

checkpoint_clear() {
    local sub="$1"
    local stage="$2"
    local f
    f="$(_checkpoint_file "$sub" "$stage")"
    if [[ -f "$f" ]]; then
        rm -f "$f"
        log_sub "DEBUG" "$sub" "Checkpoint cleared: ${stage}"
    fi
}

# Clear all checkpoints for a subject (force full rerun)
checkpoint_clear_all() {
    local sub="$1"
    local dir
    dir="$(_checkpoint_dir "$sub")"
    if [[ -d "$dir" ]]; then
        rm -f "${dir}"/*.done
        log_sub "DEBUG" "$sub" "All checkpoints cleared"
    fi
}

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

timer_start() {
    echo "$SECONDS"
}

timer_elapsed() {
    local start="$1"
    echo $(( SECONDS - start ))
}
