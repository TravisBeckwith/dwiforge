#!/usr/bin/env bash
# lib/env_setup.sh — dwiforge environment setup
#
# Sets up the neuroimaging toolchain (FSL, FreeSurfer, MRtrix3, ANTs)
# and the Python environment. Sourced once at pipeline startup.
#
# Exports:
#   PYTHON_EXECUTABLE   Resolved Python binary
#   GPU_AVAILABLE       true/false
#   PYTORCH_AVAILABLE   true/false
#   AMICO_AVAILABLE     true/false
#   SYNTHMORPH_AVAILABLE true/false

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------

# Default location for dwiforge's isolated Python dependencies.
# Install with: pip install --target "${DWIFORGE_DEPS_DIR}" dipy>=1.12.0 ...
# This directory is prepended to PYTHONPATH so our scripts see dipy>=1.12
# while DESIGNER (a separate process) keeps its pinned dipy==1.9.0.
DWIFORGE_DEPS_DIR="${DWIFORGE_DEPS_DIR:-${HOME}/.local/share/dwiforge/deps}"

setup_python() {
    # ---- Resolve base Python executable ----
    # Priority: active venv → active conda → system python3 → python
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        export PYTHON_EXECUTABLE="${VIRTUAL_ENV}/bin/python"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        export PYTHON_EXECUTABLE="${CONDA_PREFIX}/bin/python"
    else
        export PYTHON_EXECUTABLE
        PYTHON_EXECUTABLE="$(command -v python3 2>/dev/null \
                          || command -v python 2>/dev/null \
                          || echo python3)"
    fi

    # SLURM: activate venv if configured
    if [[ -n "${DWIFORGE_SLURM_VENV_PATH:-}" && \
          -f "${DWIFORGE_SLURM_VENV_PATH}/bin/activate" ]]; then
        # shellcheck disable=SC1090
        set +u
        source "${DWIFORGE_SLURM_VENV_PATH}/bin/activate"
        set -u
        export PYTHON_EXECUTABLE="${DWIFORGE_SLURM_VENV_PATH}/bin/python"
        log "DEBUG" "Activated venv: ${DWIFORGE_SLURM_VENV_PATH}"
    fi

    # ---- Prepend dwiforge deps directory to PYTHONPATH ----
    # This gives our scripts priority over DESIGNER's pinned packages
    # (specifically dipy>=1.12 vs DESIGNER's dipy==1.9.0) without
    # requiring a virtualenv or activation step.
    # DESIGNER is called as a subprocess via $DESIGNER_BIN and never
    # inherits this PYTHONPATH prefix — it imports from user site-packages.
    if [[ -d "${DWIFORGE_DEPS_DIR}" ]]; then
        # Check dipy version in deps dir to confirm it's populated
        local deps_dipy_ver
        deps_dipy_ver=$(
            PYTHONPATH="${DWIFORGE_DEPS_DIR}" \
            "${PYTHON_EXECUTABLE}" -c \
            "import dipy; print(dipy.__version__)" 2>/dev/null || echo ""
        )
        if [[ -n "$deps_dipy_ver" ]]; then
            export PYTHONPATH="${DWIFORGE_DEPS_DIR}:${PYTHONPATH:-}"
            log "DEBUG" "dwiforge deps: ${DWIFORGE_DEPS_DIR} (dipy ${deps_dipy_ver})"
        else
            log "WARN" "dwiforge deps dir exists but dipy not found in it"
            log "WARN" "Run: pip install --target '${DWIFORGE_DEPS_DIR}' dipy>=1.12.0"
        fi
    else
        log "WARN" "dwiforge deps dir not found: ${DWIFORGE_DEPS_DIR}"
        log "WARN" "Run bootstrap to create it:"
        log "WARN" "  pip install --target '${DWIFORGE_DEPS_DIR}' \\"
        log "WARN" "    dipy>=1.12.0 nibabel>=5.0 numpy>=1.24 \\"
        log "WARN" "    scipy>=1.10 scikit-learn>=1.3 tqdm>=4.65"
    fi

    log "INFO" "Python: ${PYTHON_EXECUTABLE} ($("$PYTHON_EXECUTABLE" --version 2>&1))"
}

# ---------------------------------------------------------------------------
# FSL
# ---------------------------------------------------------------------------

setup_fsl() {
    export FSLDIR="${FSLDIR:-/usr/local/fsl}"
    if [[ -f "${FSLDIR}/etc/fslconf/fsl.sh" ]]; then
        set +e +u
        # shellcheck disable=SC1090
        . "${FSLDIR}/etc/fslconf/fsl.sh" >/dev/null 2>&1 || true
        set -e -u
        export PATH="${FSLDIR}/bin:${PATH}"
        log "DEBUG" "FSL: ${FSLDIR} ($(flirt -version 2>/dev/null | head -1 || echo 'version unknown'))"
    else
        log "WARN" "FSL not found at ${FSLDIR} — FSL-dependent stages will fail"
    fi

    # Append FSL Python packages at low priority so venv takes precedence
    export PYTHONPATH="${PYTHONPATH:-}:${FSLDIR}/lib/python3.12/site-packages"
}

# ---------------------------------------------------------------------------
# FreeSurfer
# ---------------------------------------------------------------------------

setup_freesurfer() {
    export FREESURFER_HOME="${FREESURFER_HOME:-/usr/local/freesurfer}"
    export SUBJECTS_DIR="${DWIFORGE_DIR_FREESURFER}"

    if [[ -f "${FREESURFER_HOME}/SetUpFreeSurfer.sh" ]]; then
        set +e +u
        export FS_FREESURFERENV_NO_OUTPUT=1
        # shellcheck disable=SC1090
        . "${FREESURFER_HOME}/SetUpFreeSurfer.sh" >/dev/null 2>&1 || true
        set -e -u
        log "DEBUG" "FreeSurfer: ${FREESURFER_HOME}"
    else
        log "WARN" "FreeSurfer not found at ${FREESURFER_HOME} — connectivity and SynthMorph stages will fail"
    fi
}

# ---------------------------------------------------------------------------
# MRtrix3
# ---------------------------------------------------------------------------

setup_mrtrix() {
    if [[ -n "${MRTRIX_HOME:-}" && -d "${MRTRIX_HOME}/bin" ]]; then
        export PATH="${MRTRIX_HOME}/bin:${PATH}"
    fi
    if command -v mrconvert >/dev/null 2>&1; then
        log "DEBUG" "MRtrix3: $(mrconvert --version 2>/dev/null | head -1 || echo 'version unknown')"
    else
        log "WARN" "MRtrix3 not found in PATH — MRtrix-dependent stages will fail"
    fi
}

# ---------------------------------------------------------------------------
# ANTs
# ---------------------------------------------------------------------------

setup_ants() {
    export ANTSPATH="${ANTSPATH:-/usr/local/bin}"
    export PATH="${ANTSPATH}:${PATH}"
    if command -v antsRegistration >/dev/null 2>&1; then
        log "DEBUG" "ANTs: $(antsRegistration --version 2>/dev/null | head -1 || echo 'version unknown')"
    else
        log "WARN" "ANTs not found — ANTs registration will not be available"
    fi
}


# ---------------------------------------------------------------------------
# DESIGNER-v2
# ---------------------------------------------------------------------------
# DESIGNER uses its own pinned dependencies (dipy==1.9.0) installed in the
# user site-packages. Our pipeline scripts (denoise.py etc.) run under a
# separate virtualenv with dipy>=1.12. The designer binary at
# ~/.local/bin/designer must be called with MRTRIX_PYTHON_PATH prepended to
# PYTHONPATH so it can import the mrtrix3 Python bindings.

setup_designer() {
    DESIGNER_AVAILABLE=false
    DESIGNER_BIN=""
    MRTRIX_PYTHON_PATH=""

    # ---- Locate MRtrix3 Python bindings ----
    # Auto-detect from mrconvert location: <mrtrix_root>/lib/mrtrix3/
    local mrtrix_bin
    mrtrix_bin=$(command -v mrconvert 2>/dev/null || true)
    if [[ -n "$mrtrix_bin" ]]; then
        local mrtrix_root
        mrtrix_root=$(dirname "$(dirname "$mrtrix_bin")")
        if [[ -d "${mrtrix_root}/lib/mrtrix3" ]]; then
            MRTRIX_PYTHON_PATH="${mrtrix_root}/lib"
        fi
    fi
    # Config override
    if [[ -n "${DWIFORGE_DESIGNER_PYTHON_PATH:-}" ]]; then
        MRTRIX_PYTHON_PATH="${DWIFORGE_DESIGNER_PYTHON_PATH}"
    fi

    # ---- Locate designer binary ----
    local candidates=(
        "${DWIFORGE_DESIGNER_BIN:-}"
        "${HOME}/.local/bin/designer"
    )
    # Check active virtualenv bin first (highest priority after explicit config)
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/designer" ]]; then
        candidates=("${VIRTUAL_ENV}/bin/designer" "${candidates[@]}")
    fi
    # Common virtualenv locations
    for venv_name in neuroimaging_env dwiforge_env mrtrix_env; do
        local venv_candidate="${HOME}/${venv_name}/bin/designer"
        [[ -x "$venv_candidate" ]] && candidates+=("$venv_candidate")
    done
    # Also check PATH, but skip Qt Designer (which also uses 'designer')
    local path_designer
    path_designer=$(command -v designer 2>/dev/null || true)
    if [[ -n "$path_designer" ]]; then
        candidates+=("$path_designer")
    fi

    for candidate in "${candidates[@]}"; do
        [[ -z "$candidate" || ! -x "$candidate" ]] && continue
        # Verify it's the neuroimaging DESIGNER, not Qt Designer.
        # Check file content rather than executing — avoids GPU/CUDA init
        # which can crash WSL or consume excessive memory on startup.
        if grep -q "designer2\|Benjamin Ades-Aron\|NYU-DiffusionMRI"                 "$candidate" 2>/dev/null; then
            DESIGNER_BIN="$candidate"
            log "DEBUG" "DESIGNER found: ${candidate}"
            break
        fi
    done

    if [[ -z "$DESIGNER_BIN" ]]; then
        log "WARN" "DESIGNER binary not found or not functional — steps 5-8 will be skipped"
        export DESIGNER_AVAILABLE DESIGNER_BIN MRTRIX_PYTHON_PATH
        return 0
    fi

    if [[ -z "$MRTRIX_PYTHON_PATH" ]]; then
        log "WARN" "MRtrix3 Python path not found — DESIGNER may fail to import mrtrix3"
    fi

    DESIGNER_AVAILABLE=true
    log "DEBUG" "DESIGNER: ${DESIGNER_BIN}"
    log "DEBUG" "DESIGNER MRtrix3 Python: ${MRTRIX_PYTHON_PATH}"
    export DESIGNER_AVAILABLE DESIGNER_BIN MRTRIX_PYTHON_PATH
}

# ---------------------------------------------------------------------------
# CUDA / GPU
# ---------------------------------------------------------------------------

setup_gpu() {
    # CUDA environment
    if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    fi
    export TF_CPP_MIN_LOG_LEVEL=3
    export TF_ENABLE_ONEDNN_OPTS=0
    export PYTHONWARNINGS="ignore"

    # Detect GPU via PyTorch
    if "${PYTHON_EXECUTABLE}" -c \
        "import torch; exit(0 if torch.cuda.is_available() else 1)" \
        >/dev/null 2>&1; then
        export GPU_AVAILABLE=true
        export PYTORCH_AVAILABLE=true
        local n_gpu
        n_gpu=$("${PYTHON_EXECUTABLE}" -c \
            "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
        log "INFO" "GPU: ${n_gpu} CUDA device(s) detected"
    else
        export GPU_AVAILABLE=false
        if "${PYTHON_EXECUTABLE}" -c "import torch" >/dev/null 2>&1; then
            export PYTORCH_AVAILABLE=true
            log "INFO" "GPU: not available — PyTorch CPU mode"
        else
            export PYTORCH_AVAILABLE=false
            log "INFO" "GPU: not available — PyTorch not installed (scipy fallback)"
        fi
    fi

    # Respect config override
    if [[ "${DWIFORGE_USE_GPU:-true}" == "false" ]]; then
        export GPU_AVAILABLE=false
        log "INFO" "GPU: disabled by configuration"
    fi
}

# ---------------------------------------------------------------------------
# SLURM module loading
# ---------------------------------------------------------------------------

setup_slurm_modules() {
    if [[ -z "${DWIFORGE_SLURM_MODULES:-}" ]]; then
        return 0
    fi
    if ! command -v module >/dev/null 2>&1; then
        log "WARN" "SLURM modules configured but 'module' command not found"
        return 0
    fi
    local mod
    for mod in ${DWIFORGE_SLURM_MODULES}; do
        set +e
        module load "$mod" 2>/dev/null
        local rc=$?
        set -e
        if [[ $rc -eq 0 ]]; then
            log "DEBUG" "Loaded module: ${mod}"
        else
            log "WARN" "Failed to load module: ${mod}"
        fi
    done
}

# ---------------------------------------------------------------------------
# Optional ML dependencies
# ---------------------------------------------------------------------------

check_ml_dependencies() {
    # PyTorch (already detected in setup_gpu)
    log "DEBUG" "PyTorch available: ${PYTORCH_AVAILABLE}"

    # AMICO (for NODDI)
    if "${PYTHON_EXECUTABLE}" -c "import amico" >/dev/null 2>&1; then
        export AMICO_AVAILABLE=true
        local ver
        ver=$("${PYTHON_EXECUTABLE}" -c \
            "import amico; print(getattr(amico,'__version__','unknown'))" \
            2>/dev/null || echo "unknown")
        log "DEBUG" "AMICO: ${ver}"
    else
        export AMICO_AVAILABLE=false
        log "WARN" "AMICO not found — NODDI stage will be unavailable"
        log "WARN" "Install with: pip install dmri-amico"
    fi

    # SynthMorph (via FreeSurfer mri_synthmorph)
    if command -v mri_synthmorph >/dev/null 2>&1; then
        export SYNTHMORPH_AVAILABLE=true
        log "DEBUG" "SynthMorph: available"
    else
        export SYNTHMORPH_AVAILABLE=false
        log "DEBUG" "SynthMorph: not available (needs FreeSurfer 7.3+)"
    fi

    # scipy (for registration fallback)
    if "${PYTHON_EXECUTABLE}" -c "import scipy" >/dev/null 2>&1; then
        log "DEBUG" "scipy: available"
    else
        log "WARN" "scipy not found — registration quick mode unavailable"
        log "WARN" "Install with: pip install scipy"
    fi
}

# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------

setup_threads() {
    local n="${DWIFORGE_OMP_THREADS:-0}"
    if [[ "$n" -le 0 ]]; then
        n=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    fi
    export OMP_NUM_THREADS="$n"
    export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$n"
    export MRTRIX_NTHREADS="$n"
    log "DEBUG" "Threads: ${n}"
}

# ---------------------------------------------------------------------------
# Main entry point — call this once at startup
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Container fast-path
# ---------------------------------------------------------------------------
# When DWIFORGE_CONTAINERIZED=true (set by the Docker/Apptainer entrypoint),
# all tools are at known paths baked into the image. Skip detection entirely.
# Paths are read from variables exported by the entrypoint (sourced from
# /etc/dwiforge_env which is written during docker build).

_setup_environment_container() {
    log "INFO" "Container mode — using pre-configured paths"

    # Python — system python3 in the container
    export PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"

    # Our deps directory (dipy>=1.12, etc.)
    export PYTHONPATH="${DWIFORGE_DEPS_DIR}:${PYTHONPATH:-}"

    # DESIGNER — DESIGNER_BIN and MRTRIX_PYTHON_PATH come from entrypoint
    if [[ -n "${DESIGNER_BIN:-}" && -x "${DESIGNER_BIN}" ]]; then
        export DESIGNER_AVAILABLE=true
        log "DEBUG" "DESIGNER: ${DESIGNER_BIN}"
        log "DEBUG" "MRtrix3 Python: ${MRTRIX_PYTHON_PATH:-<not set>}"
    else
        export DESIGNER_AVAILABLE=false
        log "WARN" "DESIGNER_BIN not set or not executable in container"
    fi

    # MRtrix3Tissue — Python scripts, no build required
    export MRTRIX3TISSUE_HOME="${MRTRIX3TISSUE_HOME:-/opt/mrtrix3tissue}"
    if [[ -f "${MRTRIX3TISSUE_HOME}/bin/ss3t_csd_beta1" ]]; then
        export SS3T_CSD="${MRTRIX3TISSUE_HOME}/bin/ss3t_csd_beta1"
        export RESPONSEMEAN="${MRTRIX3TISSUE_HOME}/bin/responsemean"
        log "DEBUG" "MRtrix3Tissue: ${MRTRIX3TISSUE_HOME}"
    else
        log "WARN" "ss3t_csd_beta1 not found — tractography stage will fail"
        log "WARN" "Expected: ${MRTRIX3TISSUE_HOME}/bin/ss3t_csd_beta1"
    fi

    # Synb0-DisCo — EPI susceptibility correction
    export SYNB0_HOME="${SYNB0_HOME:-/opt/synb0}"
    export DWIFORGE_SYNB0_HOME="${SYNB0_HOME}"
    if [[ -f "${SYNB0_HOME}/src/pipeline.py" ]]; then
        log "DEBUG" "Synb0: ${SYNB0_HOME}"
    else
        log "WARN" "Synb0 pipeline.py not found — EPI correction path B unavailable"
        log "WARN" "Expected: ${SYNB0_HOME}/src/pipeline.py"
    fi

    # FreeSurfer — check license early
    if [[ ! -f "${FS_LICENSE:-/opt/freesurfer/license.txt}" ]]; then
        log "WARN" "FreeSurfer license not found — recon-all will fail"
        log "WARN" "Mount it with: --bind /path/to/license.txt:/opt/freesurfer/license.txt:ro"
    fi

    setup_threads
    setup_gpu
    check_ml_dependencies

    log "OK" "Container environment ready"
}

_setup_environment_host() {
    log "INFO" "Host mode — detecting tool locations..."
    setup_python
    setup_fsl
    setup_freesurfer
    setup_mrtrix
    setup_ants
    setup_designer
    setup_threads
    [[ -n "${SLURM_JOB_ID:-}" ]] && setup_slurm_modules
    setup_gpu
    check_ml_dependencies
    log "OK" "Environment setup complete"
}

setup_environment() {
    if [[ "${DWIFORGE_CONTAINERIZED:-false}" == "true" ]]; then
        _setup_environment_container
    else
        _setup_environment_host
    fi
}
