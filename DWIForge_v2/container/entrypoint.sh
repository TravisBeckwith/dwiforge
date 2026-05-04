#!/usr/bin/env bash
# container/entrypoint.sh
# =============================================================================
# Container entrypoint. Sources the build-time-discovered environment paths
# from /etc/dwiforge_env (written during docker build), then delegates to
# the main dwiforge.sh orchestrator.
#
# This two-step approach means env_setup.sh doesn't need to re-detect paths
# inside the container — everything is already known.
# =============================================================================

set -euo pipefail

# Source build-time-discovered paths (DESIGNER bin, MRtrix3 Python, FSLDIR)
if [[ -f /etc/dwiforge_env ]]; then
    while IFS='=' read -r key val; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        # Strip _BAKED suffix and export as the real variable name
        real_key="${key%_BAKED}"
        export "${real_key}=${val}"
    done < /etc/dwiforge_env
fi

# Signal to env_setup.sh that we are inside a container
export DWIFORGE_CONTAINERIZED=true
export DWIFORGE_DEPS_DIR="${DWIFORGE_DEPS_DIR:-/opt/dwiforge/deps}"

# FreeSurfer licence check — warn early rather than failing mid-pipeline
if [[ ! -f "${FS_LICENSE:-/opt/freesurfer/license.txt}" ]]; then
    echo "WARNING: FreeSurfer license not found at ${FS_LICENSE:-/opt/freesurfer/license.txt}"
    echo "         Mount it with: -v /path/to/license.txt:/opt/freesurfer/license.txt:ro"
    echo "         FreeSurfer-dependent stages (recon-all, SynthMorph) will fail."
fi

# MRtrix3Tissue — explicit path export for stage 09 (ss3t_csd_beta1, responsemean)
export MRTRIX3TISSUE_HOME="${MRTRIX3TISSUE_HOME:-/opt/mrtrix3tissue}"

# Synb0-DisCo — explicit path export for stage 04 (EPI correction)
export SYNB0_HOME="${SYNB0_HOME:-/opt/synb0}"
export DWIFORGE_SYNB0_HOME="${SYNB0_HOME}"

exec /opt/dwiforge/dwiforge.sh "$@"
