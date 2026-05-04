#!/usr/bin/env bash
# container/build.sh
# =============================================================================
# Helper script for building the dwiforge Docker image and converting to
# an Apptainer .sif for HPC upload.
#
# Usage:
#   bash container/build.sh [options]
#
# Options:
#   --tag TAG             Docker image tag (default: dwiforge:2.0)
#   --sif PATH            Output .sif path (default: dwiforge_2.0.sif)
#   --skip-docker         Skip Docker build (use existing image)
#   --skip-sif            Skip Apptainer conversion
#   --push-sif HOST:PATH  SCP the .sif to an HPC after building
#   --freesurfer-ver VER  FreeSurfer version (default: 7.4.1)
#   --designer-tag TAG    DESIGNER image tag (default: latest)
#   --no-cache            Build Docker image without cache
#
# Examples:
#   # Build everything and upload to HPC
#   bash container/build.sh --push-sif hpc.university.edu:/home/user/images/
#
#   # Rebuild after code change only (reuse cached base layers)
#   bash container/build.sh --tag dwiforge:2.1 --sif dwiforge_2.1.sif
# =============================================================================

set -euo pipefail

# ---- Defaults ----
TAG="dwiforge:2.0"
SIF="dwiforge_2.0.sif"
SKIP_DOCKER=false
SKIP_SIF=false
PUSH_HOST=""
FS_VERSION="7.4.1"
DESIGNER_TAG="latest"
NO_CACHE=""

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)            TAG="$2";          shift 2 ;;
        --sif)            SIF="$2";          shift 2 ;;
        --skip-docker)    SKIP_DOCKER=true;  shift   ;;
        --skip-sif)       SKIP_SIF=true;     shift   ;;
        --push-sif)       PUSH_HOST="$2";    shift 2 ;;
        --freesurfer-ver) FS_VERSION="$2";   shift 2 ;;
        --designer-tag)   DESIGNER_TAG="$2"; shift 2 ;;
        --no-cache)       NO_CACHE="--no-cache"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Must run from repo root ----
if [[ ! -f "dwiforge.sh" ]]; then
    echo "ERROR: Run this script from the dwiforge repository root."
    echo "       cd /path/to/dwiforge && bash container/build.sh"
    exit 1
fi

echo "==================================================="
echo "  dwiforge container build"
echo "  Docker tag:       ${TAG}"
echo "  Apptainer .sif:   ${SIF}"
echo "  FreeSurfer:       ${FS_VERSION}"
echo "  DESIGNER:         ${DESIGNER_TAG}"
echo "==================================================="
echo ""

# ---- Step 1: Docker build ----
if [[ "$SKIP_DOCKER" == false ]]; then
    echo "[ 1/3 ] Building Docker image: ${TAG}"
    docker build ${NO_CACHE} \
        --build-arg FREESURFER_VERSION="${FS_VERSION}" \
        --build-arg DESIGNER_TAG="${DESIGNER_TAG}" \
        -t "${TAG}" \
        -f container/Dockerfile \
        .
    echo "        Done."
else
    echo "[ 1/3 ] Skipping Docker build (--skip-docker)"
fi

# ---- Step 2: Convert to Apptainer ----
if [[ "$SKIP_SIF" == false ]]; then
    echo ""
    echo "[ 2/3 ] Converting Docker image to Apptainer .sif"
    echo "        This may take several minutes (~20 GB image)..."

    if command -v apptainer >/dev/null 2>&1; then
        APPTAINER_CMD="apptainer"
    elif command -v singularity >/dev/null 2>&1; then
        APPTAINER_CMD="singularity"
    else
        echo "        WARNING: Neither apptainer nor singularity found."
        echo "        Saving Docker image as tar.gz instead."
        TARBALL="${SIF%.sif}.tar.gz"
        docker save "${TAG}" | gzip > "${TARBALL}"
        echo "        Saved: ${TARBALL}"
        echo "        On HPC, convert with:"
        echo "          apptainer build ${SIF} docker-archive://${TARBALL}"
        SKIP_SIF=true
    fi

    if [[ "$SKIP_SIF" == false ]]; then
        ${APPTAINER_CMD} build "${SIF}" "docker-daemon://${TAG}"
        echo "        Done: ${SIF} ($(du -sh "${SIF}" | cut -f1))"
    fi
else
    echo "[ 2/3 ] Skipping .sif conversion (--skip-sif)"
fi

# ---- Step 3: Upload to HPC ----
if [[ -n "$PUSH_HOST" ]]; then
    echo ""
    echo "[ 3/3 ] Uploading ${SIF} to ${PUSH_HOST}"
    scp "${SIF}" "${PUSH_HOST}"
    echo "        Done."
else
    echo "[ 3/3 ] Skipping upload (no --push-sif specified)"
fi

echo ""
echo "==================================================="
echo "  Build complete"
echo ""
echo "  To run locally with Docker:"
echo "    docker run --rm \\"
echo "      -v /data/bids:/data/bids:ro \\"
echo "      -v /data/output:/data/output \\"
echo "      -v ~/license.txt:/opt/freesurfer/license.txt:ro \\"
echo "      ${TAG} --all"
echo ""
if [[ "$SKIP_SIF" == false && -f "$SIF" ]]; then
echo "  To run on HPC with Apptainer:"
echo "    apptainer run \\"
echo "      --bind /data/bids:/data/bids:ro \\"
echo "      --bind /data/output:/data/output \\"
echo "      --bind ~/license.txt:/opt/freesurfer/license.txt:ro \\"
echo "      --nv \\"
echo "      ${SIF} --all"
fi
echo "==================================================="
