# dwiforge v2 — Dockerfile
# =============================================================================
# Multi-stage build:
#   Stage 1 (freesurfer)  — extracts FreeSurfer from official image
#   Stage 2 (final)       — DESIGNER base + FreeSurfer + dwiforge scripts
#
# BUILD
#   docker build \
#     --build-arg FREESURFER_VERSION=7.4.1 \
#     --build-arg DESIGNER_TAG=latest \
#     -t dwiforge:2.0 \
#     -f container/Dockerfile .
#
# RUN (local)
#   docker run --rm \
#     -v /data/bids:/data/bids:ro \
#     -v /data/output:/data/output \
#     -v /path/to/license.txt:/opt/freesurfer/license.txt:ro \
#     [--gpus all] \
#     dwiforge:2.0 --config /data/bids/dwiforge.toml --subject sub-001
#
# CONVERT TO APPTAINER
#   docker save dwiforge:2.0 | gzip > dwiforge_2.0.tar.gz
#   apptainer build dwiforge_2.0.sif docker-archive://dwiforge_2.0.tar.gz
#
# FREESURFER LICENSE
#   The license.txt is NOT baked into the image (redistribution prohibited).
#   Mount it at runtime: -v /path/to/license.txt:/opt/freesurfer/license.txt:ro
#   Or set FS_LICENSE env var to a path visible inside the container.
# =============================================================================

ARG FREESURFER_VERSION=7.4.1
ARG DESIGNER_TAG=latest

# ---------------------------------------------------------------------------
# Stage 1: FreeSurfer
# ---------------------------------------------------------------------------
FROM freesurfer/freesurfer:${FREESURFER_VERSION} AS freesurfer

# FreeSurfer installs to /usr/local/freesurfer in the official image.
# We copy the entire tree — ~15 GB. The final image is large by necessity.

# ---------------------------------------------------------------------------
# Stage 2: Final image
# ---------------------------------------------------------------------------
FROM nyudiffusionmri/designer:${DESIGNER_TAG}

LABEL maintainer="dwiforge"
LABEL description="dwiforge v2 DWI preprocessing pipeline"

# ---- Copy FreeSurfer from stage 1 ----
COPY --from=freesurfer /usr/local/freesurfer /opt/freesurfer

# ---- FreeSurfer environment ----
ENV FREESURFER_HOME=/opt/freesurfer
ENV PATH=/opt/freesurfer/bin:$PATH
ENV PERL5LIB=/opt/freesurfer/mni/lib/perl5/5.8.5
ENV MNI_PERL5LIB=/opt/freesurfer/mni/lib/perl5/5.8.5
# License path — must be bind-mounted at runtime.
# FS_LICENSE is the env var FreeSurfer checks.
ENV FS_LICENSE=/opt/freesurfer/license.txt

# ---- Discover MRtrix3 Python bindings path at build time ----
# The DESIGNER image ships MRtrix3 but we need to know where its Python
# package lives so env_setup.sh can pass it to DESIGNER correctly.
# We bake the discovered path into an ENV var.
RUN MRTRIX_BIN=$(which mrconvert 2>/dev/null || true) && \
    if [ -n "$MRTRIX_BIN" ]; then \
        MRTRIX_ROOT=$(dirname "$(dirname "$MRTRIX_BIN")"); \
        if [ -d "${MRTRIX_ROOT}/lib/mrtrix3" ]; then \
            echo "MRTRIX_PYTHON_PATH_BAKED=${MRTRIX_ROOT}/lib" >> /etc/dwiforge_env; \
        fi; \
    fi && \
    DESIGNER_BIN=$(which designer 2>/dev/null || echo "") && \
    echo "DESIGNER_BIN_BAKED=${DESIGNER_BIN}" >> /etc/dwiforge_env && \
    echo "FSLDIR_BAKED=${FSLDIR:-/usr/local/fsl}" >> /etc/dwiforge_env && \
    cat /etc/dwiforge_env

# ---- dwiforge Python deps (isolated from DESIGNER's dipy==1.9.0) ----
# DESIGNER pins dipy==1.9.0 in its site-packages.
# We install our deps into /opt/dwiforge/deps, which env_setup.sh prepends
# to PYTHONPATH. DESIGNER is called with a clean PYTHONPATH containing only
# the MRtrix3 path — it never sees /opt/dwiforge/deps.
RUN pip install --target /opt/dwiforge/deps --no-deps \
    "dipy>=1.12.0" \
    "nibabel>=5.0" \
    "numpy>=1.24" \
    "scipy>=1.10" \
    "scikit-learn>=1.3" \
    "tqdm>=4.65" \
    "matplotlib>=3.7"

# ---- MRtrix3Tissue (SS3T-CSD + responsemean for tractography) ----
# ss3t_csd_beta1 and responsemean are Python scripts — no compilation needed.
# The mrtrix3.py shim in MRtrix3Tissue uses the deprecated `imp` module
# (removed in Python 3.12). We patch it with the importlib-based version
# from standard MRtrix3, which is already present in the DESIGNER image.
# MRtrix3Tissue scripts are NOT added to PATH — called by explicit full path
# in stage 09 to avoid shadowing standard MRtrix3 commands.
ARG MRTRIX3TISSUE_VERSION=3Tissue_v5.2.9
RUN git clone --depth 1         --branch ${MRTRIX3TISSUE_VERSION}         https://github.com/3Tissue/MRtrix3Tissue.git         /opt/mrtrix3tissue &&     # Discover standard MRtrix3 bin dir (from DESIGNER base image)
    MRTRIX_BIN=$(which mrconvert 2>/dev/null || true) &&     if [ -n "$MRTRIX_BIN" ]; then         MRTRIX_BINDIR=$(dirname "$MRTRIX_BIN");         # Patch mrtrix3.py shim for Python 3.12+ compatibility
        if [ -f "${MRTRIX_BINDIR}/mrtrix3.py" ]; then             cp "${MRTRIX_BINDIR}/mrtrix3.py" /opt/mrtrix3tissue/bin/mrtrix3.py;             echo "MRtrix3Tissue mrtrix3.py patched for Python 3.12";         fi;     fi &&     echo "MRTRIX3TISSUE_HOME_BAKED=/opt/mrtrix3tissue" >> /etc/dwiforge_env &&     # Verify ss3t_csd_beta1 is present
    test -f /opt/mrtrix3tissue/bin/ss3t_csd_beta1 &&         echo "ss3t_csd_beta1: OK" ||         echo "WARNING: ss3t_csd_beta1 not found in MRtrix3Tissue bin"

ENV MRTRIX3TISSUE_HOME=/opt/mrtrix3tissue

# ---- Synb0-DisCo (synthetic b0 for EPI susceptibility correction) ----
# Synb0 needs: FSL, ANTs, FreeSurfer (all already present), and PyTorch.
# We clone the source and install its Python dependencies.
# PyTorch CPU-only is sufficient — Synb0 inference is fast even on CPU.
# GPU version can be used if --gpus all is passed at runtime.
ARG SYNB0_VERSION=v3.1
RUN git clone --depth 1         --branch ${SYNB0_VERSION}         https://github.com/MASILab/Synb0-DISCO.git         /opt/synb0 2>/dev/null ||     # Fallback: clone main if tag doesn't exist
    git clone --depth 1         https://github.com/MASILab/Synb0-DISCO.git         /opt/synb0 &&     # Install Synb0 Python requirements
    # torch CPU-only to keep image size manageable (~2GB vs ~6GB for CUDA)
    pip install --no-cache-dir         torch torchvision --index-url https://download.pytorch.org/whl/cpu &&     pip install --no-cache-dir         antspyx         nibabel         numpy &&     echo "SYNB0_HOME_BAKED=/opt/synb0" >> /etc/dwiforge_env &&     # Verify pipeline script is present
    test -f /opt/synb0/src/pipeline.py &&         echo "Synb0 pipeline.py: OK" ||         echo "WARNING: Synb0 pipeline.py not found — check repo structure"

ENV SYNB0_HOME=/opt/synb0

# ---- dwiforge pipeline ----
COPY . /opt/dwiforge/

RUN chmod +x \
    /opt/dwiforge/dwiforge.sh \
    /opt/dwiforge/stages/*.sh \
    /opt/dwiforge/lib/*.sh \
    /opt/dwiforge/env/bootstrap_deps.sh \
    /opt/dwiforge/container/build.sh 2>/dev/null || true

# ---- Container environment ----
ENV DWIFORGE_CONTAINERIZED=true
ENV DWIFORGE_DEPS_DIR=/opt/dwiforge/deps
# Actual paths populated from /etc/dwiforge_env at runtime via entrypoint.
# Defaults here are sensible fallbacks if the discovery step above succeeded.
ENV DWIFORGE_ROOT=/opt/dwiforge

# ---- Scratch space ----
RUN mkdir -p /scratch && chmod 777 /scratch
ENV DWIFORGE_SCRATCH=/scratch

# ---- Entrypoint ----
COPY container/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
