# dwiforge — Dependency Tracking
<!-- Update this file as new stages are implemented. -->
<!-- Each entry notes which stage/script requires it and why the version bound exists. -->

## Status key
- CONFIRMED — explicitly used in implemented code, version bound tested
- PENDING   — will be required by a stage not yet implemented
- OPTIONAL  — improves functionality but pipeline runs without it

---

## Python runtime

| Package | Version | Status | Required by | Notes |
|---|---|---|---|---|
| Python | >= 3.10 | CONFIRMED | all scripts | 3.11+ preferred — `tomllib` is stdlib from 3.11; older needs `tomli` |
| Python | >= 3.11 | OPTIONAL | scripts/parse_config.py | Avoids `tomli` install; not hard requirement |

---

## Python packages — core (always required)

| Package | Version | Status | Required by | Notes |
|---|---|---|---|---|
| numpy | >= 1.24 | CONFIRMED | denoise.py, qc_bids.py, dwi_registration.py | 1.24 minimum for DIPY 1.12 compat |
| nibabel | >= 5.0 | CONFIRMED | denoise.py, qc_bids.py, dwi_registration.py | NIfTI I/O throughout |
| scipy | >= 1.10 | CONFIRMED | dwi_registration.py | `ndimage.shift`, `optimize.minimize` for quick-mode registration |
| scikit-learn | >= 1.3 | CONFIRMED | denoise.py (via DIPY patch2self) | patch2self OLS/ridge/lasso regressors |
| tqdm | >= 4.65 | CONFIRMED | denoise.py (via DIPY patch2self) | progress display during P2S denoising |
| tomli | >= 2.0 | CONFIRMED | scripts/parse_config.py | TOML parsing on Python < 3.11; not needed on 3.11+ |

---

## Python packages — ML / denoising

| Package | Version | Status | Required by | Notes |
|---|---|---|---|---|
| dipy | >= 1.12.0 | CONFIRMED | denoise.py | **1.12 minimum** — P2S v3 bug (PR #3631) fixed in 1.12; v3 on 1.10/1.11 returns wrong volumes |
| torch (PyTorch) | >= 2.0 | OPTIONAL | dwi_registration.py | Full-mode dense displacement registration; falls back to scipy if absent |
| torchvision | — | — | — | Not currently used; may be needed for future DL stages |

---

## Python packages — NODDI / microstructure

| Package | pip name | Version | Status | Required by | Notes |
|---|---|---|---|---|---|
| amico | dmri-amico | >= 2.0 | PENDING | stages/05_noddi.sh | NODDI fitting; AMICO 2.x requires Python >= 3.8 |

---

## Python packages — pending (not yet implemented)

| Package | pip name | Status | Required by | Notes |
|---|---|---|---|---|
| antspyx | antspyx | PENDING | stages/07_refinement.sh | Python bindings for ANTs registration |
| fury | fury | OPTIONAL | DIPY visualisation | Only needed for visual QC outputs, not core pipeline |

---

## System tools — neuroimaging

| Tool | Version | Status | Required by | Notes |
|---|---|---|---|---|
| FSL | >= 6.0.7 | PENDING | stages/01_preprocessing.sh (steps 6, 8) | eddy, topup, BET, dtifit, FLIRT, FNIRT |
| MRtrix3 | >= 3.0.4 | CONFIRMED | stages/01_preprocessing.sh | mrconvert, mrdegibbs; dwi2mask, tckgen, tck2connectome pending |
| FreeSurfer | >= 7.3.0 | PENDING | stages/06_connectivity.sh | recon-all; 7.3+ required for SynthMorph |
| ANTs | >= 2.5.0 | PENDING | stages/03_refinement.sh | antsRegistration, N4BiasFieldCorrection |

### FSL tools used (current and pending)
| FSL tool | Stage | Status |
|---|---|---|
| `eddy` | 01_preprocessing step 6 | PENDING |
| `topup` | 01_preprocessing step 5 | PENDING |
| `applytopup` | 01_preprocessing step 5 | PENDING |
| `bet` | 02_t1w_prep step 3 (fallback skull strip) | CONFIRMED |
| `flirt` | 05_tensor_fitting (b0→T1w registration) | CONFIRMED |
| `fast` | 05_tensor_fitting (WM tissue segmentation) | CONFIRMED |
| `dtifit` | not used — tmi replaces dtifit | DROPPED |
| `fslmaths` | 05_tensor_fitting (WM mask thresholding) | CONFIRMED |
| `convert_xfm` | 05_tensor_fitting (invert affine) | CONFIRMED |

### MRtrix3 tools used (current and pending)
| MRtrix3 tool | Stage | Status |
|---|---|---|
| `mrconvert` | 01_preprocessing steps 2, 3, 9 | CONFIRMED |
| `mrdegibbs` | 01_preprocessing step 4 | CONFIRMED (implemented) |
| `dwidenoise` | denoise.py (MP-PCA fallback) | CONFIRMED |
| `dwi2mask` | 01_preprocessing step 8 | PENDING |
| `dwibiascorrect` | 01_preprocessing step 7 | PENDING |
| `dwi2tensor` | 04_tensor_fitting | PENDING |
| `tensor2metric` | 04_tensor_fitting | PENDING |
| `tckgen` | 06_connectivity | PENDING |
| `tck2connectome` | 06_connectivity | PENDING |
| `tcksift2` | 06_connectivity | PENDING |
| `5ttgen` | 06_connectivity | PENDING |

---

## System tools — optional / conditional

| Tool | Version | Status | Required by | Notes |
|---|---|---|---|---|
| CUDA toolkit | >= 12.3 | OPTIONAL | GPU acceleration | RTX 3070 tested; any CUDA 12.x should work |
| cuDNN | matching CUDA | OPTIONAL | PyTorch GPU | Required if using torch with CUDA |
| Docker or Apptainer | any | OPTIONAL | Synb0-DisCo container | Only needed if no reverse PE b0 and T1w available |
| Node.js | >= 18 | OPTIONAL | bids-validator | Not used in pipeline; for pre-run BIDS validation only |

---

## Environment variables set by the pipeline

These are set in `lib/env_setup.sh` and expected by stage scripts:

```bash
FSLDIR=/usr/local/fsl
FREESURFER_HOME=/usr/local/freesurfer
SUBJECTS_DIR=${DWIFORGE_DIR_FREESURFER}
ANTSPATH=/usr/local/bin
CUDA_HOME=/usr/local/cuda-12.3
OMP_NUM_THREADS=<auto>
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=<auto>
MRTRIX_NTHREADS=<auto>
PYTHON_EXECUTABLE=<resolved>
GPU_AVAILABLE=true|false
PYTORCH_AVAILABLE=true|false
AMICO_AVAILABLE=true|false
SYNTHMORPH_AVAILABLE=true|false
```

---

## Version history

| Date | Change |
|---|---|
| 2026-04-04 | Initial tracking document — stages 00, 01 implemented |
| 2026-04-04 | Gibbs step implemented — mrdegibbs only; dldegibbs dropped (GLOBUS weights unavailable) |
| 2026-04-04 | qc_bids.py extended: PartialFourier, MRAcquisitionType, recommended_gibbs_method |
| 2026-04-04 | DESIGNER-v2 2.0.15 installed; env conflict documented; steps 5-8 implemented |
| 2026-04-04 | Stage 05 implemented — tmi DTI fitting; FAST WM mask; b0→T1w registration |
| 2026-04-04 | Stages restructured: 01 trimmed; 02 t1w_prep; 03 epi_correction; 04 designer |
| 2026-04-04 | antspyx 0.4.2 confirmed (pulled by DESIGNER); pyyaml 6.0.1 confirmed (pulled by DESIGNER) |
| 2026-04-04 | Philips PF detection fixed in qc_bids.py; multi-run detection added |
| 2026-04-04 | mrconvert -json_import added to step 2; sigma map generation added to step 3b |

---
<!-- NEXT UPDATE: add versions for FSL eddy/topup when step 5-6 implemented -->
<!-- NEXT UPDATE: add dmri-amico when noddi stage implemented -->
<!-- NEXT UPDATE: confirm antspyx version when refinement stage implemented -->

## Considered and rejected

| Tool | Reason dropped |
|---|---|
| dldegibbs (Muckley 2021) | Pretrained weights no longer available on GLOBUS; no alternative weight source |

---

## DESIGNER-v2 environment conflict

DESIGNER-v2 (2.0.15) pins `dipy==1.9.0`. Our pipeline scripts require `dipy>=1.12.0`
(P2S v3 bug fix, PR #3631). These cannot share a Python environment.

**Resolution:** Two separate environments.

| Environment | Purpose | Key package |
|---|---|---|
| `dwiforge` virtualenv | Our scripts (denoise.py, qc_bids.py, etc.) | dipy>=1.12.0 |
| user site-packages | DESIGNER binary | dipy==1.9.0 (pinned by designer2) |

DESIGNER is called as a subprocess via `~/.local/bin/designer` — it uses its own
installed packages and never imports from the dwiforge virtualenv. Our scripts
never import designer2 directly.

**MRtrix3 Python path:** DESIGNER requires the MRtrix3 Python bindings
(`mrtrix3` package) which are not installed via pip. They live in the from-source
MRtrix3 build at `<mrtrix_root>/lib/mrtrix3/`. The path `<mrtrix_root>/lib` must
be prepended to `PYTHONPATH` for every DESIGNER call. `lib/env_setup.sh`
`setup_designer()` handles this automatically by locating `mrconvert` and
deriving the root.

**Installation:**
```bash
# DESIGNER (installs into user site-packages with its own pins)
pip install git+https://github.com/NYU-DiffusionMRI/DESIGNER-v2.git

# dwiforge scripts (separate virtualenv with dipy>=1.12)
python3 -m venv ~/.venvs/dwiforge
~/.venvs/dwiforge/bin/pip install -r env/requirements.txt
```

---

## Stage 08: Response Functions

| Tool | Version | Source | Notes |
|---|---|---|---|
| `dwi2response dhollander` | MRtrix3 3.0.8 | standard MRtrix3 | per-subject, then group-averaged |
| `responsemean` | MRtrix3Tissue 5.2.9 | `/opt/mrtrix3tissue/bin/` | group averaging across subjects |

---

## Stage 09: Tractography & Connectome

### MRtrix3Tissue
**Purpose:** `ss3t_csd_beta1` for Single-Shell 3-Tissue CSD (not available in standard MRtrix3).

**Key detail:** MRtrix3Tissue scripts are Python, not compiled — no build required.
The `mrtrix3.py` shim must be patched from standard MRtrix3 for Python 3.12 compatibility
(removes deprecated `imp` module).

| Command | Source | Notes |
|---|---|---|
| `ss3t_csd_beta1` | `/opt/mrtrix3tissue/bin/` | Python script, explicit path only |
| `responsemean` | `/opt/mrtrix3tissue/bin/` | Python script, explicit path only |

**NOT added to PATH** — called by explicit full path in stage 09 to avoid shadowing
standard MRtrix3 commands.

**Installation (container):**
```dockerfile
RUN git clone --depth 1 --branch 3Tissue_v5.2.9 \
        https://github.com/3Tissue/MRtrix3Tissue.git /opt/mrtrix3tissue && \
    cp /opt/mrtrix3/bin/mrtrix3.py /opt/mrtrix3tissue/bin/mrtrix3.py
ENV MRTRIX3TISSUE_HOME=/opt/mrtrix3tissue
```

**Installation (local — Python scripts only, no build):**
```bash
git clone --branch 3Tissue_v5.2.9 https://github.com/3Tissue/MRtrix3Tissue.git ~/MRtrix3Tissue
cp /home/user/mrtrix3/bin/mrtrix3.py ~/MRtrix3Tissue/bin/mrtrix3.py
export MRTRIX3TISSUE_HOME=~/MRtrix3Tissue
# Test:
PYTHONPATH=/home/user/mrtrix3/lib ~/MRtrix3Tissue/bin/ss3t_csd_beta1 -help
```

### Standard MRtrix3 commands (stage 09)

| Command | Purpose |
|---|---|
| `mtnormalise` | Multi-tissue intensity normalisation |
| `5ttgen hsvs` | 5TT image from FreeSurfer surfaces |
| `5tt2gmwmi` | GM/WM interface seeding image |
| `tckgen` | iFOD2 probabilistic tractography (ACT) |
| `tcksift2` | SIFT2 streamline weights |
| `labelconvert` | FreeSurfer parcellation → connectome labels |
| `tck2connectome` | Connectivity matrix (count + mean FA) |
| `tcksample` | Sample FA along streamlines |

### Synb0-DisCo
**Purpose:** Synthesises an undistorted b0 from T1w for EPI susceptibility correction
when no reverse-PE acquisition exists (stage 04 Path B).

**Dependencies:** FSL, ANTs, FreeSurfer (all present in container), PyTorch (CPU).

| Package | Version | Notes |
|---|---|---|
| PyTorch | CPU-only | `--index-url https://download.pytorch.org/whl/cpu` |
| antspyx | latest | already pulled by DESIGNER; confirmed present |

**Installation (container):**
```dockerfile
RUN git clone --depth 1 --branch v3.1 \
        https://github.com/MASILab/Synb0-DISCO.git /opt/synb0 && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
ENV SYNB0_HOME=/opt/synb0
```

---

## Stage 10: QC Report

| Package | Version | Location | Notes |
|---|---|---|---|
| `matplotlib` | >=3.7 | `/opt/dwiforge/deps` | PDF generation |
| `nibabel` | >=5.0 | `/opt/dwiforge/deps` | Image slice extraction |
| `numpy` | >=1.24 | `/opt/dwiforge/deps` | Metric computation |

---

## Version History (continued)

| Date | Change |
|---|---|
| 2026-04-26 | MRtrix3Tissue 5.2.9 added (ss3t_csd_beta1, responsemean) |
| 2026-04-26 | Synb0-DisCo v3.1 added to container (PyTorch CPU) |
| 2026-04-26 | matplotlib added to dwiforge deps for QC report |
| 2026-04-26 | Stage 09 (tractography) implemented — SS3T-CSD, ACT, SIFT2, connectome |
| 2026-04-26 | Stage 10 (qc_report) implemented — PDF with metrics + slices |
| 2026-04-26 | Pipeline complete: stages 00-10 all implemented |
