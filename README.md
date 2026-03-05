# DTI_ML_project

Machine learning-enhanced diffusion tensor imaging (DTI) processing pipeline for multimodal neuroimaging research.

---

## Overview

This pipeline integrates traditional DTI processing with ML-based registration techniques to improve accuracy across structural, diffusion, and connectivity analyses. It was developed for use with BIDS-formatted datasets and is optimized for multi-drive storage environments.

**Current version:** `v1.4-ml-enhanced` (beta)

### What's new in v8 beta

- **13 fixes and implementations** — GPU detection accuracy, ANTs registration initialization, directory stack leak prevention on error paths, SSIM numerical stability for MRI data, file format integrity for NIfTI compression, and several correctness and portability improvements (see [Changelog](#changelog) for the full list)
- **SynthMorph warp application completed** — the three SynthMorph stubs (`extract_synthmorph_transform`, 5TT warp, parcellation warp) are now fully implemented with proper RAS→LPS displacement field conversion and per-volume `antsApplyTransforms` application
- **Removed hardcoded developer path** — `/home/trav/mrtrix3/bin` replaced with auto-detected `MRTRIX_HOME`
- **Improved error resilience** — all `safe_cd` error paths now properly unwind the directory stack
- **Container variable consistency** — early-execution code now uses `CONTAINER_CMD` uniformly; `DOCKER_CMD` alias deferred to `setup_environment`

---

## Features

- **Full DTI pipeline** — preprocessing, eddy correction, bias correction, tractography, and connectivity analysis
- **ML-enhanced registration** — VoxelMorph, SynthMorph, and ANTs-based registration with automatic fallback; SynthMorph displacement fields are fully converted to ITK format for downstream warp application
- **NODDI estimation** — neurite orientation dispersion and density imaging parameter fitting
- **Synb0 distortion correction** — via Synb0-DisCo with traditional fallback
- **Storage-optimized** — configurable multi-drive data management with BIDS compliance throughout
- **GPU acceleration** — CUDA support via TensorFlow; graceful CPU fallback with accurate runtime detection
- **Robust error handling** — fatal vs. advisory error contracts; checkpoint-based resume; clean directory stack unwinding on all error paths
- **Quality control** — per-subject QC reports with corrected SSIM metrics, JSONL structured logs, and HTML pipeline summary

---

## Pipeline Stages

| Stage | Description | Error Contract |
|-------|-------------|---------------|
| 1 | Basic preprocessing (denoise, Gibbs correction, DWI metrics) | Fatal |
| 2 | Eddy current & bias correction | Fatal |
| 3 | Post-hoc refinement & ML registration | Advisory |
| 4 | Connectivity analysis & tractography | Advisory |
| 5 | NODDI parameter estimation | Advisory |

---

## Dependencies

### Neuroimaging Tools
- [FSL](https://fsl.fmrib.ox.ac.uk/) (including `eddy`, `topup`, `bet`)
- [MRtrix3](https://www.mrtrix.org/)
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)
- [ANTs](http://stnava.github.io/ANTs/)

### Python / ML
- Python 3.7+ (virtualenv or conda recommended)
- TensorFlow ≥ 2.x
- [VoxelMorph](https://github.com/voxelmorph/voxelmorph)
- scikit-learn, scipy
- nibabel, numpy

### Containers (for Synb0-DisCo)
- Docker, **or**
- Singularity ≥ 3.x, **or**
- Apptainer ≥ 1.x

### System
- Bash 4+
- CUDA 12.3+ (optional, for GPU acceleration)
- `awk` (POSIX-standard; used for early validation where `bc` may not yet be confirmed)

---

## Installation

```bash
git clone https://github.com/travisbeckwith/DTI_ML_project.git
cd DTI_ML_project
chmod +x ML_v8_beta.sh
```

Activate your Python environment before running:

```bash
# conda
conda activate neuroimaging_env

# or virtualenv
source /path/to/venv/bin/activate
```

If MRtrix3 is installed outside your system `PATH`, set the `MRTRIX_HOME` environment variable so the pipeline can locate its binaries:

```bash
export MRTRIX_HOME=/opt/mrtrix3   # adjust to your installation
```

---

## Configuration

Before running, set the three required paths. You can do this in one of two ways:

### Option A: Edit the configuration block

Open `ML_v8_beta.sh` and fill in the `USER CONFIGURATION` block near the top:

```bash
USER_BIDS_DIR="/data/study01/BIDS"         # Root of your BIDS dataset
USER_STORAGE_FAST="/scratch/study01"        # Fast SSD for pipeline outputs
USER_STORAGE_LARGE="/archive/study01"       # Large drive for FreeSurfer outputs
```

### Option B: Pass paths on the command line

```bash
./ML_v8_beta.sh \
  --bids /data/study01/BIDS \
  --storage-fast /scratch/study01 \
  --storage-large /archive/study01
```

CLI flags override the configuration block. The pipeline will abort with a clear message if any path is unset.

### Storage layout

| Path | Purpose | Recommended disk | Size estimate |
|------|---------|-----------------|---------------|
| `BIDS_DIR` | Input data + processing intermediates | Fast SSD | ~100 GB free |
| `--storage-fast` | Synb0, MRtrix3, post-hoc, and QC outputs | Fast SSD | ~100 GB free |
| `--storage-large` | FreeSurfer recon-all outputs | Large capacity | ~200–300 GB free |

---

## Usage

```bash
# All subjects
./ML_v8_beta.sh

# Single subject
./ML_v8_beta.sh -s sub-001 --omp-threads 8

# ML-enhanced registration with auto-install
./ML_v8_beta.sh -s sub-001 --use-ml-registration --auto-install-ml

# Specific ML method
./ML_v8_beta.sh --ml-method synthmorph --ml-full-mode

# Skip connectivity analysis
./ML_v8_beta.sh --skip-connectome

# Dry run — preview without executing
./ML_v8_beta.sh --dry-run

# Resume from last checkpoint
./ML_v8_beta.sh --resume

# Override container runtime
./ML_v8_beta.sh --container-cmd singularity
```

---

## CLI Reference

### Basic options

| Flag | Description |
|------|-------------|
| `-b`, `--bids <dir>` | Path to BIDS directory |
| `-s`, `--subject <id>` | Process a single subject |
| `--pe <dir>` | Phase encoding direction: `AP`, `PA`, `LR`, `RL` (default: `AP`) |
| `--echo <val>` | Echo spacing in seconds (default: `0.062`) |
| `--slm-model <model>` | Eddy SLM model: `linear` or `quadratic` (default: `linear`) |
| `--omp-threads <n>` | OpenMP threads (default: auto-detected) |
| `--storage-fast <path>` | Fast SSD for outputs |
| `--storage-large <path>` | Large drive for FreeSurfer |
| `--config <file>` | Load options from a configuration file |

### Processing toggles

| Flag | Description |
|------|-------------|
| `--skip-synb0` | Skip Synb0-DisCo distortion correction |
| `--skip-connectome` | Skip connectivity analysis |
| `--no-cleanup` | Keep temporary files |
| `--dry-run` | Preview execution plan without running |
| `--resume` | Resume from last successful checkpoint |
| `--container-cmd <cmd>` | Override container runtime (`docker`, `singularity`, `apptainer`) |

### ML registration options

| Flag | Description |
|------|-------------|
| `--use-ml-registration` | Enable ML-based registration |
| `--ml-method <method>` | `auto`, `voxelmorph`, `synthmorph`, or `ants` (default: `auto`) |
| `--ml-model-path <path>` | Path to custom ML model weights |
| `--auto-install-ml` | Auto-install missing ML Python packages |
| `--ml-quick-mode` | Fast ML registration (default) |
| `--ml-full-mode` | Full ML registration (slower, more accurate) |
| `--force-gpu` | Force GPU usage |
| `--skip-quality-check` | Skip registration quality assessment |

### Backward compatibility

The legacy flags `--storage-e` and `--storage-f` are still accepted as aliases for `--storage-fast` and `--storage-large`.

---

## Output Structure

```
<storage-fast>/derivatives/
├── synb0-disco/
│   └── sub-001/                       # Synb0 distortion correction outputs
├── mrtrix3/
│   ├── sub-001_fa.nii.gz              # Fractional anisotropy
│   ├── sub-001_md.nii.gz              # Mean diffusivity
│   ├── sub-001_ndi.nii.gz             # NODDI — neurite density index
│   ├── sub-001_odi.nii.gz             # NODDI — orientation dispersion
│   ├── sub-001_fwf.nii.gz             # NODDI — free water fraction
│   └── sub-001_connectome_*.csv       # Structural connectomes
├── posthoc/
│   └── sub-001/                       # Post-hoc refinement outputs
└── qc_integrated/
    ├── sub-001_qc.txt                 # Per-subject QC report
    └── pipeline_final_report.txt      # Pipeline summary

<storage-large>/derivatives/
└── freesurfer/
    └── sub-001/                       # FreeSurfer recon-all output
```

---

## Companion Files

| File | Description |
|------|-------------|
| `ML_v8_beta.sh` | Main pipeline script |
| `voxelmorph_registration.py` | Externalized VoxelMorph registration (co-locate with script) |
| `noddi_fitting.py` | Externalized NODDI fitting (co-locate with script) |
| `test_helpers.sh` | Unit tests for pure-logic helper functions |

---

## Error Handling

Functions follow one of two contracts:

- **Fatal** — pipeline aborts for the subject on failure (`run_basic_preprocessing`, `run_eddy_and_bias_correction`)
- **Advisory** — failure is logged and processing continues (`run_synb0`, `run_posthoc_refinement`, all ML registration functions)

As of v8, all functions that use `safe_cd` (pushd) properly call `safe_cd_return` (popd) on every error exit, preventing directory stack corruption during partial failures.

---

## Changelog

### v1.4-ml-enhanced beta (v8)

**Bug fixes**
1. **GPU log message** — no longer unconditionally prints "RTX 3070 ready"; now reflects actual GPU detection state (GPU detected / CPU-only mode)
2. **Hardcoded developer path removed** — `/home/trav/mrtrix3/bin` replaced with `${MRTRIX_HOME}` auto-expansion in both PATH exports
3. **ANTs `--initial-moving-transform` missing parameter** — added third argument (`1` = center-of-mass initialization); some ANTs versions error without it
4. **Directory stack leaks on error paths** — added `safe_cd_return` before every early `return 1` in 5 functions: `run_basic_preprocessing`, `run_eddy_and_bias_correction`, `run_posthoc_refinement`, `enhance_t1w_for_freesurfer`, and their sub-paths (~15 exit points fixed)
5. **Uncompressed NIfTI copied with `.nii.gz` extension** — fallback paths now use `gzip -c` instead of bare `cp` when `mrconvert` is unavailable, preventing format-detection failures in downstream tools
6. **SSIM constants assumed unit dynamic range** — stabilization constants now computed from actual data range (`L = max - min`), preventing near-zero denominators on MRI intensity data (typical range 0–thousands)
7. **Misleading `mask_ratio` label** — changed `{ratio:.1f} of image` to `{ratio:.1%} of image` so 0.3 displays as "30.0%" rather than "0.3 of image"
8. **Redundant `FSLDIR` export** — removed duplicate `export FSLDIR=/usr/local/fsl` inside `check_required_tools` (already set at script startup)
9. **`DOCKER_CMD` referenced before initialization** — `cleanup_aggressive()` and signal handler now use `CONTAINER_CMD` which is set earlier; `DOCKER_CMD` alias is only created in `setup_environment`
10. **Echo spacing validation used `bc` before availability check** — replaced with `awk` which is always available on POSIX systems

**SynthMorph implementation completed**
11. **`extract_synthmorph_transform()`** — was a stub that only ran `mri_convert`. Now performs full RAS→LPS displacement field conversion via nibabel so the warp is compatible with ANTs/ITK. Validates displacement field shape, negates R and A components, and saves as a proper 5-D ITK vector image.
12. **`apply_ml_transform_to_5tt()` SynthMorph case** — was a no-op (`cp`). Now splits the 5-tissue-type image by volume, applies the ITK displacement field to each volume via `antsApplyTransforms` with Linear interpolation, and reassembles the result.
13. **`apply_ml_transform_to_parcellation()` SynthMorph case** — was a no-op (`mv`). Now applies the ITK displacement field via `antsApplyTransforms` with NearestNeighbor interpolation to preserve integer parcel labels.

### v1.3-ml-enhanced beta (v7)

**Critical fixes**
1. Defined missing `run_synthmorph_t1_dwi_registration()` and `run_enhanced_ants_t1_dwi_registration()` — previously caused "command not found" aborts when `USE_ML_REGISTRATION=true`
2. Fixed PYTHONPATH ordering in `setup_environment()` — FSL packages were prepending over venv, breaking TensorFlow/VoxelMorph imports
3. Fixed gradient file path derivation in `check_connectivity_readiness()` — `_preproc` suffix caused lookup failures

**Moderate fixes**
4. Replaced `((errors++))` with `errors=$((errors + 1))` — safe under `set -e`
5. Changed `return $errors` to `return $(( errors > 0 ? 1 : 0 ))` — prevents silent success when error count wraps past 255
6. Fixed `safe_int()` regex — negative numbers no longer stripped to empty string
7. Fixed `num_dirs` calculation — now counts gradient directions (`head -1 | wc -w`) instead of file lines (always 3)
8. Synb0-DisCo container invocation refactored — supports Docker, Singularity, and Apptainer via `case` dispatch

**Low-severity fixes**
9. Header version string updated to match `SCRIPT_VERSION`
10. `enhance_brain_mask()` Method 3 indentation corrected
11. `cleanup_aggressive()` only runs `docker system prune` when runtime is actually Docker
12. All `nib.load('*.mif')` calls replaced with `mrconvert` → `.nii.gz` pre-conversion (4 locations)
13. Added cleanup of temporary QC `.nii.gz` files

**Portability**
- Replaced all hardcoded paths (`/mnt/c/CLS/...`, `/mnt/e/CLS`, `/mnt/f/CLS`) with user-configurable placeholders
- Added `USER CONFIGURATION` block at top of script
- Renamed `--storage-e` / `--storage-f` to `--storage-fast` / `--storage-large` (old names kept as aliases)
- Genericized all log messages, comments, and reports (no more "C drive", "E drive", "F drive")
- Storage directories are auto-created if they don't exist

---

## Citation

If you use this pipeline in your research, please cite the relevant tools:

- Tournier et al. (2019) MRtrix3. *NeuroImage* 202, 116137
- Jenkinson et al. (2012) FSL. *NeuroImage* 62(2), 782–790
- Balakrishnan et al. (2019) VoxelMorph. *IEEE TMI* 38(8), 1788–1800
- Fischl (2012) FreeSurfer. *NeuroImage* 62(2), 774–781

---

## Author

**Travis Beckwith, Ph.D.**
Neuroimaging Scientist | Bloomington, IN
[travis.beckwith@gmail.com](mailto:travis.beckwith@gmail.com) · [ORCID](https://orcid.org/0000-0001-6128-8464) · [Google Scholar](https://scholar.google.com/citations?user=wolY848AAAAJ&hl=en)
