# DWIForge v2

A modular, checkpoint-based diffusion MRI preprocessing and analysis pipeline for single-shell and multi-shell acquisitions. Designed for Philips scanners but applicable to any BIDS-compliant dataset.

## Pipeline Stages

| Stage | Name | Description |
|-------|------|-------------|
| 00 | QC BIDS | Validate BIDS layout, compute SNR, detect acquisition parameters |
| 01 | recon-all | FreeSurfer cortical reconstruction |
| 02 | Preprocessing | MP-PCA denoising → Gibbs unringing (MRtrix3) |
| 03 | T1w Prep | Reorientation → N4 bias correction → SynthStrip skull-stripping |
| 04 | EPI Correction | Synb0-DisCo synthetic b0 → topup field estimation |
| 05 | DESIGNER | topup + eddy + Rician denoising (DESIGNER2) |
| 06 | Tensor Fitting | DTI via tmi (DESIGNER2); FA, MD, AD, RD, eigenvectors |
| 07 | NODDI | AMICO 2.x NODDI fitting; NDI, ODI, ISOVF |
| 08 | Response Functions | dhollander 3-tissue response estimation |
| 09 | Tractography | SS3T-CSD FODs → iFOD2 ACT (10M) → SIFT2 → DK84 connectome |
| 10 | QC Report | Per-subject PDF with slice mosaics, metrics table, connectome matrix |

## Requirements

- **MRtrix3** ≥ 3.0.8
- **MRtrix3Tissue** (for ss3t_csd_beta1) — patch `mrtrix3.py` for Python 3.12:
  ```bash
  cp ~/mrtrix3/bin/mrtrix3.py ~/MRtrix3Tissue/bin/mrtrix3.py
  ```
- **FSL** ≥ 6.0
- **FreeSurfer** ≥ 7.x
- **ANTs** ≥ 2.4
- **Docker** (for Synb0-DisCo) or Apptainer/Singularity
- **Python 3.12** with `neuroimaging_env` (see `env/`)
  - DESIGNER2 (`designer2`, `tmi`)
  - AMICO 2.x
  - dipy ≥ 1.12, nibabel, matplotlib, scipy

See `env/DEPENDENCIES.md` and `env/Environment_Instructions.md` for full setup.

## Quick Start

```bash
# 1. Copy and configure
cp dwiforge.toml my_study.toml
# Edit my_study.toml — set [paths] source, work, output, freesurfer, logs

# 2. Run a single subject
./dwiforge.sh --config my_study.toml --subject sub-001

# 3. Run specific stages only
./dwiforge.sh --config my_study.toml --subject sub-001 --only-stage tensor-fitting

# 4. Resume after a failure
./dwiforge.sh --config my_study.toml --subject sub-001 --resume

# 5. Rerun a specific stage
./dwiforge.sh --config my_study.toml --subject sub-001 \
    --only-stage noddi --rerun-stage noddi
```

## Configuration

All paths and options are set in `dwiforge.toml`. The key sections are:

```toml
[paths]
source     = "/path/to/BIDS"        # BIDS input directory
work       = "/path/to/work"         # per-subject working directory
output     = "/path/to/output"       # final outputs
freesurfer = "/path/to/freesurfer"  # FreeSurfer subjects dir
logs       = "/path/to/logs"

[runtime]
designer_bin         = ""  # auto-detected if empty
designer_python_path = ""  # auto-detected if empty
```

## Multi-subject / SLURM

See `container/slurm_example.sh` for a SLURM array job template.

The `responsemean` step (stage 08→09 barrier) must run after all subjects complete stage 08:

```bash
# After all stage-08 jobs complete:
PYTHONPATH=/path/to/mrtrix3/lib \
    responsemean Work/group/responses/*/response_wm.txt \
    Work/group/group_response_wm.txt -force
# repeat for gm, csf
touch Work/group/responsemean.done
```

## Notes

- Synb0-DisCo runs via Docker (`leonyichencai/synb0-disco:v3.1 --notopup`)
- DESIGNER output uses **eddy-rotated bvecs** when building `dwi_preprocessed.mif` — critical for correct tensor orientation
- Single-shell NODDI (b=1000, ~32 dirs) gives valid but lower-precision estimates vs multi-shell
- Parcellation is registered from T1w → DWI space before connectome construction
