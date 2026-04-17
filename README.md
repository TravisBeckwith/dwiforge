# dwiforge
### `dwiforge.sh` — v1.4

A comprehensive, single-script Bash pipeline for diffusion tensor imaging (DTI) processing, structural connectivity analysis, and microstructure modeling. Designed for BIDS-formatted datasets on Linux/WSL2 with support for multi-drive storage layouts.

*Note* New iteration under construction due to changes being made to VoxelMorph

---

## Features

- **Full DTI preprocessing** — denoising, Gibbs ringing correction, susceptibility and eddy current distortion correction via Synb0-DisCo and FSL eddy
- **ML-enhanced T1w–DWI registration** — automatic selection between SynthMorph (FreeSurfer), VoxelMorph, and ANTs with GPU acceleration
- **Post-hoc refinement** — N4 bias field correction, enhanced brain masking, residual distortion assessment
- **Structural connectivity** — FreeSurfer `recon-all`, CSD tractography (10M streamlines), SIFT2 filtering, Desikan-Killiany atlas connectome (84×84)
- **NODDI microstructure modeling** — NDI, ODI, and FWF maps via AMICO
- **Multiple connectome weightings** — streamline count, FA-weighted, length-weighted, SIFT2-weighted
- **Checkpoint-based resumption** — each stage checkpointed; safely resumable after interruption
- **Comprehensive QC reports** — per-subject text and HTML reports with SNR estimates, FreeSurfer quality assessment, and connectivity metrics

---

## Pipeline Stages

| Stage | Description | Key Tools |
|-------|-------------|-----------|
| **1 — Preprocessing** | DWI denoising → Gibbs correction → Synb0 fieldmap → eddy correction → N4 bias correction | MRtrix3, FSL eddy, Docker/Synb0-DisCo |
| **2 — Post-hoc Refinement** | Enhanced masking, ML T1w–DWI registration, residual distortion QC | FSL, ANTs, SynthMorph, VoxelMorph |
| **3 — Connectivity Analysis** | FreeSurfer recon-all → 5TT → CSD FODs → tractography → SIFT2 → connectome | FreeSurfer, MRtrix3 |
| **4 — NODDI Estimation** | AMICO-based NODDI model fitting → NDI/ODI/FWF maps | AMICO, Python |

---

## Repository Structure

```
dwiforge/
├── dwiforge.sh                    # Main pipeline script
├── voxelmorph_registration.py     # VoxelMorph DWI registration (optional — generated inline if absent)
├── noddi_fitting.py               # NODDI fitting script (optional — generated inline if absent)
├── test_helpers.sh                # Unit tests for helper functions
├── README.md                      # This file
├── CHANGELOG.md                   # Full version history
├── KNOWN_ISSUES.md                # Confirmed bugs, workarounds, and investigation notes
└── LICENSE
```

The two Python scripts are optional companions. If placed in the same directory as `dwiforge.sh` they are used directly, which allows independent linting, editing, and testing. If absent, the pipeline generates equivalent scripts inline from embedded heredocs — behaviour is identical either way.

To run the unit tests:
```bash
bash test_helpers.sh ./dwiforge.sh
```

---

## Requirements

### System
- Linux or WSL2 (Windows Subsystem for Linux)
- 16+ GB RAM recommended (26 GB for 10M streamline tractography)
- NVIDIA GPU recommended for ML registration and NODDI (CUDA-capable)
- ~50 GB working space on BIDS drive; ~300 GB on large storage for FreeSurfer

### Required Software
| Tool | Version Tested | Purpose |
|------|---------------|---------|
| FSL | 6.0+ | eddy, brain extraction, registration |
| MRtrix3 | 3.0+ | DWI processing, tractography, connectomes |
| FreeSurfer | 7.4.1 | Cortical parcellation, 5TT |
| ANTs | 2.4+ | Fallback registration |
| Docker or Singularity | any | Synb0-DisCo container |
| Python | 3.10+ | AMICO, VoxelMorph, SynthMorph |

### Python Packages (virtualenv recommended)
```
tensorflow>=2.13
voxelmorph
amico>=2.1
scikit-learn
nibabel
numpy
```

Install into a virtualenv:
```bash
python3 -m venv ~/neuroimaging_env
source ~/neuroimaging_env/bin/activate
pip install tensorflow voxelmorph amico scikit-learn nibabel numpy
```

---

## Input Data

BIDS-formatted dataset with:
```
BIDS_DIR/
└── sub-<id>/
    ├── anat/
    │   └── sub-<id>_T1w.nii.gz
    └── dwi/
        ├── sub-<id>_dwi.nii.gz
        ├── sub-<id>_dwi.bval
        ├── sub-<id>_dwi.bvec
        └── sub-<id>_acq-PA_dwi.nii.gz   # reverse phase-encode b0 for Synb0
```

---

## Configuration

Edit the three path variables near the top of the script before running:

```bash
USER_BIDS_DIR="/path/to/BIDS"         # BIDS dataset root
USER_STORAGE_FAST="/path/to/fast"     # SSD — for MRtrix3/NODDI outputs and QC
USER_STORAGE_LARGE="/path/to/large"   # Large drive — for FreeSurfer recon-all
```

Or pass them on the command line (see Usage below).

---

## Usage

### Basic run — all stages, all subjects
```bash
source ~/neuroimaging_env/bin/activate
./dwiforge.sh
```

### Single subject with ML registration and 8 threads
```bash
./dwiforge.sh -s sub-001 --omp-threads 8 --use-ml-registration --ml-method auto
```

### Override paths on command line
```bash
./dwiforge.sh \
  -b /data/study/BIDS \
  --storage-fast /scratch/study \
  --storage-large /archive/study \
  -s sub-001
```

### Preview without executing
```bash
./dwiforge.sh --dry-run
```

### Resume after interruption
```bash
./dwiforge.sh -s sub-001 --resume
```

### Skip optional stages
```bash
./dwiforge.sh --skip-synb0 --skip-connectome
```

---

## Full Argument Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `-b`, `--bids` | script default | Path to BIDS directory |
| `-s`, `--subject` | all subjects | Process single subject only |
| `--storage-fast` / `--storage-e` | script default | Fast storage path |
| `--storage-large` / `--storage-f` | script default | Large storage path |
| `--omp-threads` | auto | OpenMP thread count |
| `--skip-synb0` | false | Skip Synb0-DisCo fieldmap |
| `--skip-connectome` | false | Skip Stage 3 connectivity |
| `--no-cleanup` | false | Retain intermediate files |
| `--use-ml-registration` | false | Enable ML-based T1w–DWI registration |
| `--ml-method` | `auto` | ML method: `auto`, `synthmorph`, `voxelmorph`, `ants` |
| `--ml-quick-mode` | true | Faster ML registration (less accurate) |
| `--ml-full-mode` | false | Full ML registration |
| `--force-gpu` | false | Require GPU (abort if unavailable) |
| `--skip-quality-check` | false | Skip ML registration quality assessment |
| `--ml-model-path` | built-in | Path to custom VoxelMorph model |
| `--pe` | `AP` | Phase encoding direction |
| `--echo` | `0.062` | Echo spacing (ms) |
| `--slm-model` | `linear` | eddy second-level model |
| `--dry-run` | false | Preview without executing |
| `--resume` | false | Resume from last checkpoint |
| `--container-cmd` | auto | Container runtime (`docker`/`singularity`/`apptainer`) |
| `--config` | none | Load settings from config file |
| `-h`, `--help` | — | Show usage |
| `--help-ml` | — | Show ML-specific help |

---

## Outputs

All outputs are organized under the storage paths specified in configuration.

### Fast storage (`STORAGE_FAST/derivatives/`)
```
mrtrix3/sub-<id>/
├── sub-<id>_dwi_preproc.nii.gz       # Preprocessed DWI
├── sub-<id>_fa.nii.gz                # Fractional anisotropy
├── sub-<id>_md.nii.gz                # Mean diffusivity
├── sub-<id>_rd.nii.gz                # Radial diffusivity
├── sub-<id>_ad.nii.gz                # Axial diffusivity
├── sub-<id>_ev.nii.gz                # Primary eigenvector, FA-modulated (signed)
├── sub-<id>_dec.nii.gz               # DEC map for RGB visualisation (abs of ev)
├── sub-<id>_ndi.nii.gz               # Neurite density index (NODDI)
├── sub-<id>_odi.nii.gz               # Orientation dispersion index (NODDI)
├── sub-<id>_fwf.nii.gz               # Free water fraction (NODDI)
├── sub-<id>_connectome_dk.csv        # Primary streamline connectome (84×84)
├── sub-<id>_connectome_fa.csv        # FA-weighted connectome
├── sub-<id>_connectome_length.csv    # Length-weighted connectome
└── sub-<id>_connectome_sift2.csv     # SIFT2-weighted connectome

qc_integrated/
├── sub-<id>_qc.txt                   # Main QC report
├── sub-<id>_connectivity_comprehensive.txt
├── pipeline_final_report.txt
└── pipeline_report.html              # HTML summary
```

### Large storage (`STORAGE_LARGE/derivatives/`)
```
freesurfer/sub-<id>/                  # Full FreeSurfer recon-all output
```

### BIDS derivatives (`BIDS_DIR/derivatives/`)
```
logs/
├── pipeline_run*.log                 # Full pipeline logs
├── sub-<id>_progress.json            # Stage progress tracking
├── checkpoints/sub-<id>_checkpoints.txt
└── pipeline_events.jsonl             # Structured event log

work/sub-<id>/                        # Intermediate files (removed if --no-cleanup not set)
```

### Eigenvector outputs — ev vs dec

Two eigenvector files are produced per subject:

- **`sub-<id>_ev.nii.gz`** — signed primary eigenvector, FA-modulated. Use this for tractography and any quantitative analysis. Sign is meaningful and must not be discarded.
- **`sub-<id>_dec.nii.gz`** — absolute value of ev, FA-modulated. Use this for RGB/DEC visualisation in viewers that do not apply abs() automatically (FSLeyes, ITK-SNAP). mrview users can load `ev.nii.gz` directly as mrview applies abs() internally.

---

## Checkpoint System

The pipeline uses a two-level checkpoint system to support safe resumption:

1. **Stage-level** (`sub-<id>_checkpoints.txt`) — records completed stages (e.g., `connectivity_complete`)
2. **Progress JSON** (`sub-<id>_progress.json`) — tracks fine-grained progress percentages

To rerun a completed stage, remove its checkpoint entry:
```bash
sed -i '/^connectivity_complete/d' \
    derivatives/logs/checkpoints/sub-001_checkpoints.txt
```

For connectivity, also remove the output CSV (the pipeline uses file existence as a secondary guard):
```bash
rm -f /path/to/fast/derivatives/mrtrix3/sub-001/sub-001_connectome_dk.csv
```

---

## WSL2 Notes

When running on Windows Subsystem for Linux 2, Windows drives must be mounted before the pipeline starts. The pipeline will detect unmounted drives and exit with clear instructions, but it's best to mount them first:

```bash
sudo mkdir -p /mnt/e /mnt/f
sudo mount -t drvfs E: /mnt/e
sudo mount -t drvfs F: /mnt/f
```

To make mounts persistent across WSL sessions, add them to `/etc/fstab`:
```
E: /mnt/e drvfs defaults 0 0
F: /mnt/f drvfs defaults 0 0
```

Always activate the Python virtualenv before running:
```bash
source ~/neuroimaging_env/bin/activate
./dwiforge.sh ...
```

---

## Tested Configuration

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 (WSL2) |
| FreeSurfer | 7.4.1 |
| FSL | 6.0.7 |
| MRtrix3 | 3.0.4 |
| Python | 3.12.12 |
| TensorFlow | 2.19.1 |
| AMICO | 2.1.0 |
| VoxelMorph | 0.2 |
| GPU | NVIDIA RTX 3070 8GB |
| RAM | 27 GB |

---

## Known Limitations

- Single-shell DWI (one non-zero b-value) is supported; multi-shell uses optimized CSD automatically
- Synb0-DisCo requires Docker or Singularity and a FreeSurfer license
- SIFT2 requires a 5TT image and FOD image; falls back to uniform weights if these are unavailable
- NODDI validation requires AMICO 2.1+; older API (`get_params()`) is handled automatically
- TensorFlow startup may log duplicate CUDA factory registration warnings — these are benign and do not affect processing

---

## License

See `LICENSE` in the repository root.
