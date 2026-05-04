#!/usr/bin/env python3
"""
python/qc_bids.py — BIDS validation and capability profiling for dwiforge
==========================================================================
Analyses a single subject's DWI data and writes a capability profile JSON
that all downstream stages read to decide whether and how to run.

Exit codes:
  0  QC passed (warnings may be present)
  1  One or more critical failures — subject should not be processed

Usage:
  python qc_bids.py <subject_id> <bids_source_dir> <output_json> \\
      [--b0-threshold 50] \\
      [--noddi-min-directions 30] \\
      [--noddi-high-directions 60] \\
      [--run-ndc] \\
      [--t1w-search-dir /path]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

DWIFORGE_VERSION = os.environ.get("DWIFORGE_VERSION", "2.0")

# ---------------------------------------------------------------------------
# Direction / capability thresholds
# ---------------------------------------------------------------------------

DTI_MIN_DIRECTIONS      = 6
NODDI_MIN_DIRECTIONS    = 30   # overridden by --noddi-min-directions
NODDI_HIGH_DIRECTIONS   = 60   # overridden by --noddi-high-directions
DKI_MIN_DIRECTIONS      = 15   # minimum per shell for reliable DKI
DKI_MIN_SHELLS          = 2
CSD_MIN_DIRECTIONS      = 45   # for reliable fibre ODF estimation
MSMT_MIN_SHELLS         = 2

# b-value clustering tolerance (values within this range → same shell)
BVAL_CLUSTER_TOLERANCE  = 50   # s/mm²

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class QCResult:
    def __init__(self, sub: str):
        self.sub              = sub
        self.critical: list[str] = []
        self.warnings: list[str] = []
        self.data:     dict[str, Any] = {}
        self.shells:   dict[str, Any] = {}
        self.caps:     dict[str, Any] = {}
        self.acq:      dict[str, Any] = {}
        self.quality:  dict[str, Any] = {}

    def fail(self, msg: str) -> None:
        self.critical.append(msg)
        print(f"[{self.sub}] CRITICAL: {msg}", file=sys.stderr)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"[{self.sub}] WARNING:  {msg}")

    def info(self, msg: str) -> None:
        print(f"[{self.sub}] INFO:     {msg}")

    @property
    def passed(self) -> bool:
        return len(self.critical) == 0

    def to_dict(self) -> dict:
        return {
            "subject":           self.sub,
            "generated":         datetime.now(timezone.utc).isoformat(),
            "dwiforge_version":  DWIFORGE_VERSION,
            "data":              self.data,
            "shells":            self.shells,
            "capabilities":      self.caps,
            "acquisition":       self.acq,
            "quality":           self.quality,
            "warnings":          self.warnings,
            "critical_failures": self.critical,
            "qc_passed":         self.passed,
        }


# ---------------------------------------------------------------------------
# Step 1: File discovery
# ---------------------------------------------------------------------------

def discover_files(
    sub: str,
    source_dir: Path,
    result: QCResult,
) -> dict[str, Path | None]:
    """Locate DWI NIfTI, bval, bvec, JSON sidecar, reverse-PE b0, and T1w."""

    dwi_dir = source_dir / sub / "dwi"
    anat_dir = source_dir / sub / "anat"

    # ---- Primary DWI NIfTI ----
    dwi_nii: Path | None = None
    for pattern in [
        f"{sub}_dir-AP_dwi.nii.gz",
        f"{sub}_dwi.nii.gz",
    ]:
        candidate = dwi_dir / pattern
        if candidate.exists():
            dwi_nii = candidate
            break
    # Broader glob fallback
    if dwi_nii is None:
        hits = sorted(dwi_dir.glob("*_dwi.nii.gz"))
        if hits:
            dwi_nii = hits[0]
            if len(hits) > 1:
                result.warn(
                    f"Multiple DWI NIfTI files found; using {dwi_nii.name}. "
                    f"Others: {[h.name for h in hits[1:]]}"
                )

    if dwi_nii is None:
        result.fail(f"No DWI NIfTI found under {dwi_dir}")
        return {}

    # ---- bval / bvec ----
    stem = dwi_nii.name.replace(".nii.gz", "")
    bval = dwi_dir / f"{stem}.bval"
    bvec = dwi_dir / f"{stem}.bvec"

    # BIDS fallback: any .bval/.bvec in dwi/
    if not bval.exists():
        candidates = sorted(dwi_dir.glob("*.bval"))
        if candidates:
            bval = candidates[0]
    if not bvec.exists():
        candidates = sorted(dwi_dir.glob("*.bvec"))
        if candidates:
            bvec = candidates[0]

    if not bval.exists():
        result.fail(f"bval file not found (expected {bval})")
    if not bvec.exists():
        result.fail(f"bvec file not found (expected {bvec})")

    # ---- JSON sidecar ----
    json_sidecar = dwi_dir / f"{stem}.json"
    if not json_sidecar.exists():
        # Try without direction entity
        alt = dwi_dir / f"{sub}_dwi.json"
        json_sidecar = alt if alt.exists() else None
        if json_sidecar is None:
            result.warn(
                "No JSON sidecar found — PhaseEncodingDirection and "
                "TotalReadoutTime unavailable; topup will be skipped"
            )

    # ---- Reverse phase-encode b0 ----
    rpe: Path | None = None
    for pattern in [
        f"{sub}_dir-PA_dwi.nii.gz",
        f"{sub}_dir-PA_epi.nii.gz",
        f"{sub}_acq-PA_dwi.nii.gz",
        f"{sub}_acq-revPE_dwi.nii.gz",
    ]:
        candidate = dwi_dir / pattern
        if candidate.exists():
            rpe = candidate
            break

    # ---- T1w ----
    t1w: Path | None = None
    if anat_dir.exists():
        for pattern in [
            f"{sub}_T1w.nii.gz",
            f"{sub}_acq-MPRAGE_T1w.nii.gz",
        ]:
            candidate = anat_dir / pattern
            if candidate.exists():
                t1w = candidate
                break
        if t1w is None:
            hits = sorted(anat_dir.glob("*_T1w.nii.gz"))
            if hits:
                t1w = hits[0]

    result.info(f"DWI:         {dwi_nii.name}")
    result.info(f"bval:        {bval.name if bval.exists() else 'MISSING'}")
    result.info(f"bvec:        {bvec.name if bvec.exists() else 'MISSING'}")
    result.info(f"JSON:        {json_sidecar.name if json_sidecar else 'not found'}")
    result.info(f"Reverse PE:  {rpe.name if rpe else 'not found'}")
    result.info(f"T1w:         {t1w.name if t1w else 'not found'}")

    if not result.passed:
        return {}

    return {
        "dwi_nii":      dwi_nii,
        "bval":         bval,
        "bvec":         bvec,
        "json_sidecar": json_sidecar,
        "rpe":          rpe,
        "t1w":          t1w,
    }


# ---------------------------------------------------------------------------
# Step 2: NIfTI integrity
# ---------------------------------------------------------------------------

def check_nifti(
    files: dict,
    result: QCResult,
    b0_threshold: int,
) -> dict[str, Any] | None:
    """Load and validate the DWI NIfTI. Returns data dict or None on failure."""
    import numpy as np
    import nibabel as nib

    nii_path = files["dwi_nii"]
    try:
        img = nib.load(str(nii_path))
    except Exception as exc:
        result.fail(f"Cannot load NIfTI: {exc}")
        return None

    shape = img.shape
    if len(shape) != 4:
        result.fail(f"Expected 4D NIfTI, got {len(shape)}D (shape: {shape})")
        return None

    n_vols = shape[3]
    voxel_size = [round(float(v), 4) for v in img.header.get_zooms()[:3]]
    matrix = list(shape[:3])

    # Load data for signal checks (memory-mapped)
    try:
        data = img.get_fdata(dtype=np.float32)
    except Exception as exc:
        result.fail(f"Cannot read NIfTI data array: {exc}")
        return None

    # All-zero check
    if data.max() == 0:
        result.fail("DWI data is all zeros — NIfTI appears corrupt or empty")
        return None

    # NaN/Inf check
    n_nan = int(np.isnan(data).sum())
    n_inf = int(np.isinf(data).sum())
    if n_nan > 0:
        result.warn(f"{n_nan} NaN values found in DWI data — will be zeroed during preprocessing")
    if n_inf > 0:
        result.warn(f"{n_inf} Inf values found in DWI data — will be zeroed during preprocessing")

    # Voxel size sanity
    for i, vs in enumerate(voxel_size):
        if vs <= 0 or vs > 10:
            result.warn(
                f"Unusual voxel size in dimension {i}: {vs} mm "
                "(expected 1–5 mm for typical DWI)"
            )

    result.info(f"Shape:      {shape}")
    result.info(f"Voxel size: {voxel_size} mm")

    return {
        "dwi_file":       str(nii_path),
        "n_volumes":      n_vols,
        "matrix":         matrix,
        "voxel_size_mm":  voxel_size,
        "n_nan":          n_nan,
        "n_inf":          n_inf,
        "_data":          data,   # kept in memory for quality checks; removed before JSON
        "_affine":        img.affine,
    }


# ---------------------------------------------------------------------------
# Step 3: Gradient table validation
# ---------------------------------------------------------------------------

def check_gradients(
    files: dict,
    nifti_info: dict,
    result: QCResult,
    b0_threshold: int,
) -> dict[str, Any] | None:
    """Parse and validate bval/bvec. Returns gradient info dict or None."""
    import numpy as np

    n_vols = nifti_info["n_volumes"]

    # ---- Load bvals / bvecs — concatenate all runs if present ----
    all_bval_files = files.get("all_bval", [files["bval"]])
    all_bvec_files = files.get("all_bvec", [files["bvec"]])

    bvals_list = []
    for bf in all_bval_files:
        try:
            bv = np.loadtxt(str(bf))
            if bv.ndim == 0:
                bv = bv.reshape(1)
            bvals_list.append(bv)
        except Exception as exc:
            result.fail(f"Cannot read bval file {bf}: {exc}")
            return None
    bvals = np.concatenate(bvals_list)

    bvecs_list = []
    for bf in all_bvec_files:
        try:
            bvr = np.loadtxt(str(bf))
            # Normalise to (3, N) before concatenating
            if bvr.ndim == 1:
                bvr = bvr.reshape(3, 1)
            elif bvr.shape == (3, len(bvals_list[0])):
                pass  # already (3, N)
            else:
                bvr = bvr.T  # (N, 3) → (3, N)
            bvecs_list.append(bvr)
        except Exception as exc:
            result.fail(f"Cannot read bvec file {bf}: {exc}")
            return None

    n_vols_primary = nifti_info["n_volumes"]
    bvecs_raw = np.concatenate(bvecs_list, axis=1)

    # bvecs_raw is already (3, N_total) after concatenation above
    bvecs = bvecs_raw
    n_vols_total = len(bvals)
    if bvecs.shape != (3, n_vols_total):
        result.fail(
            f"Concatenated bvec shape {bvecs.shape} does not match "
            f"total volumes {n_vols_total}"
        )
        return None

    # ---- Dimension agreement ----
    # For multi-run subjects, bvals is concatenated across all runs
    n_vols_total = sum(
        np.loadtxt(str(bf)).shape[-1]
        if np.loadtxt(str(bf)).ndim > 0 else 1
        for bf in all_bval_files
    ) if len(all_bval_files) > 1 else n_vols
    if len(bvals) != n_vols_total:
        result.fail(
            f"Concatenated bval length ({len(bvals)}) does not match "
            f"total NIfTI n_volumes ({n_vols_total})"
        )
        return None

    # ---- NaN/Inf in gradients ----
    if np.any(~np.isfinite(bvals)):
        result.fail("NaN or Inf values in bval file")
        return None
    if np.any(~np.isfinite(bvecs)):
        result.fail("NaN or Inf values in bvec file")
        return None

    # ---- b0 detection ----
    b0_mask   = bvals <= b0_threshold
    dwi_mask  = ~b0_mask
    n_b0      = int(b0_mask.sum())
    n_dwi     = int(dwi_mask.sum())

    if n_b0 == 0:
        result.fail(
            f"No b0 volumes found (b-value ≤ {b0_threshold} s/mm²). "
            "Check b0_threshold or bval file."
        )
        return None

    result.info(f"b0 volumes:  {n_b0}")
    result.info(f"DWI volumes: {n_dwi}")

    # ---- Direction vector checks (non-b0 only) ----
    gradient_issues: list[str] = []

    dwi_vecs = bvecs[:, dwi_mask]
    norms = np.linalg.norm(dwi_vecs, axis=0)

    # Zero-norm vectors on non-b0 volumes
    zero_dirs = np.where(norms < 0.01)[0]
    if len(zero_dirs) > 0:
        msg = f"{len(zero_dirs)} non-b0 volume(s) have zero-length gradient vectors at indices {zero_dirs.tolist()}"
        result.warn(msg)
        gradient_issues.append(msg)

    # Badly non-unit vectors (after zeroes excluded)
    nonzero = norms > 0.01
    if nonzero.any():
        bad_norm = np.where(nonzero & (np.abs(norms - 1.0) > 0.01))[0]
        if len(bad_norm) > 0:
            msg = f"{len(bad_norm)} gradient vector(s) are not unit-normalised (max deviation: {np.abs(norms[nonzero] - 1.0).max():.4f})"
            result.warn(msg)
            gradient_issues.append(msg)

    # Duplicate or near-duplicate directions
    if dwi_vecs.shape[1] > 1:
        dot_matrix = np.abs(dwi_vecs.T @ dwi_vecs)
        np.fill_diagonal(dot_matrix, 0)
        near_dup = int((dot_matrix > 0.99).sum() // 2)
        if near_dup > 0:
            msg = f"{near_dup} near-duplicate gradient direction pair(s) detected"
            result.warn(msg)
            gradient_issues.append(msg)

    n_runs = files.get("n_runs", 1)
    if n_runs > 1:
        result.info(
            f"Gradient table concatenated across {n_runs} runs: "
            f"{n_b0} b0 volumes, {n_dwi} DWI directions total"
        )

    return {
        "bval_file":          str(files["bval"]),
        "bvec_file":          str(files["bvec"]),
        "n_b0":               n_b0,
        "n_dwi":              n_dwi,
        "n_runs":             n_runs,
        "b0_threshold":       b0_threshold,
        "b0_indices":         b0_mask.nonzero()[0].tolist(),
        "gradient_issues":    gradient_issues,
        "_bvals":             bvals,
        "_bvecs":             bvecs,
        "_b0_mask":           b0_mask,
        "_dwi_mask":          dwi_mask,
    }


# ---------------------------------------------------------------------------
# Step 4: Shell structure
# ---------------------------------------------------------------------------

def analyse_shells(
    grad_info: dict,
    result: QCResult,
) -> dict[str, Any]:
    """Cluster b-values into shells and report structure."""
    import numpy as np

    bvals    = grad_info["_bvals"]
    dwi_mask = grad_info["_dwi_mask"]
    dwi_bvals = bvals[dwi_mask]

    # Cluster b-values: sort unique values, merge within tolerance
    shells: list[float] = []
    for bv in sorted(set(dwi_bvals.tolist())):
        if not shells or abs(bv - shells[-1]) > BVAL_CLUSTER_TOLERANCE:
            shells.append(round(bv))
        # else merge into existing shell (keep rounded representative)

    dirs_per_shell: dict[str, int] = {}
    for shell_bval in shells:
        n_dirs = int(
            np.sum(np.abs(bvals - shell_bval) <= BVAL_CLUSTER_TOLERANCE)
        )
        dirs_per_shell[str(int(shell_bval))] = n_dirs
        result.info(f"Shell b={int(shell_bval):5d}: {n_dirs} directions")

    n_shells       = len(shells)
    is_single      = n_shells == 1
    is_multi       = n_shells >= 2
    total_dirs     = sum(dirs_per_shell.values())
    min_shell_dirs = min(dirs_per_shell.values()) if dirs_per_shell else 0

    if min_shell_dirs < DTI_MIN_DIRECTIONS:
        result.fail(
            f"Minimum directions across shells is {min_shell_dirs} "
            f"(need ≥ {DTI_MIN_DIRECTIONS} for DTI)"
        )

    return {
        "count":               n_shells,
        "b_values":            shells,
        "directions_per_shell": dirs_per_shell,
        "total_dwi_directions": total_dirs,
        "min_directions_any_shell": min_shell_dirs,
        "is_single_shell":     is_single,
        "is_multi_shell":      is_multi,
    }


# ---------------------------------------------------------------------------
# Step 5: Capability determination
# ---------------------------------------------------------------------------

def determine_capabilities(
    shells: dict,
    result: QCResult,
    noddi_min: int,
    noddi_high: int,
) -> dict[str, Any]:
    """Map shell structure to per-model capability flags."""

    total_dirs     = shells["total_dwi_directions"]
    min_dirs       = shells["min_directions_any_shell"]
    n_shells       = shells["count"]
    is_single      = shells["is_single_shell"]
    is_multi       = shells["is_multi_shell"]

    # DTI
    dti = total_dirs >= DTI_MIN_DIRECTIONS

    # NODDI confidence rules:
    #
    # Multi-shell (proper NODDI):
    #   min_dirs >= noddi_min  → high confidence
    #     Multi-shell acquisition provides the shell-structure constraints that
    #     single-shell lacks, so reaching the minimum threshold is already
    #     equivalent to high confidence for the full NODDI model.
    #   min_dirs < noddi_min   → disabled
    #
    # Single-shell (AMICO approximation):
    #   total >= noddi_high    → high confidence
    #   total >= noddi_min     → standard confidence
    #   total < noddi_min      → disabled
    if is_multi:
        if min_dirs >= noddi_min:
            noddi      = True
            noddi_conf = "high"
        else:
            noddi      = False
            noddi_conf = None
            result.warn(
                f"NODDI disabled: multi-shell but minimum directions per shell "
                f"= {min_dirs} (need >= {noddi_min})"
            )
    else:
        # Single-shell
        if total_dirs >= noddi_high:
            noddi      = True
            noddi_conf = "high"
        elif total_dirs >= noddi_min:
            noddi      = True
            noddi_conf = "standard"
        else:
            noddi      = False
            noddi_conf = None
            result.warn(
                f"NODDI disabled: {total_dirs} directions "
                f"(need >= {noddi_min} for standard, >= {noddi_high} for high confidence)"
            )

    # DKI — requires multi-shell with adequate directions per shell
    dki = is_multi and min_dirs >= DKI_MIN_DIRECTIONS and n_shells >= DKI_MIN_SHELLS
    if is_multi and not dki:
        result.warn(
            f"DKI disabled: multi-shell detected but minimum directions "
            f"per shell = {min_dirs} (need ≥ {DKI_MIN_DIRECTIONS})"
        )

    # CSD
    csd_single  = total_dirs >= CSD_MIN_DIRECTIONS if is_single else False
    msmt_csd    = is_multi and min_dirs >= CSD_MIN_DIRECTIONS
    if is_single and not csd_single:
        result.warn(
            f"Single-shell CSD disabled: {total_dirs} directions "
            f"(need ≥ {CSD_MIN_DIRECTIONS})"
        )

    # Tractography requires CSD or at minimum DTI (for deterministic)
    tractography = dti  # deterministic tractography possible with DTI
    tractography_advanced = csd_single or msmt_csd

    # Log capability summary
    result.info("─── Capability summary ───────────────────")
    result.info(f"  DTI:              {'✓' if dti else '✗'}")
    result.info(f"  NODDI:            {'✓' if noddi else '✗'}"
                + (f"  [{noddi_conf} confidence]" if noddi else "  [disabled]"))
    result.info(f"  DKI:              {'✓' if dki else '✗'}")
    result.info(f"  CSD (SS):         {'✓' if csd_single else '✗'}")
    result.info(f"  MSMT-CSD:         {'✓' if msmt_csd else '✗'}")
    result.info(f"  Tractography:     {'✓' if tractography else '✗'}")
    result.info("───────────────────────────────────────────")

    return {
        "dti":                       dti,
        "noddi":                     noddi,
        "noddi_confidence":          noddi_conf,
        "noddi_single_shell_approx": is_single and noddi,
        "dki":                       dki,
        "csd_single_shell":          csd_single,
        "msmt_csd":                  msmt_csd,
        "tractography":              tractography,
        "tractography_advanced":     tractography_advanced,
        "noddi_min_threshold":       noddi_min,
        "noddi_high_threshold":      noddi_high,
    }


# ---------------------------------------------------------------------------
# Step 6: JSON sidecar and acquisition metadata
# ---------------------------------------------------------------------------

# Partial Fourier fraction lookup (BIDS float → human label)
_PF_LABELS = {
    1.0:   "none",
    0.875: "7/8",
    0.75:  "6/8",
    0.625: "5/8",
}

def _pf_label(pf: float | None) -> str:
    """Convert BIDS PartialFourier float to a readable fraction string."""
    if pf is None:
        return "unknown"
    for val, label in _PF_LABELS.items():
        if abs(pf - val) < 0.01:
            return label
    return f"{pf:.3f}"


def _recommend_gibbs_method(
    pf_label: str,
    acquisition_3d: bool,
    result: QCResult,
) -> str:
    """Determine the recommended Gibbs correction method.

    Decision table (mrdegibbs is the only supported method):
      fully sampled, 2D  →  mrdegibbs (default slice-wise)
      fully sampled, 3D  →  mrdegibbs -mode 3d
      PF = 7/8 or 6/8   →  mrdegibbs (SuShi handles these adequately)
      PF = 5/8           →  mrdegibbs + warning (no reliable correction
                             exists for PF=5/8 from magnitude NIfTI data)
      PF unknown         →  mrdegibbs + warning
    """
    if pf_label == "5/8":
        result.warn(
            "Partial Fourier = 5/8 detected. No reliable Gibbs correction "
            "exists for PF=5/8 from reconstructed magnitude data. mrdegibbs "
            "will be applied but may not fully correct ringing. Consider "
            "acquiring at PF=6/8 or 7/8 in future sessions."
        )
    elif pf_label == "unknown":
        result.warn(
            "PartialFourier not found in JSON sidecar — cannot confirm "
            "acquisition type. mrdegibbs will be applied; results should be "
            "visually inspected for residual ringing."
        )

    return "mrdegibbs_3d" if acquisition_3d else "mrdegibbs"


def check_acquisition(
    files: dict,
    result: QCResult,
) -> dict[str, Any]:
    """Parse BIDS JSON sidecar, detect acquisition type, assess correction readiness."""

    acq: dict[str, Any] = {
        "reverse_pe_available":      files.get("rpe") is not None,
        "reverse_pe_file":           str(files["rpe"]) if files.get("rpe") else None,
        "t1w_available":             files.get("t1w") is not None,
        "t1w_file":                  str(files["t1w"]) if files.get("t1w") else None,
        "json_sidecar_available":    files.get("json_sidecar") is not None,
        "phase_encoding_direction":  None,
        "total_readout_time":        None,
        "effective_echo_spacing":    None,
        "multiband_factor":          None,
        "slice_timing_available":    False,
        # Partial Fourier
        "partial_fourier":           None,   # raw float from sidecar (1.0, 0.875, etc.)
        "partial_fourier_fraction":  "unknown",  # "none","7/8","6/8","5/8","unknown"
        # Acquisition geometry
        "acquisition_3d":            False,  # True for 3D Fourier-encoded sequences
        # Derived recommendations
        "topup_ready":               False,
        "synb0_possible":            False,
        "recommended_gibbs_method":  "mrdegibbs",  # set after sidecar parsed
    }

    if not files.get("t1w"):
        result.warn("No T1w found — FreeSurfer recon-all and Synb0-DisCo will be skipped")

    if files.get("json_sidecar"):
        try:
            with open(files["json_sidecar"], encoding="utf-8") as f:
                sidecar = json.load(f)

            # PhaseEncodingDirection: BIDS standard field.
            # Philips dcm2niix may write PhaseEncodingAxis (axis only, no polarity).
            # We accept both; direction without polarity defaults to positive (j = AP).
            pe_dir = sidecar.get("PhaseEncodingDirection") or                      sidecar.get("PhaseEncodingAxis")
            acq["phase_encoding_direction"] = pe_dir
            acq["phase_encoding_axis"]      = pe_dir.replace("-", "") if pe_dir else None

            # TotalReadoutTime: BIDS standard.
            # Philips dcm2niix writes EstimatedTotalReadoutTime instead.
            acq["total_readout_time"] = (
                sidecar.get("TotalReadoutTime") or
                sidecar.get("EstimatedTotalReadoutTime")
            )
            # Also capture echo spacing — BIDS or Philips estimated variant
            acq["effective_echo_spacing"] = (
                sidecar.get("EffectiveEchoSpacing") or
                sidecar.get("EstimatedEffectiveEchoSpacing")
            )
            acq["multiband_factor"]          = sidecar.get("MultibandAccelerationFactor")
            acq["slice_timing_available"]    = "SliceTiming" in sidecar

            # Partial Fourier detection — handles both standard BIDS and
            # Philips-specific field naming.
            #
            # Standard BIDS:  "PartialFourier": 0.875  (float, e.g. 6/8)
            # Philips:        "PartialFourierEnabled": "YES"
            #                 "AcquisitionMatrixPE": N
            #                 "PhaseEncodingStepsNoPartialFourier": M
            #                 "PercentSampling": 100
            #
            # On Philips, PartialFourierEnabled=YES means the option is
            # active in the protocol but may be set to 100% (fully sampled).
            # The acquisition is truly partial Fourier only when
            # AcquisitionMatrixPE < PhaseEncodingStepsNoPartialFourier.
            pf_raw = sidecar.get("PartialFourier")
            if pf_raw is not None:
                # Standard BIDS float field
                acq["partial_fourier"]          = float(pf_raw)
                acq["partial_fourier_fraction"] = _pf_label(float(pf_raw))
            else:
                # Philips-specific detection
                acq_matrix = sidecar.get("AcquisitionMatrixPE")
                pf_steps   = sidecar.get("PhaseEncodingStepsNoPartialFourier")
                pf_enabled = sidecar.get("PartialFourierEnabled", "NO")
                pct_sample = sidecar.get("PercentSampling", 100)

                if (pf_enabled == "YES"
                        and acq_matrix is not None
                        and pf_steps is not None):
                    if int(acq_matrix) >= int(pf_steps) or int(pct_sample) >= 100:
                        # Protocol has PF enabled but set to 100% — fully sampled
                        acq["partial_fourier"]          = 1.0
                        acq["partial_fourier_fraction"] = "none"
                        result.info(
                            "Philips PartialFourierEnabled=YES but "
                            f"AcquisitionMatrixPE={acq_matrix} == "
                            f"PhaseEncodingStepsNoPartialFourier={pf_steps} "
                            "— data is fully sampled"
                        )
                    else:
                        # Genuinely partial Fourier — calculate fraction
                        pf_fraction = int(acq_matrix) / int(pf_steps)
                        acq["partial_fourier"]          = pf_fraction
                        acq["partial_fourier_fraction"] = _pf_label(pf_fraction)
                        result.info(
                            f"Philips partial Fourier detected: "
                            f"{acq_matrix}/{pf_steps} = "
                            f"{pf_fraction:.3f} "
                            f"({acq['partial_fourier_fraction']})"
                        )
                elif pf_enabled == "NO" or pct_sample >= 100:
                    acq["partial_fourier"]          = 1.0
                    acq["partial_fourier_fraction"] = "none"
                else:
                    # No usable PF fields — flag as unknown
                    acq["partial_fourier_fraction"] = "unknown"

            # 3D acquisition detection:
            # BIDS field MRAcquisitionType = "3D" is definitive.
            # Fallback: 3D sequences typically have no SliceTiming and have
            # a non-standard PhaseEncodingDirection pattern (k or i components).
            mr_acq_type = sidecar.get("MRAcquisitionType", "")
            if mr_acq_type == "3D":
                acq["acquisition_3d"] = True
            elif mr_acq_type == "2D":
                acq["acquisition_3d"] = False
            else:
                # Heuristic: 3D EPI typically lacks SliceTiming
                acq["acquisition_3d"] = not acq["slice_timing_available"]
                if acq["acquisition_3d"]:
                    result.info(
                        "MRAcquisitionType not in sidecar — inferred 3D acquisition "
                        "from absent SliceTiming. Verify with scanner protocol."
                    )

            # Check that we resolved PE direction and readout time
            # (either from BIDS standard keys or Philips fallback keys)
            missing = []
            if not acq.get("phase_encoding_direction"):
                missing.append("PhaseEncodingDirection / PhaseEncodingAxis")
            if not acq.get("total_readout_time"):
                missing.append("TotalReadoutTime / EstimatedTotalReadoutTime")
            if missing:
                result.warn(
                    f"JSON sidecar missing fields required for topup: "
                    f"{', '.join(missing)}"
                )

        except Exception as exc:
            result.warn(f"Could not parse JSON sidecar: {exc}")

    # Set recommended Gibbs method based on what we now know
    acq["recommended_gibbs_method"] = _recommend_gibbs_method(
        pf_label      = acq["partial_fourier_fraction"],
        acquisition_3d = acq["acquisition_3d"],
        result        = result,
    )

    result.info(
        f"Partial Fourier: {acq['partial_fourier_fraction']}  |  "
        f"Acquisition: {'3D' if acq['acquisition_3d'] else '2D'}  |  "
        f"Gibbs method: {acq['recommended_gibbs_method']}"
    )

    # topup requires reverse PE b0 AND PhaseEncodingDirection AND TotalReadoutTime
    acq["topup_ready"] = (
        acq["reverse_pe_available"]
        and acq["phase_encoding_direction"] is not None
        and acq["total_readout_time"] is not None
    )

    # Synb0 requires T1w (generates synthetic b0 when no reverse PE available)
    acq["synb0_possible"] = acq["t1w_available"] and not acq["reverse_pe_available"]

    if not acq["topup_ready"] and not acq["synb0_possible"]:
        result.warn(
            "Neither topup nor Synb0-DisCo is available — susceptibility "
            "distortion correction will be skipped"
        )

    return acq


# ---------------------------------------------------------------------------
# Step 7: Signal quality
# ---------------------------------------------------------------------------

def check_signal_quality(
    nifti_info: dict,
    grad_info:  dict,
    result: QCResult,
    run_ndc: bool,
) -> dict[str, Any]:
    """Compute SNR estimate, detect outlier volumes, optionally run NDC."""
    import numpy as np

    data     = nifti_info["_data"]
    b0_mask  = grad_info["_b0_mask"]
    dwi_mask = grad_info["_dwi_mask"]

    quality: dict[str, Any] = {
        "snr_b0_estimate":        None,
        "n_outlier_volumes":      0,
        "outlier_volume_indices": [],
        "gradient_issues":        grad_info["gradient_issues"],
        "ndc_run":                False,
        "ndc_values":             [],
    }

    # ---- SNR estimate from b0 volumes ----
    if b0_mask.any():
        b0_data = data[..., b0_mask].astype(np.float64)

        # Philips float scaling correction:
        # When UsePhilipsFloatNotDisplayScaling=1, dcm2niix stores raw scanner
        # float values. The NIfTI scl_slope/scl_inter may be 0/0 (disabled),
        # leaving raw values that can be very small. We normalise to [0,1]
        # before computing SNR to get a scale-independent estimate.
        b0_max = b0_data.max()
        if b0_max > 0:
            b0_data = b0_data / b0_max

        # Brain mask: above 10% of mean b0 signal
        b0_mean_vol = b0_data.mean(axis=-1)
        brain_thresh = b0_mean_vol.mean() * 0.1
        brain_mask_3d = b0_mean_vol > brain_thresh

        if brain_mask_3d.sum() > 0:
            signal = float(b0_mean_vol[brain_mask_3d].mean())
            # Noise estimate: std across b0 volumes in brain voxels
            # If only one b0, use background (non-brain) std as noise estimate
            if b0_data.shape[-1] > 1:
                noise_std = float(b0_data[brain_mask_3d].std(axis=-1).mean())
            else:
                bg_mask = ~brain_mask_3d
                noise_std = float(b0_mean_vol[bg_mask].std()) if bg_mask.any() else 0.0

            if noise_std > 0:
                quality["snr_b0_estimate"] = round(signal / noise_std, 2)
                result.info(f"SNR (b0):    {quality['snr_b0_estimate']:.1f}")
                if quality["snr_b0_estimate"] < 5:
                    result.warn(
                        f"Very low SNR estimate ({quality['snr_b0_estimate']:.1f}) — "
                        "data may be too noisy for reliable fitting"
                    )

    # ---- Outlier volume detection (signal dropout) ----
    # Flag volumes where mean brain signal drops more than 3 std below mean
    try:
        vol_means = np.array([
            float(data[..., i].mean())
            for i in range(data.shape[-1])
        ])
        dwi_means = vol_means[dwi_mask]
        if len(dwi_means) > 3:
            threshold = dwi_means.mean() - 3 * dwi_means.std()
            dwi_indices = np.where(dwi_mask)[0]
            outlier_idx = dwi_indices[dwi_means < threshold].tolist()
            quality["n_outlier_volumes"]      = len(outlier_idx)
            quality["outlier_volume_indices"] = outlier_idx
            if outlier_idx:
                result.warn(
                    f"{len(outlier_idx)} potential outlier volume(s) detected "
                    f"(signal dropout at indices {outlier_idx}) — "
                    "eddy --repol will handle these during correction"
                )
    except Exception as exc:
        result.warn(f"Outlier detection failed: {exc}")

    # ---- NDC (Neighboring DWI Correlation) — DIPY >= 1.10 ----
    if run_ndc:
        try:
            from dipy.segment.mask import median_otsu
            from dipy.core.gradients import gradient_table
            import importlib.metadata as _im
            dipy_ver = tuple(int(x) for x in _im.version("dipy").split(".")[:2])
            if dipy_ver >= (1, 10):
                from dipy.denoise.noise_estimate import piesno
                # NDC added in 1.10 — import path
                try:
                    from dipy.stats.qc import neighboring_dwi_correlation as ndc_fn
                    bvals = grad_info["_bvals"]
                    bvecs = grad_info["_bvecs"].T   # NDC expects (N,3)
                    gtab = gradient_table(bvals, bvecs)
                    ndc_vals = ndc_fn(data, gtab)
                    quality["ndc_run"]    = True
                    quality["ndc_values"] = [round(float(v), 4) for v in ndc_vals]
                    ndc_mean = float(np.mean(ndc_vals))
                    result.info(f"NDC mean:    {ndc_mean:.4f}")
                    if ndc_mean < 0.4:
                        result.warn(
                            f"Low mean NDC value ({ndc_mean:.4f}) — "
                            "may indicate substantial noise or motion"
                        )
                except ImportError:
                    result.info("NDC metric not available in this DIPY build")
            else:
                result.info("NDC requires DIPY >= 1.10 — skipping")
        except Exception as exc:
            result.warn(f"NDC computation failed: {exc}")

    return quality


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_qc(
    sub:             str,
    source_dir:      Path,
    output_json:     Path,
    b0_threshold:    int  = 50,
    noddi_min:       int  = NODDI_MIN_DIRECTIONS,
    noddi_high:      int  = NODDI_HIGH_DIRECTIONS,
    run_ndc:         bool = False,
) -> bool:
    """Run all QC checks. Returns True if passed, False if critical failures."""

    result = QCResult(sub)
    print(f"[{sub}] ━━ BIDS QC starting ━━")

    # Step 1: File discovery
    files = discover_files(sub, source_dir, result)
    if not result.passed:
        _write_json(result, output_json)
        return False

    # Step 2: NIfTI integrity
    nifti_info = check_nifti(files, result, b0_threshold)
    if not result.passed or nifti_info is None:
        _write_json(result, output_json)
        return False
    result.data.update({k: v for k, v in nifti_info.items() if not k.startswith("_")})
    result.data["dwi_file"]       = str(files["dwi_nii"])
    result.data["bval_file"]      = str(files["bval"])
    result.data["bvec_file"]      = str(files["bvec"])
    result.data["b0_threshold"]   = b0_threshold
    result.data["n_runs"]         = files.get("n_runs", 1)
    result.data["additional_runs"] = [
        {"nii": str(r["nii"]), "bval": str(r["bval"]),
         "bvec": str(r["bvec"]), "stem": r["stem"]}
        for r in files.get("additional_runs", [])
    ]
    result.data["all_nii"]  = [str(p) for p in files.get("all_nii",  [files["dwi_nii"]])]
    result.data["all_bval"] = [str(p) for p in files.get("all_bval", [files["bval"]])]
    result.data["all_bvec"] = [str(p) for p in files.get("all_bvec", [files["bvec"]])]

    # Step 3: Gradient table
    grad_info = check_gradients(files, nifti_info, result, b0_threshold)
    if not result.passed or grad_info is None:
        _write_json(result, output_json)
        return False
    result.data["n_b0"]               = grad_info["n_b0"]
    result.data["n_dwi"]              = grad_info["n_dwi"]
    result.data["gradient_issues"]    = grad_info["gradient_issues"]

    # Step 4: Shell structure
    result.shells = analyse_shells(grad_info, result)
    if not result.passed:
        _write_json(result, output_json)
        return False

    # Step 5: Capabilities
    result.caps = determine_capabilities(
        result.shells, result, noddi_min, noddi_high
    )

    # Step 6: Acquisition metadata
    result.acq = check_acquisition(files, result)

    # Step 7: Signal quality
    result.quality = check_signal_quality(
        nifti_info, grad_info, result, run_ndc
    )

    # Final status
    _write_json(result, output_json)

    print(f"[{sub}] ━━ BIDS QC {'PASSED' if result.passed else 'FAILED'} "
          f"({len(result.warnings)} warning(s), "
          f"{len(result.critical)} critical failure(s)) ━━")

    return result.passed


def _write_json(result: QCResult, path: Path) -> None:
    """Write capability profile JSON, stripping internal numpy arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, default=_json_default)
    print(f"[{result.sub}] Capability profile: {path}")


def _json_default(obj: Any) -> Any:
    """JSON serialiser for numpy types."""
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BIDS QC and capability profiling for dwiforge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("subject_id",   help="Subject ID (e.g. sub-001)")
    p.add_argument("source_dir",   help="BIDS source root directory")
    p.add_argument("output_json",  help="Path to write capability.json")
    p.add_argument("--b0-threshold",        type=int, default=50,
                   help="b-value ≤ threshold treated as b0 (default: 50)")
    p.add_argument("--noddi-min-directions",  type=int, default=NODDI_MIN_DIRECTIONS,
                   help=f"Min directions for NODDI standard confidence (default: {NODDI_MIN_DIRECTIONS})")
    p.add_argument("--noddi-high-directions", type=int, default=NODDI_HIGH_DIRECTIONS,
                   help=f"Min directions for NODDI high confidence (default: {NODDI_HIGH_DIRECTIONS})")
    p.add_argument("--run-ndc", action="store_true",
                   help="Run NDC quality metric (requires DIPY >= 1.10, slower)")
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    passed = run_qc(
        sub          = args.subject_id,
        source_dir   = Path(args.source_dir),
        output_json  = Path(args.output_json),
        b0_threshold = args.b0_threshold,
        noddi_min    = args.noddi_min_directions,
        noddi_high   = args.noddi_high_directions,
        run_ndc      = args.run_ndc,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
