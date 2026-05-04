#!/usr/bin/env python3
"""
python/denoise.py — DWI denoising for dwiforge
===============================================
Method priority (first available wins):
  1. DIPY patch2self version=3  (Patch2Self3, DIPY >= 1.12.0, Mar 2026)
  2. DIPY patch2self version=1  (original Patch2Self, DIPY >= 1.3.0)
  3. Standalone P2S2             (pip install patch2self2, CVPR 2024)
  4. MRtrix3 dwidenoise          (MP-PCA, subprocess)
  5. Passthrough                 (copy input to output, warn)

DIPY version notes:
  >= 1.12  P2S v3 recommended — bug fixed (PR #3631: wrong volumes returned)
  1.10-1.11  P2S v3 present but has confirmed denoised-volume bug; use v1
  >= 1.3   P2S v1 (original, stable)
  < 1.3    No P2S support

Usage:
  python denoise.py \\
      <dwi.nii.gz> <bvals> <output.nii.gz> <subject_id> \\
      [--method auto|p2s3|p2s1|p2s2|mppca|passthrough] \\
      [--model ols|ridge|lasso]   (P2S regression model) \\
      [--b0-threshold 50]         (b-value below which volumes are b0) \\
      [--sketch-size 50000]       (P2S2 coreset size; 0 = auto) \\
      [--b0-denoising]            (also denoise b0 volumes) \\
      [--tmp-dir /path]           (P2S temp dir for memory maps) \\
      [--sidecar /path.json]      (write QC metadata here) \\
      [--nthreads N]              (MRtrix3 MP-PCA thread count)

Exit codes:
  0  success
  1  fatal error (input missing, all methods failed)
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# ---------------------------------------------------------------------------
# DIPY version detection
# ---------------------------------------------------------------------------

def _dipy_version() -> tuple[int, int] | None:
    """Return (major, minor) of installed DIPY, or None if not installed."""
    try:
        ver = importlib.metadata.version("dipy")
        parts = ver.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def _dipy_p2s_max_version() -> int:
    """Return the highest *safe* patch2self version for the installed DIPY.

    DIPY version thresholds:
      >= 1.12   version 3 — Patch2Self3, bug-fixed (PR #3631 fixed wrong
                volumes being returned in denoised output on 1.10/1.11)
      1.10-1.11 version 1 — P2S v3 exists but has confirmed bug; fall back
      >= 1.3    version 1 — original Patch2Self, stable
      < 1.3     0          — no P2S support
    """
    v = _dipy_version()
    if v is None:
        return 0
    major, minor = v
    # P2S v3 is only safe from 1.12 onwards (bug fix in PR #3631)
    if major > 1 or (major == 1 and minor >= 12):
        return 3
    # P2S v3 present in 1.10/1.11 but has a denoised-volume bug — use v1
    if major == 1 and minor >= 10:
        return 1  # intentionally capped; v3 exists but is unsafe
    if major == 1 and minor >= 3:
        return 1
    return 0


def _dipy_p2s_v3_available_but_buggy() -> bool:
    """True if DIPY has P2S v3 but the denoised-volume bug is present (1.10/1.11)."""
    v = _dipy_version()
    if v is None:
        return False
    major, minor = v
    return major == 1 and 10 <= minor <= 11


# ---------------------------------------------------------------------------
# Denoising implementations
# ---------------------------------------------------------------------------

def _denoise_dipy(
    data: Any,
    bvals: Any,
    p2s_version: int,
    model: str,
    b0_threshold: int,
    b0_denoising: bool,
    tmp_dir: str | None,
) -> tuple[Any, dict]:
    """Run DIPY patch2self. Returns (denoised_array, metadata_dict)."""
    from dipy.denoise.patch2self import patch2self

    kwargs: dict[str, Any] = dict(
        model=model,
        b0_threshold=b0_threshold,
        b0_denoising=b0_denoising,
        shift_intensity=True,
        clip_negative_vals=False,
        verbose=True,
    )

    # version param added in DIPY 1.10
    if p2s_version >= 3:
        kwargs["version"] = 3
    else:
        kwargs["version"] = 1

    if tmp_dir is not None:
        kwargs["tmp_dir"] = tmp_dir

    t0 = time.perf_counter()
    denoised = patch2self(data, bvals, **kwargs)
    elapsed = time.perf_counter() - t0

    dipy_ver = importlib.metadata.version("dipy")
    label = f"Patch2Self{p2s_version}"
    meta = {
        "method":        label,
        "dipy_version":  dipy_ver,
        "p2s_version":   p2s_version,
        "model":         model,
        "b0_threshold":  b0_threshold,
        "b0_denoising":  b0_denoising,
        "elapsed_s":     round(elapsed, 1),
    }
    return denoised, meta


def _denoise_standalone_p2s2(
    data: Any,
    bvals: Any,
    sketch_size: int,
) -> tuple[Any, dict]:
    """Run standalone P2S2 (pip install patch2self2)."""
    from models.patch2self2 import patch2self as p2s2_fn  # type: ignore

    effective_sketch = sketch_size if sketch_size > 0 else 50_000

    t0 = time.perf_counter()
    denoised = p2s2_fn(
        data,
        bvals,
        sketching_method="leverage_scores",
        sketch_size=effective_sketch,
    )
    elapsed = time.perf_counter() - t0

    meta = {
        "method":       "Patch2Self2-standalone",
        "sketch_size":  effective_sketch,
        "elapsed_s":    round(elapsed, 1),
    }
    return denoised, meta


def _denoise_mppca(
    input_nii: str,
    output_nii: str,
    sub: str,
    nthreads: int,
) -> dict:
    """Run MRtrix3 dwidenoise via subprocess. Writes output_nii directly."""
    # dwidenoise works on .mif or .nii.gz; we pass NIfTI directly
    cmd = [
        "dwidenoise",
        input_nii,
        output_nii,
        "-nthreads", str(nthreads),
        "-quiet",
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"dwidenoise failed (exit {result.returncode}):\n{result.stderr}"
        )

    meta = {
        "method":    "MP-PCA (dwidenoise)",
        "elapsed_s": round(elapsed, 1),
    }
    return meta


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def _compute_qc_metrics(
    original: Any,
    denoised: Any,
    bvals: Any,
    b0_threshold: int,
) -> dict:
    """Compute SNR improvement estimate and noise level for QC sidecar."""
    import numpy as np

    original = original.astype(np.float64)
    denoised = denoised.astype(np.float64)
    residuals = original - denoised

    # Brain mask: voxels above 10% of mean signal in b0 volume
    b0_mask = bvals <= b0_threshold
    if b0_mask.any():
        b0_mean = original[..., b0_mask].mean(axis=-1)
    else:
        b0_mean = original.mean(axis=-1)

    threshold = b0_mean.mean() * 0.1
    brain_mask = b0_mean > threshold

    n_voxels = int(brain_mask.sum())
    if n_voxels == 0:
        return {"qc_warning": "brain mask empty — metrics unreliable"}

    # Noise std estimated from residuals within mask
    residuals_masked = residuals[brain_mask]           # (n_vox, n_vol)
    noise_std = float(np.std(residuals_masked))

    # Approximate SNR: mean b0 signal / noise std (within mask)
    b0_signal = float(b0_mean[brain_mask].mean()) if b0_mask.any() else float("nan")
    snr = b0_signal / noise_std if noise_std > 0 else float("nan")

    # Percent signal change from denoising (mean absolute)
    pct_change = float(
        np.mean(np.abs(residuals[brain_mask]) / (np.abs(original[brain_mask]) + 1e-6))
        * 100
    )

    return {
        "brain_voxels":      n_voxels,
        "noise_std":         round(noise_std, 4),
        "snr_estimate":      round(snr, 2) if not (snr != snr) else None,
        "mean_pct_change":   round(pct_change, 3),
    }


# ---------------------------------------------------------------------------
# Main denoising dispatcher
# ---------------------------------------------------------------------------

def run_denoising(
    dwi_file:     str,
    bvals_file:   str,
    output_file:  str,
    sub:          str,
    method:       str = "auto",
    model:        str = "ols",
    b0_threshold: int = 50,
    sketch_size:  int = 0,
    b0_denoising: bool = True,
    tmp_dir:      str | None = None,
    sidecar_path: str | None = None,
    nthreads:     int = 1,
) -> bool:
    import numpy as np
    import nibabel as nib

    print(f"[{sub}] Denoising: {Path(dwi_file).name} → {Path(output_file).name}")

    # --- Load data ---
    if not Path(dwi_file).exists():
        print(f"[{sub}] ERROR: input not found: {dwi_file}", file=sys.stderr)
        return False

    img = nib.load(dwi_file)
    data = img.get_fdata(dtype=np.float32)
    bvals = np.loadtxt(bvals_file)

    if data.ndim != 4:
        print(f"[{sub}] ERROR: expected 4D DWI, got {data.ndim}D", file=sys.stderr)
        return False

    n_vols = data.shape[-1]
    n_b0   = int((bvals <= b0_threshold).sum())
    print(f"[{sub}]   Shape: {data.shape}  b0 volumes: {n_b0}/{n_vols}")

    if n_vols < 7:
        print(
            f"[{sub}] WARNING: only {n_vols} volumes — "
            "P2S requires at least 7; falling back to MP-PCA",
        )
        method = "mppca"

    # --- Method resolution ---
    p2s_max = _dipy_p2s_max_version()
    resolved_method = _resolve_method(method, p2s_max)
    print(f"[{sub}]   Method: {resolved_method}  "
          f"(requested: {method}, DIPY P2S max: {p2s_max})")

    # --- Run denoising ---
    denoised:  np.ndarray | None = None
    meta: dict[str, Any] = {"subject": sub, "input": dwi_file}

    try:
        if resolved_method in ("p2s3", "p2s1"):
            p2s_ver = 3 if resolved_method == "p2s3" else 1
            denoised, run_meta = _denoise_dipy(
                data, bvals, p2s_ver, model,
                b0_threshold, b0_denoising, tmp_dir,
            )
            meta.update(run_meta)

        elif resolved_method == "p2s2_standalone":
            denoised, run_meta = _denoise_standalone_p2s2(data, bvals, sketch_size)
            meta.update(run_meta)

        elif resolved_method == "mppca":
            # MP-PCA writes directly to output; skip the numpy save below
            _ensure_parent(output_file)
            run_meta = _denoise_mppca(dwi_file, output_file, sub, nthreads)
            meta.update(run_meta)
            print(f"[{sub}] Denoising complete: {run_meta['method']} "
                  f"({run_meta['elapsed_s']}s)")
            if sidecar_path:
                _write_sidecar(sidecar_path, meta)
            return True

        elif resolved_method == "passthrough":
            import shutil
            _ensure_parent(output_file)
            shutil.copy2(dwi_file, output_file)
            meta["method"] = "passthrough"
            print(f"[{sub}] WARNING: no denoising method available — input copied as-is")
            if sidecar_path:
                _write_sidecar(sidecar_path, meta)
            return True  # not a fatal error; pipeline can continue

        else:
            print(f"[{sub}] ERROR: unknown method: {resolved_method}", file=sys.stderr)
            return False

    except Exception as exc:
        print(f"[{sub}] ERROR during denoising ({resolved_method}): {exc}",
              file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

    if denoised is None:
        print(f"[{sub}] ERROR: denoising returned None", file=sys.stderr)
        return False

    # --- QC metrics ---
    try:
        qc = _compute_qc_metrics(data, denoised, bvals, b0_threshold)
        meta.update(qc)
        print(f"[{sub}]   SNR estimate:    {qc.get('snr_estimate', 'N/A')}")
        print(f"[{sub}]   Noise std:       {qc.get('noise_std', 'N/A')}")
        print(f"[{sub}]   Mean % change:   {qc.get('mean_pct_change', 'N/A')}")
    except Exception as exc:
        print(f"[{sub}] WARNING: QC metric computation failed: {exc}")

    # --- Save output ---
    _ensure_parent(output_file)
    nib.save(
        nib.Nifti1Image(denoised, img.affine, img.header),
        output_file,
    )

    elapsed = meta.get("elapsed_s", "?")
    print(f"[{sub}] Denoising complete: {meta.get('method','?')} ({elapsed}s)")
    print(f"[{sub}] Output: {output_file}")

    if sidecar_path:
        _write_sidecar(sidecar_path, meta)

    return True


def _resolve_method(requested: str, p2s_max: int) -> str:
    """Map requested method name to a concrete implementation ID."""
    if requested == "auto":
        if p2s_max >= 3:
            return "p2s3"
        if _dipy_p2s_v3_available_but_buggy():
            # P2S v3 technically present but has denoised-volume bug in 1.10/1.11
            print(
                "INFO: DIPY 1.10/1.11 detected. Patch2Self v3 has a confirmed "
                "bug on this version (PR #3631). Using P2S v1. "
                "Upgrade to DIPY >= 1.12 for the fixed v3.",
                file=sys.stderr,
            )
        if p2s_max >= 1:
            return "p2s1"
        if _standalone_p2s2_available():
            return "p2s2_standalone"
        if _mppca_available():
            return "mppca"
        return "passthrough"

    if requested == "p2s3":
        if p2s_max >= 3:
            return "p2s3"
        if _dipy_p2s_v3_available_but_buggy():
            print(
                "WARNING: DIPY 1.10/1.11 has Patch2Self v3 but it contains a "
                "confirmed bug (PR #3631 — wrong volumes in denoised output). "
                "Falling back to P2S v1. Upgrade to DIPY >= 1.12 to use v3.",
                file=sys.stderr,
            )
        else:
            print("WARNING: DIPY >= 1.12 required for safe P2S v3; "
                  "falling back to available version", file=sys.stderr)
        return _resolve_method("auto", p2s_max)

    if requested in ("p2s2", "patch2self2"):
        if _standalone_p2s2_available():
            return "p2s2_standalone"
        # P2S2 matrix sketching is incorporated into DIPY P2S v3
        if p2s_max >= 3:
            print("INFO: standalone P2S2 not found; using DIPY P2S v3 "
                  "(incorporates P2S2 matrix sketching)", file=sys.stderr)
            return "p2s3"
        return _resolve_method("auto", p2s_max)

    if requested in ("p2s1", "patch2self"):
        if p2s_max >= 1:
            return "p2s1"
        return _resolve_method("auto", p2s_max)

    if requested in ("mppca", "dwidenoise"):
        if _mppca_available():
            return "mppca"
        print("WARNING: dwidenoise not found; falling back to auto", file=sys.stderr)
        return _resolve_method("auto", p2s_max)

    if requested == "passthrough":
        return "passthrough"

    print(f"WARNING: unknown method {requested!r}; using auto", file=sys.stderr)
    return _resolve_method("auto", p2s_max)


def _standalone_p2s2_available() -> bool:
    try:
        from models import patch2self2  # noqa: F401
        return True
    except ImportError:
        return False


def _mppca_available() -> bool:
    result = subprocess.run(
        ["which", "dwidenoise"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _write_sidecar(path: str, meta: dict) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DWI denoising for dwiforge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("dwi_file",   help="Input DWI NIfTI (.nii.gz)")
    p.add_argument("bvals_file", help="FSL bvals text file")
    p.add_argument("output",     help="Output denoised NIfTI (.nii.gz)")
    p.add_argument("subject_id", help="Subject ID for logging")

    p.add_argument(
        "--method",
        default="auto",
        choices=["auto", "p2s3", "p2s2", "p2s1", "mppca", "passthrough"],
        help="Denoising method (default: auto — uses best available)",
    )
    p.add_argument(
        "--model",
        default="ols",
        choices=["ols", "ridge", "lasso"],
        help="Regression model for Patch2Self (default: ols)",
    )
    p.add_argument(
        "--b0-threshold",
        type=int, default=50, metavar="N",
        help="b-value below which volumes are treated as b0 (default: 50)",
    )
    p.add_argument(
        "--sketch-size",
        type=int, default=0, metavar="N",
        help="Coreset size for standalone P2S2 (default: 0 = auto 50000)",
    )
    p.add_argument(
        "--no-b0-denoising",
        action="store_true",
        help="Skip denoising of b0 volumes",
    )
    p.add_argument(
        "--tmp-dir",
        default=None, metavar="PATH",
        help="Directory for Patch2Self memory-mapped temporary files",
    )
    p.add_argument(
        "--sidecar",
        default=None, metavar="PATH",
        help="Write JSON QC sidecar to this path",
    )
    p.add_argument(
        "--nthreads",
        type=int, default=1, metavar="N",
        help="Threads for MRtrix3 dwidenoise (default: 1)",
    )
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    success = run_denoising(
        dwi_file      = args.dwi_file,
        bvals_file    = args.bvals_file,
        output_file   = args.output,
        sub           = args.subject_id,
        method        = args.method,
        model         = args.model,
        b0_threshold  = args.b0_threshold,
        sketch_size   = args.sketch_size,
        b0_denoising  = not args.no_b0_denoising,
        tmp_dir       = args.tmp_dir,
        sidecar_path  = args.sidecar,
        nthreads      = args.nthreads,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
