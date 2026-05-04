#!/usr/bin/env python3
"""
env/check_env.py — dwiforge environment checker
================================================
Run this before your first pipeline run to verify all dependencies.
Prints a summary of what is installed, what is missing, and what version
constraints are critical.

Usage:
    python env/check_env.py
    python env/check_env.py --strict   # exit 1 if any REQUIRED package missing
"""
from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
from typing import Any


def _ver(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cmd_ver(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip().split("\n")[0] if result.returncode == 0 else None
    except Exception:
        return None


def _parse_ver(ver_str: str | None) -> tuple[int, ...] | None:
    if ver_str is None:
        return None
    try:
        return tuple(int(x) for x in ver_str.split(".")[:3] if x.isdigit())
    except Exception:
        return None


def _meets(ver_str: str | None, minimum: str) -> bool:
    v = _parse_ver(ver_str)
    m = _parse_ver(minimum)
    if v is None or m is None:
        return False
    return v >= m


# ---------------------------------------------------------------------------
# Check results
# ---------------------------------------------------------------------------

OK   = "OK  "
WARN = "WARN"
FAIL = "FAIL"
INFO = "INFO"

results: list[tuple[str, str, str]] = []   # (status, label, detail)


def check(status: str, label: str, detail: str = "") -> None:
    results.append((status, label, detail))
    symbol = {"OK  ": "\033[32m✓\033[0m",
               "WARN": "\033[33m!\033[0m",
               "FAIL": "\033[31m✗\033[0m",
               "INFO": "\033[34m·\033[0m"}.get(status, " ")
    print(f"  {symbol} [{status}] {label:<40s} {detail}")


# ---------------------------------------------------------------------------
# Python runtime
# ---------------------------------------------------------------------------

print("\nPython runtime")
print("─" * 60)
py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
if sys.version_info >= (3, 11):
    check(OK, "Python", f"{py_ver} (tomllib stdlib available)")
elif sys.version_info >= (3, 10):
    check(WARN, "Python", f"{py_ver} (3.11+ preferred for tomllib stdlib)")
else:
    check(FAIL, "Python", f"{py_ver} (need >= 3.10)")

# ---------------------------------------------------------------------------
# Python packages — confirmed required
# ---------------------------------------------------------------------------

print("\nPython packages — required")
print("─" * 60)

# numpy
v = _ver("numpy")
if _meets(v, "1.24"):
    check(OK, "numpy", v)
else:
    check(FAIL, "numpy", f"{v or 'not installed'} (need >= 1.24)")

# nibabel
v = _ver("nibabel")
if _meets(v, "5.0"):
    check(OK, "nibabel", v)
else:
    check(FAIL, "nibabel", f"{v or 'not installed'} (need >= 5.0)")

# scipy
v = _ver("scipy")
if _meets(v, "1.10"):
    check(OK, "scipy", v)
else:
    check(FAIL, "scipy", f"{v or 'not installed'} (need >= 1.10)")

# scikit-learn
v = _ver("scikit-learn")
if _meets(v, "1.3"):
    check(OK, "scikit-learn", v)
else:
    check(FAIL, "scikit-learn", f"{v or 'not installed'} (need >= 1.3)")

# tqdm
v = _ver("tqdm")
if _meets(v, "4.65"):
    check(OK, "tqdm", v)
else:
    check(FAIL, "tqdm", f"{v or 'not installed'} (need >= 4.65)")

# DIPY — critical version check
print()
v = _ver("dipy")
parsed = _parse_ver(v)
if parsed is None:
    check(FAIL, "dipy", "not installed (need >= 1.12.0)")
elif parsed >= (1, 12, 0):
    check(OK, "dipy", f"{v} — P2S v3 safe")
elif parsed >= (1, 10, 0):
    check(FAIL, "dipy",
          f"{v} — P2S v3 BUG (PR #3631 not fixed until 1.12). "
          "Will auto-fall back to P2S v1 but UPGRADE RECOMMENDED.")
elif parsed >= (1, 3, 0):
    check(WARN, "dipy", f"{v} — P2S v1 only (upgrade to 1.12 for v3)")
else:
    check(FAIL, "dipy", f"{v} — too old (need >= 1.12.0)")

# tomli (only needed on Python < 3.11)
if sys.version_info < (3, 11):
    v = _ver("tomli")
    if _meets(v, "2.0"):
        check(OK, "tomli", f"{v} (Python < 3.11 fallback)")
    else:
        check(FAIL, "tomli",
              f"{v or 'not installed'} (needed for TOML parsing on Python < 3.11)")

# ---------------------------------------------------------------------------
# Python packages — optional / pending
# ---------------------------------------------------------------------------

print("\nPython packages — optional / pending")
print("─" * 60)

# PyTorch
v = _ver("torch")
if v is None:
    check(WARN, "torch (PyTorch)",
          "not installed — full-mode registration falls back to scipy")
else:
    try:
        import torch
        gpu = torch.cuda.is_available()
        n_gpu = torch.cuda.device_count() if gpu else 0
        check(OK, "torch (PyTorch)",
              f"{v} — {'GPU: ' + str(n_gpu) + ' device(s)' if gpu else 'CPU only'}")
    except Exception as e:
        check(WARN, "torch (PyTorch)", f"{v} installed but import failed: {e}")

# AMICO
v = _ver("dmri-amico")
if v is None:
    check(WARN, "dmri-amico",
          "not installed — NODDI stage (05) will be unavailable")
else:
    check(OK, "dmri-amico", f"{v}")

# antspyx
v = _ver("antspyx")
if v is None:
    check(WARN, "antspyx",
          "not installed — ANTs refinement stage (03) will be unavailable")
else:
    check(OK, "antspyx", f"{v}")

# ---------------------------------------------------------------------------
# System tools
# ---------------------------------------------------------------------------

print("\nSystem tools")
print("─" * 60)

# MRtrix3
v = _cmd_ver(["mrconvert", "--version"])
if v:
    check(OK, "MRtrix3 (mrconvert)", v[:60])
else:
    check(FAIL, "MRtrix3 (mrconvert)", "not found in PATH — required")

v = _cmd_ver(["mrdegibbs", "--version"])
check(OK if v else FAIL, "MRtrix3 (mrdegibbs)",
      (v or "not found")[:60])

v = _cmd_ver(["dwidenoise", "--version"])
check(OK if v else WARN, "MRtrix3 (dwidenoise)",
      (v or "not found — MP-PCA denoising unavailable")[:60])

# FSL
fsl_ver = None
if shutil.which("flirt"):
    fsl_ver = _cmd_ver(["flirt", "-version"])
if fsl_ver:
    check(OK, "FSL (flirt)", fsl_ver[:60])
else:
    check(WARN, "FSL", "not found — eddy/topup/BET stages pending")

if shutil.which("eddy"):
    check(OK, "FSL (eddy)", "available")
else:
    check(WARN, "FSL (eddy)", "not found — eddy correction pending")

if shutil.which("fast"):
    check(OK, "FSL (fast)", "available")
else:
    check(WARN, "FSL (fast)", "not found — WM mask for tensor fitting unavailable")

if shutil.which("flirt"):
    check(OK, "FSL (flirt)", "available")
else:
    check(WARN, "FSL (flirt)", "not found — b0→T1w registration unavailable")

# tmi (DESIGNER-v2 fitting companion)
v = _ver("designer2")
tmi_bin = shutil.which("tmi") or (
    os.path.expanduser("~/.local/bin/tmi")
    if os.path.exists(os.path.expanduser("~/.local/bin/tmi")) else None
)
if tmi_bin:
    check(OK, "tmi (DESIGNER fitting)",
          f"found at {tmi_bin}" + (f" (designer2 {v})" if v else ""))
else:
    check(WARN, "tmi (DESIGNER fitting)",
          "not found — tensor fitting stage will fail")

# FreeSurfer
fs_ver = _cmd_ver(["recon-all", "--version"])
if fs_ver:
    check(OK, "FreeSurfer (recon-all)", fs_ver[:60])
else:
    check(WARN, "FreeSurfer", "not found — connectivity stage pending")

sm = shutil.which("mri_synthmorph")
check(OK if sm else WARN, "SynthMorph (mri_synthmorph)",
      "available" if sm else "not found (needs FreeSurfer >= 7.3)")

# ANTs
ants_ver = _cmd_ver(["antsRegistration", "--version"])
if ants_ver:
    check(OK, "ANTs (antsRegistration)", ants_ver[:60])
else:
    check(WARN, "ANTs", "not found — ANTs refinement pending")

n4 = shutil.which("N4BiasFieldCorrection")
check(OK if n4 else WARN, "ANTs (N4BiasFieldCorrection)",
      "available" if n4 else "not found")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("─" * 60)
n_fail = sum(1 for s, _, _ in results if s == FAIL)
n_warn = sum(1 for s, _, _ in results if s == WARN)
n_ok   = sum(1 for s, _, _ in results if s == OK)

print(f"  Results: {n_ok} OK  |  {n_warn} warnings  |  {n_fail} failures")
print()

if n_fail == 0 and n_warn == 0:
    print("  All dependencies satisfied. Pipeline is ready.")
elif n_fail == 0:
    print("  Core dependencies satisfied. Warnings are for pending stages.")
else:
    print("  Fix failures before running the pipeline.")
    print()
    print("  Quick fix:")
    print("    pip install -r env/requirements.txt")
    if sys.version_info < (3, 11):
        print("    pip install tomli")

print()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if any REQUIRED package is missing")
    args = parser.parse_args()
    if args.strict and n_fail > 0:
        sys.exit(1)
