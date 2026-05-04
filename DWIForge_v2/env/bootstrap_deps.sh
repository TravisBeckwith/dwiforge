#!/usr/bin/env bash
# dwiforge — bootstrap_deps.sh
# =============================================================================
# One-time setup of dwiforge's isolated Python dependency directory.
#
# WHY THIS EXISTS
# ---------------
# DESIGNER-v2 pins dipy==1.9.0 in user site-packages. dwiforge's own scripts
# (denoise.py, qc_bids.py, etc.) require dipy>=1.12.0 for the Patch2Self v3
# bug fix (PR #3631, fixed in 1.12). These cannot share a Python environment.
#
# SOLUTION: pip install --target
# --------------------------------
# This script installs dwiforge's deps into an isolated directory:
#   ~/.local/share/dwiforge/deps/
#
# env_setup.sh then prepends this directory to PYTHONPATH when running
# dwiforge scripts. Python's import system finds dwiforge's dipy>=1.12 first.
#
# DESIGNER is called as a subprocess ($DESIGNER_BIN) in a clean environment
# without the DWIFORGE_DEPS_DIR prefix, so it continues to import its own
# pinned dipy==1.9.0 from user site-packages. The two never interact.
#
# USAGE
# -----
#   bash env/bootstrap_deps.sh            # install to default location
#   DWIFORGE_DEPS_DIR=/custom/path \
#     bash env/bootstrap_deps.sh          # install to custom location
#   bash env/bootstrap_deps.sh --upgrade  # upgrade existing deps
#
# RE-RUNNING
# ----------
# Safe to run multiple times. Use --upgrade to update packages.
# =============================================================================

set -euo pipefail

DEPS_DIR="${DWIFORGE_DEPS_DIR:-${HOME}/.local/share/dwiforge/deps}"
UPGRADE="${1:-}"
PYTHON="${PYTHON_EXECUTABLE:-python3}"

echo "dwiforge dependency bootstrap"
echo "  Target dir: ${DEPS_DIR}"
echo "  Python:     ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo ""

# Create the target directory
mkdir -p "${DEPS_DIR}"

# Build the pip install command
PIP_CMD=(
    "${PYTHON}" -m pip install
    --target "${DEPS_DIR}"
    --no-deps        # install only what we specify — avoid pulling in
                     # DESIGNER's pinned transitive deps
)

if [[ "$UPGRADE" == "--upgrade" ]]; then
    PIP_CMD+=(--upgrade)
    echo "Mode: upgrade existing packages"
else
    echo "Mode: install (use --upgrade to update)"
fi

echo ""
echo "Installing packages..."
echo ""

# Core packages
"${PIP_CMD[@]}" \
    "dipy>=1.12.0" \
    "nibabel>=5.0" \
    "numpy>=1.24" \
    "scipy>=1.10" \
    "scikit-learn>=1.3" \
    "tqdm>=4.65"

echo ""
echo "Verifying installation..."
echo ""

# Verify key packages and versions
PYTHONPATH="${DEPS_DIR}" "${PYTHON}" - <<'PYEOF'
import sys
sys.path.insert(0, '')  # ensure deps dir is first

import importlib.metadata as meta

checks = [
    ("dipy",         "1.12.0"),
    ("nibabel",      "5.0"),
    ("numpy",        "1.24"),
    ("scipy",        "1.10"),
    ("scikit-learn", "1.3"),
    ("tqdm",         "4.65"),
]

all_ok = True
for pkg, minimum in checks:
    try:
        ver = meta.version(pkg)
        parts_got = tuple(int(x) for x in ver.split(".")[:3] if x.isdigit())
        parts_min = tuple(int(x) for x in minimum.split(".")[:3] if x.isdigit())
        ok = parts_got >= parts_min
        mark = "✓" if ok else "✗"
        print(f"  {mark} {pkg:<18s} {ver:>10s}  (need >= {minimum})")
        if not ok:
            all_ok = False
    except meta.PackageNotFoundError:
        print(f"  ✗ {pkg:<18s} {'NOT FOUND':>10s}  (need >= {minimum})")
        all_ok = False

print()
if all_ok:
    print("All dependencies satisfied.")
else:
    print("Some dependencies missing or out of date.")
    print("Run with --upgrade to update.")
    sys.exit(1)

# Verify P2S v3 is actually available (the critical check)
import dipy
from packaging.version import Version
if Version(dipy.__version__) >= Version("1.12.0"):
    try:
        from dipy.denoise.patch2self import patch2self
        print(f"P2S v3 available in dipy {dipy.__version__}  ✓")
    except ImportError:
        print(f"WARNING: dipy {dipy.__version__} but P2S import failed")
PYEOF

echo ""
echo "Bootstrap complete."
echo ""
echo "The deps directory is: ${DEPS_DIR}"
echo ""
echo "dwiforge will use these packages automatically."
echo "DESIGNER continues to use its own dipy==1.9.0 from user site-packages."
echo ""
echo "To verify at any time:"
echo "  PYTHONPATH='${DEPS_DIR}' python3 -c 'import dipy; print(dipy.__version__)'"
