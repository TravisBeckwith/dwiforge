#!/usr/bin/env python3
"""
scripts/write_resolved_config.py — snapshot the orchestrator's resolved
configuration as a standalone dwiforge.toml.

Called by dwiforge.sh's `_submit_slurm()` when the pipeline was invoked
without an explicit --config (i.e. the config was auto-discovered or built
entirely from CLI flags / environment). SLURM array workers each start a
fresh `dwiforge.sh` process on a compute node, which may have a different
CWD (and therefore a different auto-discovery result) than the submitting
shell. Writing the fully-resolved configuration to a fixed path and passing
it explicitly via --config guarantees every worker sees identical paths and
options.

This script does NOT re-resolve anything itself — parse_config.py already
did that, and dwiforge.sh's main() has exported the results as DWIFORGE_*
environment variables by the time this script runs. This script simply
reads those exported variables back out and re-serializes them as TOML,
using the same key tables as parse_config.py so the two stay in sync.

Usage:
    python3 scripts/write_resolved_config.py > resolved.toml

Exit codes:
    0  success — TOML printed to stdout
    1  required environment variables were not exported (dwiforge.sh bug —
       this script should only ever be invoked after _load_config)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Reuse the canonical key tables from parse_config.py so this script can
# never silently drift out of sync with what the orchestrator resolves.
sys.path.insert(0, str(Path(__file__).parent))
from parse_config import (  # noqa: E402
    _DERIVED_PATH_DEFAULTS,
    _OPTION_DEFAULTS,
    _SLURM_DEFAULTS,
)

_PATH_KEYS = ["source", *_DERIVED_PATH_DEFAULTS.keys()]

# Runtime overrides — mirrors the runtime_map in parse_config.emit_exports.
_RUNTIME_ENV_MAP = {
    "designer_bin":         "DWIFORGE_DESIGNER_BIN",
    "designer_python_path": "DWIFORGE_DESIGNER_PYTHON_PATH",
    "deps_dir":              "DWIFORGE_DEPS_DIR",
    "mrtrix3tissue_home":    "MRTRIX3TISSUE_HOME",
    "synb0_home":            "DWIFORGE_SYNB0_HOME",
    "fs_threads":            "DWIFORGE_FS_THREADS",
    "n_streamlines":         "DWIFORGE_N_STREAMLINES",
}


def _die(msg: str) -> None:
    print(f"write_resolved_config error: {msg}", file=sys.stderr)
    sys.exit(1)


def _toml_value(raw: str) -> str:
    """Render an already-resolved string env value as a TOML scalar."""
    if raw in ("true", "false"):
        return raw
    try:
        int(raw)
        return raw
    except ValueError:
        pass
    try:
        float(raw)
        return raw
    except ValueError:
        pass
    # String — escape backslashes and quotes for a TOML basic string.
    escaped = raw.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def main() -> None:
    env = os.environ

    lines = [
        "# --- dwiforge resolved configuration (auto-generated) ---",
        "# Written by write_resolved_config.py so all SLURM array workers",
        "# for this run use identical, already-resolved paths and options,",
        "# regardless of each worker's working directory.",
        "",
        "[paths]",
    ]

    missing = []
    for key in _PATH_KEYS:
        env_key = f"DWIFORGE_DIR_{key.upper()}"
        if env_key not in env:
            missing.append(env_key)
            continue
        lines.append(f"{key} = {_toml_value(env[env_key])}")

    lines += ["", "[options]"]
    for key in _OPTION_DEFAULTS:
        env_key = f"DWIFORGE_{key.upper()}"
        if env_key in env:
            lines.append(f"{key} = {_toml_value(env[env_key])}")

    lines += ["", "[slurm]"]
    for key in _SLURM_DEFAULTS:
        env_key = f"DWIFORGE_SLURM_{key.upper()}"
        if env_key in env:
            lines.append(f"{key} = {_toml_value(env[env_key])}")

    runtime_lines = []
    for toml_key, env_key in _RUNTIME_ENV_MAP.items():
        if env_key in env and env[env_key]:
            runtime_lines.append(f"{toml_key} = {_toml_value(env[env_key])}")
    if runtime_lines:
        lines += ["", "[runtime]", *runtime_lines]

    if missing:
        _die(
            "required configuration was not found in the environment: "
            + ", ".join(missing)
            + ". This script must be run after dwiforge.sh's _load_config "
              "has exported DWIFORGE_* variables — it is not meant to be "
              "invoked standalone."
        )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
