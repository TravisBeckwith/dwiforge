#!/usr/bin/env python3
"""
scripts/parse_config.py — dwiforge configuration resolver

Reads dwiforge.toml, applies CLI overrides, resolves derived defaults,
and prints bash export statements that the orchestrator evaluates:

    eval "$($PYTHON_EXECUTABLE scripts/parse_config.py [options])"

Exit codes:
    0  success — bash exports printed to stdout
    1  configuration error — human-readable message on stderr
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# TOML loading — tomllib (stdlib 3.11+) with fallback to tomli
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> dict:
    try:
        import tomllib                          # Python 3.11+
        with open(path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        pass
    try:
        import tomli                            # pip install tomli
        with open(path, "rb") as f:
            return tomli.load(f)
    except ImportError:
        pass
    # Last resort: minimal parser for simple key=value TOML subsets
    return _fallback_toml_parser(path)


def _fallback_toml_parser(path: Path) -> dict:
    """Minimal TOML parser — handles simple key = value and [sections] only.
    Sufficient for dwiforge.toml; does not handle inline tables or arrays."""
    result: dict[str, Any] = {}
    section: dict[str, Any] = result
    section_name = ""

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section_name = line[1:-1].strip()
                section = {}
                result[section_name] = section
                continue
            if "=" in line:
                key, _, raw_val = line.partition("=")
                key = key.strip()
                raw_val = raw_val.strip()
                # Strip inline comments
                if "#" in raw_val and not raw_val.startswith('"'):
                    raw_val = raw_val[:raw_val.index("#")].strip()
                # Detect type
                if raw_val.startswith('"') or raw_val.startswith("'"):
                    val: Any = raw_val.strip("\"'")
                elif raw_val.lower() == "true":
                    val = True
                elif raw_val.lower() == "false":
                    val = False
                elif raw_val.isdigit():
                    val = int(raw_val)
                else:
                    try:
                        val = float(raw_val)
                    except ValueError:
                        val = raw_val.strip("\"'")
                section[key] = val

    return result


# ---------------------------------------------------------------------------
# Config file discovery
# ---------------------------------------------------------------------------

CONFIG_SEARCH_PATHS = [
    Path.cwd() / "dwiforge.toml",
    Path.home() / ".config" / "dwiforge" / "dwiforge.toml",
]


def find_config(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            _die(f"Config file not found: {explicit}")
        return p
    for candidate in CONFIG_SEARCH_PATHS:
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Defaults and resolution
# ---------------------------------------------------------------------------

# Derived defaults relative to DIR_SOURCE.
# Each value is a tuple of path components joined onto DIR_SOURCE.
_DERIVED_PATH_DEFAULTS = {
    "work":        ("derivatives", "dwiforge", "work"),
    "output":      ("derivatives", "dwiforge", "outputs"),
    "freesurfer":  ("derivatives", "freesurfer"),
    "logs":        ("derivatives", "dwiforge", "logs"),
    "qc":          ("derivatives", "dwiforge", "qc"),
}

_OPTION_DEFAULTS = {
    "cleanup_tier":          0,
    "run_preprocessing":     True,
    "run_registration":      True,
    "run_refinement":        True,
    "run_tensor_fitting":    True,
    "run_noddi":             True,
    "run_connectivity":      True,
    "run_qc_report":         True,
    "ml_method":             "auto",
    "ml_quick_mode":         True,
    "use_gpu":               True,
    "omp_threads":           0,
    "min_free_gb_work":      20,
    "min_free_gb_output":    5,
    "min_free_gb_freesurfer":10,
    "parallel_subjects":     1,
}

_SLURM_DEFAULTS = {
    "partition":         "general",
    "account":           "",
    "mem_gb":            32,
    "cpus_per_task":     8,
    "time_limit":        "12:00:00",
    "max_simultaneous":  0,
    "modules":           "",
    "venv_path":         "",
    "extra_directives":  "",
}

_VALID_ML_METHODS = {"auto", "synthmorph", "ants"}


def _die(msg: str) -> None:
    print(f"dwiforge config error: {msg}", file=sys.stderr)
    sys.exit(1)


def resolve_paths(
    toml_paths: dict,
    env: dict,
    overrides: dict,
) -> dict[str, str]:
    """Resolve all six storage paths with priority: CLI > TOML > ENV > default."""

    def get(key: str) -> str:
        return (
            overrides.get(key)
            or toml_paths.get(key, "").strip()
            or env.get(f"DWIFORGE_DIR_{key.upper()}", "").strip()
            or ""
        )

    source = get("source")
    if not source:
        _die(
            "[paths].source is required.\n"
            "Set it in dwiforge.toml, pass --source /path, or set "
            "DWIFORGE_DIR_SOURCE in your environment."
        )

    source_path = Path(source)
    if not source_path.exists():
        _die(f"[paths].source does not exist: {source}")

    resolved = {"source": str(source_path.resolve())}

    for key, default_parts in _DERIVED_PATH_DEFAULTS.items():
        raw = get(key)
        if raw:
            resolved[key] = str(Path(raw).resolve())
        else:
            resolved[key] = str((source_path / Path(*default_parts)).resolve())

    # Safety: work must not be the same as source (would pollute BIDS tree)
    if resolved["work"] == resolved["source"]:
        _die(
            "[paths].work must not be the same as [paths].source.\n"
            "Set work to a subdirectory or a separate location."
        )

    return resolved


def resolve_options(
    toml_opts: dict,
    env: dict,
    overrides: dict,
) -> dict[str, Any]:
    """Merge options with priority: CLI overrides > TOML > defaults."""
    resolved = dict(_OPTION_DEFAULTS)

    for key in _OPTION_DEFAULTS:
        env_key = f"DWIFORGE_{key.upper()}"
        if key in overrides:
            resolved[key] = overrides[key]
        elif key in toml_opts:
            resolved[key] = toml_opts[key]
        elif env_key in env:
            raw = env[env_key]
            default = _OPTION_DEFAULTS[key]
            if isinstance(default, bool):
                resolved[key] = raw.lower() in ("1", "true", "yes")
            elif isinstance(default, int):
                try:
                    resolved[key] = int(raw)
                except ValueError:
                    _die(f"Environment variable {env_key} must be an integer, got: {raw!r}")
            else:
                resolved[key] = raw

    # Validate ml_method
    ml = resolved["ml_method"]
    if ml not in _VALID_ML_METHODS:
        _die(
            f"[options].ml_method must be one of: "
            f"{', '.join(sorted(_VALID_ML_METHODS))}. Got: {ml!r}"
        )

    # Auto-detect omp_threads
    if resolved["omp_threads"] == 0:
        resolved["omp_threads"] = os.cpu_count() or 1

    return resolved


def resolve_slurm(toml_slurm: dict, env: dict, overrides: dict) -> dict[str, Any]:
    resolved = dict(_SLURM_DEFAULTS)
    for key in _SLURM_DEFAULTS:
        env_key = f"DWIFORGE_SLURM_{key.upper()}"
        if key in overrides:
            resolved[key] = overrides[key]
        elif key in toml_slurm:
            resolved[key] = toml_slurm[key]
        elif env_key in env:
            resolved[key] = env[env_key]
    return resolved


# ---------------------------------------------------------------------------
# Bash export generation
# ---------------------------------------------------------------------------

def _bash_str(val: Any) -> str:
    """Convert a Python value to a safely quoted bash string."""
    if isinstance(val, bool):
        return "true" if val else "false"
    return f"'{str(val)}'"


def emit_exports(
    paths: dict[str, str],
    options: dict[str, Any],
    slurm: dict[str, Any],
    config_path: str,
    runtime: dict | None = None,
) -> None:
    lines = [
        "# --- dwiforge resolved configuration ---",
        f"export DWIFORGE_CONFIG={_bash_str(config_path)}",
        "",
        "# Paths",
    ]
    for key, val in paths.items():
        lines.append(f"export DWIFORGE_DIR_{key.upper()}={_bash_str(val)}")

    lines += ["", "# Options"]
    for key, val in options.items():
        lines.append(f"export DWIFORGE_{key.upper()}={_bash_str(val)}")

    lines += ["", "# SLURM"]
    for key, val in slurm.items():
        lines.append(f"export DWIFORGE_SLURM_{key.upper()}={_bash_str(val)}")

    # Runtime overrides — tool paths and host-specific settings
    if runtime:
        lines += ["", "# Runtime overrides"]
        runtime_map = {
            "designer_bin":         "DWIFORGE_DESIGNER_BIN",
            "designer_python_path": "DWIFORGE_DESIGNER_PYTHON_PATH",
            "deps_dir":             "DWIFORGE_DEPS_DIR",
            "mrtrix3tissue_home":   "MRTRIX3TISSUE_HOME",
            "synb0_home":           "DWIFORGE_SYNB0_HOME",
            "fs_threads":           "DWIFORGE_FS_THREADS",
            "n_streamlines":        "DWIFORGE_N_STREAMLINES",
        }
        for toml_key, env_key in runtime_map.items():
            val = runtime.get(toml_key, "")
            # Skip empty / false / zero values so auto-detection still works
            if val and val != "false" and val != 0:
                lines.append(f"export {env_key}={_bash_str(val)}")

    lines.append("")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Disk space validation
# ---------------------------------------------------------------------------

def check_disk_space(paths: dict[str, str], options: dict[str, Any]) -> None:
    """Warn (don't abort) if any location is below its configured minimum."""
    checks = [
        ("work",       options["min_free_gb_work"]),
        ("output",     options["min_free_gb_output"]),
        ("freesurfer", options["min_free_gb_freesurfer"]),
    ]
    for key, min_gb in checks:
        path = paths[key]
        try:
            stat = os.statvfs(path) if hasattr(os, "statvfs") else None
            if stat is None:
                continue
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
            if free_gb < min_gb:
                print(
                    f"dwiforge warning: {key} location has {free_gb:.1f} GB free "
                    f"(minimum configured: {min_gb} GB): {path}",
                    file=sys.stderr,
                )
        except (OSError, PermissionError):
            pass  # location may not exist yet — orchestrator will create it


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Resolve dwiforge configuration and emit bash exports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", metavar="PATH",
                   help="Explicit path to dwiforge.toml")
    p.add_argument("--check-space", action="store_true",
                   help="Warn if storage locations are below minimum free space")

    # Path overrides
    pg = p.add_argument_group("path overrides")
    for name in ("source", "work", "output", "freesurfer", "logs", "qc"):
        pg.add_argument(f"--{name}", metavar="PATH",
                        help=f"Override [paths].{name}")

    # Key option overrides (most commonly needed on CLI)
    og = p.add_argument_group("option overrides")
    og.add_argument("--ml-method", choices=sorted(_VALID_ML_METHODS))
    og.add_argument("--cleanup-tier", type=int, choices=range(5))
    og.add_argument("--omp-threads", type=int, metavar="N")
    og.add_argument("--parallel-subjects", type=int, metavar="N")
    og.add_argument("--use-gpu", dest="use_gpu",
                    action="store_true", default=None)
    og.add_argument("--no-gpu", dest="use_gpu",
                    action="store_false")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --- Locate config file ---
    config_path = find_config(args.config)
    toml: dict[str, Any] = {}
    if config_path:
        toml = _load_toml(config_path)
    elif args.config is None:
        # No config found anywhere — proceed with CLI/ENV/defaults only
        pass

    toml_paths   = toml.get("paths",   {})
    toml_options = toml.get("options", {})
    toml_slurm   = toml.get("slurm",   {})
    toml_runtime = toml.get("runtime", {})

    # --- Build CLI override dicts ---
    path_overrides: dict[str, str] = {}
    for key in ("source", "work", "output", "freesurfer", "logs", "qc"):
        val = getattr(args, key, None)
        if val is not None:
            path_overrides[key] = val

    option_overrides: dict[str, Any] = {}
    if args.ml_method is not None:
        option_overrides["ml_method"] = args.ml_method
    if args.cleanup_tier is not None:
        option_overrides["cleanup_tier"] = args.cleanup_tier
    if args.omp_threads is not None:
        option_overrides["omp_threads"] = args.omp_threads
    if args.parallel_subjects is not None:
        option_overrides["parallel_subjects"] = args.parallel_subjects
    if args.use_gpu is not None:
        option_overrides["use_gpu"] = args.use_gpu

    # --- Resolve ---
    paths   = resolve_paths(toml_paths, os.environ, path_overrides)
    options = resolve_options(toml_options, os.environ, option_overrides)
    slurm   = resolve_slurm(toml_slurm, os.environ, {})

    # --- Optional disk space check ---
    if args.check_space:
        check_disk_space(paths, options)

    # --- Emit ---
    emit_exports(
        paths, options, slurm,
        config_path=str(config_path) if config_path else "",
        runtime=toml_runtime,
    )


if __name__ == "__main__":
    main()
