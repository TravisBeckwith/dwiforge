# Changelog

All notable changes to dwiforge are documented here. Per-stage design notes
and detailed bug reviews for individual features live alongside this file
(see `CHANGES_stage11.md` for an example); this file tracks release-level
summaries.

## [2.1] — 2026-07-31

### Fixed

- **`env/environment.yml` was invalid YAML.** A `matplotlib>=3.7` entry was
  misplaced as a list item under the `variables:` mapping instead of under
  `dependencies:`. `conda env create -f environment.yml` failed to parse the
  file at all. Moved to the correct location; the file now parses cleanly
  and matplotlib installs as intended.

- **`env/check_env.py` crashed on every run.** This is the script the README
  and `env/DEPENDENCIES.md` tell every new user to run first. It used
  `os.path.expanduser` / `os.path.exists` while checking for the `tmi`
  binary but never imported `os`, so it raised `NameError` partway through
  the tool check on every single invocation. Added the missing import;
  verified the script now runs to completion.

- **Config errors were silently swallowed, surfacing later as a confusing
  crash.** `dwiforge.sh` resolved configuration via
  `eval "$(parse_config.py ...)"`. `parse_config.py` correctly prints a
  clear error and exits 1 on invalid input (e.g. a source path that doesn't
  exist), but wrapping the call in `eval "$(...)"` discarded that exit
  code — `eval`'s own status reflects whether the captured string evaluated
  cleanly, not whether the command that produced it succeeded. The pipeline
  would print the real error, then continue with unset `DWIFORGE_*`
  variables and die several lines later on an unrelated-looking
  `unbound variable` error instead. Config resolution now checks the
  script's actual exit status and stops immediately with the real error
  when it fails.

- **`--ml-quick-mode` / `--ml-full-mode` were silent no-ops.** `dwiforge.sh`
  parsed both flags into `ARG_ML_QUICK` but never forwarded that value to
  `parse_config.py`, and `parse_config.py` had no corresponding CLI option
  to receive it in the first place — both ends of the wiring were missing.
  Neither flag had any effect on pipeline behavior. Added a
  `--ml-quick-mode` / `--ml-full-mode` override pair to `parse_config.py`
  (mirroring the existing `--use-gpu` / `--no-gpu` pattern) and wired
  `dwiforge.sh` to forward the flag through. Verified end-to-end:
  `--ml-full-mode` now correctly resolves `quick mode: false` in
  `--show-config` output.

### Added

- **New optional stage: gradient nonlinearity correction (`stages/12_gnc.sh`).**
  Corrects spatial geometric distortion from gradient coil nonlinearity using
  the HCP Pipelines' `gradient_unwarp.py` and a vendor-supplied gradient
  coefficient file. Most relevant on high-gradient-strength systems, where
  distortion near the edge of the field of view can be substantial.
  - Off by default (`run_gnc = false`) — most sites won't have a
    coefficient file, and those files are vendor-supplied and not
    redistributable.
  - Runs between DESIGNER (stage 05) and tensor fitting (stage 06);
    overwrites `dwi_preprocessed.mif` in place (keeping a
    `dwi_preprocessed_pre_gnc.mif` backup), so no downstream stage needed
    any changes.
  - Degrades gracefully: a missing coefficient file, a missing
    `gradient_unwarp.py` install, or a failed correction all log a clear
    warning and let the pipeline continue uncorrected, rather than failing
    the run over what is an enhancement, not a requirement.
  - **Scope limitation, stated directly in the stage and in
    `capability.json` (`bmatrix_corrected: false`):** this corrects voxel
    *positions* only. It does not recompute a per-voxel b-matrix, which is
    what full gradient-nonlinearity-aware diffusion modeling requires,
    since true gradient strength and direction also vary spatially. Treat
    this as a partial correction, not a complete one, for precision
    microstructure modeling (e.g. NODDI).
  - New config options in `parse_config.py` / `dwiforge.toml`: `run_gnc`,
    `gnc_coeff_file`.
  - Inserting the stage shifted `STAGE_ORDER` indices, which required
    updating the `STAGE_ORDER_PHASE1` / `STAGE_ORDER_PHASE2` array-slice
    split (used to gate the `responsemean` group barrier) from `:0:9`/`:9`
    to `:0:10`/`:10`. Verified stage registration, ordering, and the
    disabled/enabled/missing-dependency paths all behave correctly.

### Known issue (pre-existing, not introduced or fixed in this release)

- `b0_threshold` is documented in `dwiforge.toml` under `[options]` but is
  still absent from `_OPTION_DEFAULTS` in `scripts/parse_config.py` (flagged
  originally in `CHANGES_stage11.md`). It is never exported as
  `DWIFORGE_B0_THRESHOLD`, so setting it in the TOML currently has no
  effect — readers silently fall back to their hardcoded default of 50.
  Still open.

### Note on stage numbering

`stages/12_gnc.sh` runs 7th in execution order (between `designer` and
`tensor-fitting`), not 12th — the filename reflects when the stage was
added, not where it runs. Renumbering the stage files to match execution
order is a possible cosmetic follow-up, not done in this release to keep
the change contained.
