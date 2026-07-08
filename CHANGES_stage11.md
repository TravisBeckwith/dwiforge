# Stage 11 (Connectome Stats) — Change Log & Bug Review

This note documents the addition of Stage 11 (compositional / CLR transform
of the structural connectome) and the bugs found and fixed during review.

## What Stage 11 does

Adds a centred-log-ratio (CLR) transform of the SIFT2-weighted streamline-count
connectome (`connectome_count.csv`) as a derived, group-analysis-ready output.
Structural connectome edge weights are compositional data (relative magnitudes
carry the signal; total streamline mass is a scan-specific nuisance). Running
correlation / PCA / regression / group tests directly on raw counts ignores the
constant-sum constraint and can manufacture spurious correlations. The CLR
matrix is written alongside — never replacing — the raw matrix.

New files:
- `python/connectome_stats.py` — the transform
- `stages/11_connectome_stats.sh` — the stage wrapper

Wiring:
- `dwiforge.sh` — `STAGE_SCRIPTS`, `STAGE_ORDER`, `STAGE_CONFIG_VAR`, help text, `--show-config`
- `scripts/parse_config.py` — `run_connectome_stats`, `connectome_stats_mode`, `connectome_stats_delta_frac` added to `_OPTION_DEFAULTS`
- `dwiforge.toml` — the three options documented under `[options]`
- `README.md` — Stage 11 row

## Bugs found and fixed during review

### 1. (Python) CSV delimiter assumption — would crash on real MRtrix output — FIXED
The first draft used `np.loadtxt(delimiter=",")`. MRtrix3 `tck2connectome`
writes **whitespace-delimited** values regardless of the `.csv` extension, so
the loader would raise `ValueError` on actual pipeline output. Replaced with a
`detect_delimiter()` sniffer (comma / tab / whitespace) and the input delimiter
is now preserved on write, so the CLR matrix is a drop-in companion to
`connectome_count.csv`. Verified against both comma- and space-delimited inputs.

### 2. (Python) Zero replacement was mislabelled and statistically wrong — FIXED
The first draft added a pseudocount to **every** part and then rescaled — this
is additive (Laplace) smoothing, not the "Bayesian-multiplicative replacement"
the comment claimed, and it distorts the ratios among observed non-zero edges
(the very quantities CLR depends on). Worse, the rescale step was dead code:
CLR is scale-invariant, so multiplying the whole row by a constant has no effect
on the output. Replaced with genuine **multiplicative replacement**
(Martin-Fernandez et al. 2003/2015): only zeros are set to a small delta, and
the non-zero parts are scaled down to preserve the total, leaving their ratios
untouched. Verified: non-zero ratios and the composition total are both
preserved (Test 8).

Consequence: the config option changed from `connectome_stats_pseudocount`
(additive count) to `connectome_stats_delta_frac` (fraction of the smallest
non-zero edge, in (0,1)).

### 3. (Python) No symmetric option; asymmetric output presented as "correct" — FIXED
The first draft only did row-wise CLR, which produces an asymmetric matrix from
a symmetric input (each row centred by its own log-mean). That is defensible for
node-wise compositions but is a poor default: most connectome group analyses
treat the network as a single compositional object and expect symmetry. Added a
`--mode` switch:
- `matrix` (new default) — the unique upper-triangle edges form one composition;
  output is symmetric.
- `row` — the original per-node behaviour, retained for when it is actually wanted.
Verified: matrix mode output is symmetric with a zero diagonal (Test 1).

### 4. (Shell) Double-logging and masked exit code — FIXED
The stage piped the Python helper through `... 2>&1 | while read line; do _log ...`.
Two problems: (a) the orchestrator's `_run_stage` already routes stage stdout
through the per-stage logger, so every line would be prefixed twice; (b) the
pipe made the `while` loop's exit status the command's exit status, so a Python
failure would be silently swallowed. Rewritten to call Python directly and check
its real exit code — matching the pattern already used by `stages/07_noddi.sh`.

## Pre-existing bug found (NOT introduced here, NOT fixed — flagged for your awareness)

While tracing how options are exported, I found that **`b0_threshold` is set in
`dwiforge.toml` under `[options]` but is absent from `_OPTION_DEFAULTS` in
`scripts/parse_config.py`.** Only keys present in `_OPTION_DEFAULTS` are exported
as `DWIFORGE_*` env vars, so `DWIFORGE_B0_THRESHOLD` is never actually set from
the config — `stages/07_noddi.sh` (and any other reader) silently falls back to
its hard-coded `:-50` default. If you ever set `b0_threshold` to something other
than 50 in the TOML expecting it to take effect, it currently does not.

Fix (if you want it): add `"b0_threshold": 50` to `_OPTION_DEFAULTS`. I left it
untouched because it is outside the scope of this stage and changing it would
alter behaviour for subjects whose data assumed the current (default-50) path.

The same pattern may affect other `[options]`/`[section]` keys that aren't in
`_OPTION_DEFAULTS` (e.g. the `[tensor_fitting]` and `[designer]` sub-tables are
read through separate mechanisms — worth an audit if any of those options seem
to be ignored).

## Tests run

1. Symmetric input → matrix mode output symmetric, zero diagonal — PASS
2. Space-delimited (MRtrix-style) input auto-detected, no crash — PASS
3. Row mode runs, produces asymmetric output — PASS
4. CLR scale-invariance (output identical for input × 1000) — PASS
5. Disconnected node (row mode) → NaN row + warning, exit 0 — PASS
6. All-zero matrix (degenerate) → all-NaN + warning, no crash, exit 0 — PASS
7. Invalid `--delta-frac` (≥1) rejected with exit 1 — PASS
8. Multiplicative replacement preserves non-zero ratios and total — PASS
9. `bash -n` on `dwiforge.sh`, `stages/11_connectome_stats.sh` — PASS
10. `py_compile` on `connectome_stats.py`, `parse_config.py` — PASS
11. Config resolution: TOML values honoured, defaults applied — PASS
