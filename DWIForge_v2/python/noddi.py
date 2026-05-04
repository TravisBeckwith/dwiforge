#!/usr/bin/env python3
"""
python/noddi.py -- AMICO NODDI fitting wrapper for dwiforge

Runs under user site-packages Python (dipy==1.9.0, AMICO 2.x).
Do NOT prepend DWIFORGE_DEPS_DIR to PYTHONPATH when calling this script.

Kernel caching: AMICO manages kernel storage internally.
generate_kernels(regenerate=False) skips generation if kernels already exist.

Usage:
    python3 noddi.py
        --dwi              dwi.nii.gz
        --bval             dwi.bval
        --bvec             dwi.bvec
        --mask             wm_mask.nii.gz
        --output           <subject_noddi_dir>
        --b0_threshold     50
        --nthreads         4
        --noddi_confidence standard|high
        --capability_json  capability.json
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="AMICO NODDI fitting for dwiforge")
    p.add_argument("--dwi",              required=True)
    p.add_argument("--bval",             required=True)
    p.add_argument("--bvec",             required=True)
    p.add_argument("--mask",             required=True)
    p.add_argument("--output",           required=True)
    p.add_argument("--b0_threshold",     type=float, default=50.0)
    p.add_argument("--nthreads",         type=int,   default=4)
    p.add_argument("--noddi_confidence", default="standard",
                   choices=["standard", "high"])
    p.add_argument("--capability_json",  default=None)
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import amico
        import amico.util
    except ImportError as e:
        print(f"ERROR: AMICO import failed: {e}", file=sys.stderr)
        print("Install: pip install dmri-amico dmri-dicelib", file=sys.stderr)
        return 1

    print(f"AMICO version: {amico.__version__}")

    for path, label in [(args.dwi, "DWI"), (args.bval, "bval"),
                        (args.bvec, "bvec"), (args.mask, "mask")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 1

    os.makedirs(args.output, exist_ok=True)

    # -- Step 1: Scheme file ---------------------------------------------------
    # Saved alongside the output so it's traceable per-subject.

    scheme_path = os.path.join(args.output, "acquisition.scheme")
    if not os.path.exists(scheme_path):
        print("Generating AMICO scheme file...")
        amico.util.fsl2scheme(
            bvalsFilename=args.bval,
            bvecsFilename=args.bvec,
            schemeFilename=scheme_path,
            bStep=1.0,
        )
        print(f"  Scheme: {scheme_path}")
    else:
        print(f"  Using cached scheme: {scheme_path}")

    # -- Step 2: Evaluation setup ----------------------------------------------

    out = Path(args.output)
    # AMICO 2.x: output goes to study_path/subject/ automatically.
    # output_path constructor arg was removed in 2.x.
    ae = amico.Evaluation(
        study_path  = str(out.parent),
        subject     = out.name,
    )
    ae.BLAS_nthreads = args.nthreads
    ae.nthreads      = args.nthreads

    # -- Step 3: Load data -----------------------------------------------------

    print("Loading DWI data...")
    # AMICO 2.x: b0_threshold is no longer a load_data() argument.
    # Set it via the config dict before loading data.
    if hasattr(ae, 'CONFIG') and isinstance(ae.CONFIG, dict):
        ae.CONFIG['b0_threshold'] = args.b0_threshold
    elif hasattr(ae, 'set_config'):
        try:
            ae.set_config('b0_threshold', args.b0_threshold)
        except Exception:
            pass  # AMICO 2.x may not support this either — silently skip

    ae.load_data(
        dwi_filename    = args.dwi,
        scheme_filename = scheme_path,
        mask_filename   = args.mask,
    )

    # -- Step 4: Model ---------------------------------------------------------

    ae.set_model("NODDI")
    if args.noddi_confidence == "standard":
        print("NOTE: Standard-confidence NODDI (b=1000 ~32 dirs).")
        print("      Valid estimates, less precise than multi-shell.")

    # -- Step 5: Kernels -------------------------------------------------------
    # generate_kernels(regenerate=False) skips generation if kernels already
    # exist in AMICO's internal location for this evaluation (~3 min first run).
    # load_kernels() takes no arguments — AMICO resolves the path internally.

    print("Generating/loading NODDI kernels...")
    ae.generate_kernels(regenerate=False)
    ae.load_kernels()

    # -- Step 6: Fit -----------------------------------------------------------

    print("Fitting NODDI model...")
    ae.fit()

    # -- Step 7: Save ----------------------------------------------------------

    print("Saving results...")
    # AMICO 2.x: save_results() takes no arguments — path set in constructor
    ae.save_results()

    # -- Step 8: Verify + update capability profile ----------------------------

    # AMICO 2.x writes fit_NDI/ODI/FWF/dir — rename to canonical NODDI_* names
    import shutil as _shutil, glob as _glob
    _amico_out = None
    _out_path = Path(args.output)
    for _candidate in [
        _out_path / "AMICO" / "NODDI",
        _out_path.parent / "AMICO" / "NODDI",
    ]:
        if _candidate.exists():
            _amico_out = _candidate
            break
    if _amico_out is None:
        _hits = _glob.glob(str(_out_path / "**" / "fit_NDI.nii.gz"), recursive=True)
        if not _hits:
            _hits = _glob.glob(str(_out_path.parent / "**" / "fit_NDI.nii.gz"), recursive=True)
        if _hits:
            _amico_out = Path(_hits[0]).parent
    if _amico_out:
        print(f"AMICO output: {_amico_out}")
        for _src_n, _dst_n in [
            ("fit_NDI.nii.gz", "NODDI_icvf.nii.gz"),
            ("fit_ODI.nii.gz", "NODDI_odi.nii.gz"),
            ("fit_FWF.nii.gz", "NODDI_isovf.nii.gz"),
            ("fit_dir.nii.gz", "NODDI_directions.nii.gz"),
        ]:
            _s = _amico_out / _src_n
            _d = _out_path / _dst_n
            if _s.exists() and not _d.exists():
                _shutil.copy2(_s, _d)
                print(f"  {_src_n} -> {_dst_n}")
    else:
        print("WARNING: AMICO output dir not found — output files may be missing")

    expected = {
        "icvf":       "NODDI_icvf.nii.gz",       # NDI: neurite density index
        "odi":        "NODDI_odi.nii.gz",          # ODI: orientation dispersion
        "isovf":      "NODDI_isovf.nii.gz",        # ISOVF: free water fraction
        "directions": "NODDI_directions.nii.gz",   # principal fiber direction
    }
    found, missing = {}, []
    for label, fname in expected.items():
        full = os.path.join(args.output, fname)
        if os.path.exists(full):
            found[label] = full
            print(f"  {label:12s}: {fname} ({os.path.getsize(full)/1e6:.1f} MB)")
        else:
            missing.append(fname)

    if missing:
        print(f"WARNING: Missing outputs: {missing}", file=sys.stderr)

    if args.capability_json and os.path.exists(args.capability_json):
        with open(args.capability_json) as f:
            cap = json.load(f)

        cap["noddi"] = {
            "status":          "complete" if not missing else "partial",
            "model":           "NODDI",
            "amico_version":   amico.__version__,
            "confidence":      args.noddi_confidence,
            "single_shell":    True,
            "b0_threshold":    args.b0_threshold,
            "output_dir":      args.output,
            "metrics":         found,
            "missing_outputs": missing,
        }
        if args.noddi_confidence == "standard":
            cap["noddi"]["warning"] = (
                "Standard-confidence NODDI (single-shell b=1000, ~32 dirs). "
                "Estimates valid but less precise than multi-shell acquisitions."
            )

        with open(args.capability_json, "w") as f:
            json.dump(cap, f, indent=2)
        print("capability.json updated")

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
