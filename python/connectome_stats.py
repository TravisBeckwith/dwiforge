#!/usr/bin/env python3
"""
python/connectome_stats.py -- Compositional (CLR) transform for dwiforge connectomes

Structural connectome edge weights (SIFT2-weighted streamline counts) are
compositional data: the meaningful signal lies in the *relative* magnitudes
of connections, while the overall scale (total streamline mass) is an
artifact of scan-specific factors -- streamline budget, SNR, acquisition
length, intracranial volume -- that varies subject to subject and carries
little biological meaning on its own.

Running standard statistics (Pearson/Spearman correlation, PCA, linear
regression, group t-tests) directly on raw connectome matrices ignores the
constant-sum nature of the data and can manufacture spurious negative
correlations purely from the closure constraint. This is the same problem
addressed in microbiome relative-abundance and geochemical percentage data.
References:
  Aitchison (1982) J. R. Stat. Soc. B;
  Pawlowsky-Glahn & Egozcue (2006);
  Pawlowsky-Glahn, Egozcue & Tolosana-Delgado (2015), "Modeling and
  Analysis of Compositional Data";
  for connectomes specifically, see discussions of CoDa treatment of
  streamline-count networks in the network-neuroscience literature.

This script adds a centred log-ratio (CLR) transform of the SIFT2-weighted
streamline-count connectome (connectome_count.csv) as a derived output. It
does NOT replace the raw matrix: raw counts remain the QC/visualisation
reference (Stage 10) and the appropriate input to any rank-based or
network-based-statistic analysis that expects raw weights. The CLR matrix
is intended for group-level linear analyses (correlation, PCA, regression,
mixed models) where compositional validity matters.

--------------------------------------------------------------------------
Transform modes (--mode):

  matrix (default)
      Treat the connectome as ONE composition: the vector of unique
      off-diagonal edges (upper triangle) is CLR-transformed as a single
      closed composition, then written back into a symmetric matrix.
      This preserves symmetry (clr[i,j] == clr[j,i]) and is the standard
      choice when the connectome as a whole is the compositional object.

  row
      Treat EACH node's off-diagonal row as its own composition ("how is
      this node's streamline mass distributed across its targets"). CLR is
      applied per row. The result is NOT symmetric, because each row is
      centred by its own log-mean. Use this only if your downstream model
      consumes node-wise compositions (e.g. per-node distribution
      regression) rather than edge-wise features.

--------------------------------------------------------------------------
Zero handling (multiplicative replacement):

    Streamline counts contain structural zeros (no plausible connection)
    and sampling zeros (a true weak connection the finite streamline
    budget missed); these cannot be distinguished from the count alone.
    CLR requires strictly positive parts, so zeros are replaced before
    transforming using multiplicative replacement (Martin-Fernandez et
    al. 2003, 2015): each zero is set to a small value delta, and the
    non-zero parts are multiplicatively adjusted DOWN so the composition
    still sums to its original total. This preserves the ratios among the
    observed non-zero parts -- the property CLR actually depends on --
    unlike naive additive smoothing, which distorts those ratios.

    delta is expressed as a fraction of the smallest observed non-zero
    part (--delta-frac, default 0.5), a common default that keeps
    replaced zeros safely below the detection limit. This is a defensible
    generic choice, NOT a substitute for judgment about your own data's
    zero structure and sparsity.

--------------------------------------------------------------------------
Usage:
    python3 connectome_stats.py
        --connectome       connectome_count.csv
        --output-dir       <subject_tractography_dir>
        --mode             matrix|row       (default: matrix)
        --delta-frac       0.5
        --capability_json  capability.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="CLR transform of a dwiforge SIFT2-weighted connectome"
    )
    p.add_argument("--connectome", required=True,
                    help="Path to connectome_count.csv (SIFT2-weighted counts)")
    p.add_argument("--output-dir", required=True,
                    help="Directory to write CLR matrix and diagnostics")
    p.add_argument("--mode", choices=["matrix", "row"], default="matrix",
                    help="Compositional unit: whole matrix (symmetric, default) "
                         "or per-node row (asymmetric)")
    p.add_argument("--delta-frac", type=float, default=0.5,
                    help="Zero-replacement value as a fraction of the smallest "
                         "non-zero part (default: 0.5)")
    p.add_argument("--capability_json", default=None,
                    help="Optional capability.json to update in place")
    return p.parse_args()


def detect_delimiter(path: str) -> str:
    """
    Infer the delimiter of a numeric matrix file.

    MRtrix3 tck2connectome writes space-delimited values regardless of the
    .csv extension; other tools may write commas or tabs. Sniff the first
    non-empty, non-comment line rather than assuming.
    """
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "," in s:
                return ","
            if "\t" in s:
                return "\t"
            return None  # None => any run of whitespace (np.loadtxt default)
    return None


def load_connectome(path: str) -> tuple[np.ndarray, str]:
    """Load a square connectome, auto-detecting the delimiter."""
    delim = detect_delimiter(path)
    mat = np.loadtxt(path, delimiter=delim)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected a square connectome matrix, got shape {mat.shape}")
    return mat, (delim if delim is not None else " ")


def _multiplicative_replacement(parts: np.ndarray, delta_frac: float) -> np.ndarray:
    """
    Multiplicative zero replacement for a single composition (1-D array).

    Zeros are set to delta = delta_frac * (smallest non-zero part); the
    non-zero parts are scaled by (1 - sum(delta)/total) so the closed sum
    is preserved and the ratios among non-zero parts are unchanged.
    Returns a strictly-positive array of the same shape. If there are no
    non-zero parts, returns an all-NaN array (caller handles this).
    """
    parts = parts.astype(float)
    nonzero = parts[parts > 0]
    if nonzero.size == 0:
        return np.full_like(parts, np.nan)

    total = parts.sum()
    delta = delta_frac * nonzero.min()
    n_zero = int((parts == 0).sum())

    if n_zero == 0:
        return parts

    replaced = parts.copy()
    replaced[parts == 0] = delta
    # Scale non-zero parts down so the total is preserved.
    scale = (total - n_zero * delta) / total
    if scale <= 0:
        # Degenerate: too many zeros / delta too large relative to mass.
        # Fall back to a tiny fraction that keeps positivity without a
        # negative scale; flagged to the caller via diagnostics upstream.
        delta = total / (n_zero * 2.0 + nonzero.size)
        replaced[parts == 0] = delta
        scale = (total - n_zero * delta) / total
    replaced[parts > 0] = parts[parts > 0] * scale
    return replaced


def _clr(parts: np.ndarray) -> np.ndarray:
    """CLR of a strictly-positive 1-D composition."""
    log_parts = np.log(parts)
    return log_parts - log_parts.mean()


def clr_matrix_mode(mat: np.ndarray, delta_frac: float) -> tuple[np.ndarray, dict]:
    """
    Whole-matrix CLR: the unique upper-triangle edges form one composition.
    Returns a symmetric CLR matrix (zero diagonal) and diagnostics.
    """
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    edges = mat[iu].astype(float)

    n_zero = int((edges == 0).sum())
    total = edges.sum()

    out = np.zeros_like(mat, dtype=float)
    if total <= 0 or (edges > 0).sum() == 0:
        out[:] = np.nan
        np.fill_diagonal(out, 0.0)
        diagnostics = {
            "mode": "matrix", "n_nodes": n, "n_edges": int(edges.size),
            "zero_fraction": float(n_zero / edges.size) if edges.size else np.nan,
            "delta_frac": delta_frac, "degenerate": True,
            "note": "connectome has no positive edges; CLR undefined",
        }
        return out, diagnostics

    replaced = _multiplicative_replacement(edges, delta_frac)
    clr_edges = _clr(replaced)

    out[iu] = clr_edges
    out = out + out.T  # symmetric fill; diagonal stays 0
    diagnostics = {
        "mode": "matrix",
        "n_nodes": n,
        "n_edges": int(edges.size),
        "zero_fraction": float(n_zero / edges.size) if edges.size else np.nan,
        "delta_frac": delta_frac,
        "degenerate": False,
    }
    return out, diagnostics


def clr_row_mode(mat: np.ndarray, delta_frac: float) -> tuple[np.ndarray, dict]:
    """
    Row-wise CLR: each node's off-diagonal row is its own composition.
    Returns an asymmetric CLR matrix (zero diagonal) and diagnostics.
    """
    n = mat.shape[0]
    off = ~np.eye(n, dtype=bool)
    clr = np.zeros_like(mat, dtype=float)
    zero_fractions = np.zeros(n)
    row_sums = np.zeros(n)

    for i in range(n):
        row = mat[i, off[i]].astype(float)
        row_sums[i] = row.sum()
        zero_fractions[i] = (row == 0).sum() / row.size if row.size else np.nan
        if row.sum() <= 0 or (row > 0).sum() == 0:
            clr[i, off[i]] = np.nan
            continue
        replaced = _multiplicative_replacement(row, delta_frac)
        clr[i, off[i]] = _clr(replaced)

    diagnostics = {
        "mode": "row",
        "n_nodes": n,
        "mean_zero_fraction": float(np.nanmean(zero_fractions)),
        "max_zero_fraction": float(np.nanmax(zero_fractions)),
        "delta_frac": delta_frac,
        "n_disconnected_nodes": int(np.sum(row_sums == 0)),
        "disconnected_node_indices": [int(i) for i in np.where(row_sums == 0)[0]],
    }
    return clr, diagnostics


def main():
    args = parse_args()

    conn_path = Path(args.connectome)
    if not conn_path.exists():
        print(f"ERROR: connectome not found: {conn_path}", file=sys.stderr)
        return 1

    if args.delta_frac <= 0 or args.delta_frac >= 1:
        print("ERROR: --delta-frac must be in (0, 1)", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        mat, delim = load_connectome(str(conn_path))
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.mode == "matrix":
        clr, diagnostics = clr_matrix_mode(mat, args.delta_frac)
    else:
        clr, diagnostics = clr_row_mode(mat, args.delta_frac)

    clr_out = out_dir / "connectome_count_clr.csv"
    diag_out = out_dir / "connectome_count_clr_diagnostics.json"

    # Write with the same delimiter the input used, so the CLR matrix is a
    # drop-in companion to connectome_count.csv for downstream tooling.
    write_delim = "," if delim == "," else (" " if delim == " " else delim)
    np.savetxt(clr_out, clr, delimiter=write_delim, fmt="%.6f")
    with open(diag_out, "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"CLR-transformed connectome: {clr_out}")
    print(f"  Mode:                    {diagnostics['mode']}")
    print(f"  Nodes:                   {diagnostics['n_nodes']}")
    if args.mode == "matrix":
        print(f"  Edges (upper triangle):  {diagnostics['n_edges']}")
        print(f"  Zero fraction:           {diagnostics['zero_fraction']:.3f}")
        if diagnostics.get("degenerate"):
            print("  WARNING: connectome has no positive edges; "
                  "CLR matrix is all-NaN.")
    else:
        print(f"  Mean row zero fraction:  {diagnostics['mean_zero_fraction']:.3f}")
        print(f"  Disconnected nodes:      {diagnostics['n_disconnected_nodes']}")
        if diagnostics["n_disconnected_nodes"] > 0:
            print("  WARNING: one or more nodes have zero total streamline "
                  "count; their CLR rows are NaN -- exclude or impute before "
                  "group-level analysis.")

    if args.capability_json:
        cap_path = Path(args.capability_json)
        if cap_path.exists():
            with open(cap_path) as f:
                cap = json.load(f)
        else:
            cap = {}
        cap["connectome_stats"] = {
            "status": "complete",
            "mode": args.mode,
            "method": "CLR with multiplicative zero replacement",
            "delta_frac": args.delta_frac,
            "input": str(conn_path),
            "input_delimiter": "comma" if delim == "," else
                               ("tab" if delim == "\t" else "whitespace"),
            "clr_output": str(clr_out),
            "diagnostics": diagnostics,
        }
        with open(cap_path, "w") as f:
            json.dump(cap, f, indent=2)
        print(f"capability.json updated: {cap_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
