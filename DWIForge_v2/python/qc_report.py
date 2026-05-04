#!/usr/bin/env python3
"""
python/qc_report.py — dwiforge per-subject QC report generator
===============================================================
Generates a PDF QC report from the subject's capability.json and
pipeline output files. Called by stage 10 (10_qc_report.sh).

Sections:
  1. Header       — subject ID, date, pipeline version, acquisition summary
  2. Stage status — pass / skip / fail / warn for each stage
  3. Warnings     — all flagged issues collected from capability profile
  4. Image slices — axial slices of key volumes (b0, FA, MD, NODDI maps)
  5. DTI metrics  — mean ± SD in WM mask (FA, MD, AD, RD)
  6. NODDI metrics— mean ± SD in WM mask (NDI, ODI, ISOVF)
  7. Connectome   — adjacency matrix heatmap + summary statistics

Requires: matplotlib, numpy (both in DWIFORGE_DEPS_DIR)
Optional: nibabel (for image slice extraction — graceful skip if absent)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

# nibabel optional — for extracting image slices
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

COLORS = {
    'complete': '#2ecc71',
    'skipped':  '#95a5a6',
    'failed':   '#e74c3c',
    'warn':     '#f39c12',
    'unknown':  '#bdc3c7',
    'bg':       '#f8f9fa',
    'header':   '#2c3e50',
    'accent':   '#3498db',
    'text':     '#2c3e50',
    'light':    '#ecf0f1',
}

STAGE_LABELS = {
    'qc-bids':            '00 QC BIDS',
    'recon-all':          '01 recon-all',
    'preprocessing':      '02 Preprocessing',
    't1w-prep':           '03 T1w Prep',
    'epi-correction':     '04 EPI Correction',
    'designer':           '05 DESIGNER',
    'tensor-fitting':     '06 Tensor Fitting',
    'noddi':              '07 NODDI',
    'response-functions': '08 Response Functions',
    'tractography':       '09 Tractography',
    'qc-report':          '10 QC Report',
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='dwiforge QC report generator')
    p.add_argument('--subject',          required=True)
    p.add_argument('--capability_json',  required=True)
    p.add_argument('--work_dir',         required=True)
    p.add_argument('--output',           required=True)
    p.add_argument('--pipeline_version', default='2.0')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper: load capability profile
# ---------------------------------------------------------------------------

def load_cap(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def cap_get(cap: dict, *keys, default=None):
    """Safe nested dict access."""
    d = cap
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d


# ---------------------------------------------------------------------------
# Helper: extract axial slices from NIfTI
# ---------------------------------------------------------------------------

def extract_slices(nii_path: str, n_slices: int = 5) -> list | None:
    """Return n evenly-spaced axial slices as numpy arrays, or None."""
    if not NIBABEL_AVAILABLE:
        return None
    # Accept .nii, .nii.gz, and .mif (skip .mif — needs mrconvert)
    if not os.path.exists(nii_path):
        return None
    if nii_path.endswith('.mif'):
        return None
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        # Handle 4D — take first volume
        if data.ndim == 4:
            data = data[..., 0]
        nz = data.shape[2]
        indices = np.linspace(int(nz * 0.2), int(nz * 0.8), n_slices, dtype=int)
        slices = [data[:, :, i].T for i in indices]
        return slices
    except Exception:
        return None


def plot_slices(ax_row: list, nii_path: str, title: str,
                cmap: str = 'gray', vmin=None, vmax=None,
                label: str = ''):
    """Plot image slices across a row of axes. Returns True if plotted."""
    slices = extract_slices(nii_path, n_slices=len(ax_row))
    if slices is None:
        for ax in ax_row:
            ax.set_visible(False)
        return False
    for ax, sl in zip(ax_row, slices):
        vm = np.nanpercentile(sl, 99) if vmax is None else vmax
        ax.imshow(np.rot90(sl), cmap=cmap, vmin=vmin or 0, vmax=vm,
                  aspect='auto', interpolation='bilinear')
        ax.axis('off')
    ax_row[0].set_title(title, fontsize=7, color=COLORS['text'],
                         loc='left', pad=2)
    if label:
        ax_row[-1].text(1.0, 0.5, label, transform=ax_row[-1].transAxes,
                        fontsize=6, color=COLORS['text'], va='center', ha='left')
    return True


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_header(pdf: PdfPages, sub: str, cap: dict, version: str):
    fig = plt.figure(figsize=(8.5, 2.5))
    fig.patch.set_facecolor(COLORS['header'])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(COLORS['header'])
    ax.axis('off')

    ax.text(0.05, 0.75, f'dwiforge v{version}',
            color='white', fontsize=22, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.05, 0.50, f'QC Report — {sub}',
            color='#ecf0f1', fontsize=14, transform=ax.transAxes)
    ax.text(0.05, 0.30,
            f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}',
            color='#bdc3c7', fontsize=9, transform=ax.transAxes)

    # Acquisition summary (right side)
    acq = cap.get('acquisition', {})
    shells = cap.get('shells', {})
    b_vals = shells.get('b_values', [])
    n_dirs = cap.get('n_dwi', '?')
    pe_dir = acq.get('phase_encoding_direction', acq.get('phase_encoding_axis', '?'))
    single = shells.get('is_single_shell', True)
    shell_str = f"{'Single' if single else 'Multi'}-shell: b={b_vals}"

    info_lines = [
        f"b-values: {b_vals}",
        f"DWI dirs: {n_dirs}",
        f"PE dir: {pe_dir}",
        f"Scanner: {acq.get('manufacturer', '?')} {acq.get('magnetic_field_strength', '')}T",
    ]
    for i, line in enumerate(info_lines):
        ax.text(0.65, 0.80 - i * 0.18, line,
                color='#ecf0f1', fontsize=9, transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def render_stage_status(pdf: PdfPages, cap: dict):
    """One row per stage showing status badge + key info."""
    stages_info = cap.get('_stage_log', {})

    stage_data = []
    for key, label in STAGE_LABELS.items():
        # Infer status from capability profile keys
        status = 'unknown'
        detail = ''

        if key == 'qc-bids':
            status = 'complete' if cap.get('n_dwi') else 'unknown'
            detail = f"{cap.get('n_dwi', '?')} DWI dirs"
        elif key == 'recon-all':
            rc = cap.get('recon_all', {})
            status = rc.get('status', 'unknown')
            detail = 'aparc+aseg ✓' if rc.get('aparc_aseg') else ''
        elif key == 'preprocessing':
            status = 'complete' if cap.get('n_dwi') else 'unknown'
        elif key == 't1w_prep' or key == 't1w-prep':
            tp = cap.get('t1w_prep', {})
            status = 'complete' if tp.get('t1w_brain') else 'skipped'
            detail = tp.get('skull_strip_tool', '')
        elif key == 'epi-correction':
            ec = cap.get('epi_correction', {})
            method = ec.get('method', '')
            status = 'complete' if method else 'unknown'
            sdc = ec.get('sdc_performed', False)
            detail = f"{method} {'(SDC ✓)' if sdc else '(no SDC)'}"
        elif key == 'designer':
            pre = cap.get('preprocessing', {})
            status = pre.get('status', 'unknown')
        elif key == 'tensor-fitting':
            tf = cap.get('tensor_fitting', {})
            status = tf.get('status', 'unknown')
            detail = ', '.join(tf.get('models_run', []))
        elif key == 'noddi':
            nd = cap.get('noddi', {})
            status = nd.get('status', 'unknown')
            detail = nd.get('confidence', '')
        elif key == 'response-functions':
            rf = cap.get('response_functions', {})
            status = rf.get('status', 'unknown')
            detail = rf.get('algorithm', '')
        elif key == 'tractography':
            tr = cap.get('tractography', {})
            status = tr.get('status', 'unknown')
            n = tr.get('n_streamlines', '')
            detail = f"{int(n)/1e6:.0f}M streamlines" if n else ''

        stage_data.append((label, status, detail))

    n = len(stage_data)
    fig_h = max(3.5, n * 0.45 + 1.0)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.set_facecolor(COLORS['bg'])
    ax.axis('off')
    fig.patch.set_facecolor(COLORS['bg'])

    ax.text(0.02, 0.97, 'Pipeline Stage Status',
            fontsize=13, fontweight='bold', color=COLORS['header'],
            transform=ax.transAxes, va='top')

    row_h = 0.85 / n
    for i, (label, status, detail) in enumerate(stage_data):
        y = 0.88 - i * row_h
        color = COLORS.get(status, COLORS['unknown'])

        # Status badge
        bbox = FancyBboxPatch((0.02, y - row_h * 0.6), 0.12, row_h * 0.75,
                              boxstyle='round,pad=0.01',
                              facecolor=color, edgecolor='none',
                              transform=ax.transAxes)
        ax.add_patch(bbox)
        ax.text(0.08, y, status.upper(), fontsize=7, fontweight='bold',
                color='white', transform=ax.transAxes, va='center', ha='center')

        # Stage label
        ax.text(0.16, y, label, fontsize=9, color=COLORS['text'],
                transform=ax.transAxes, va='center')

        # Detail
        if detail:
            ax.text(0.55, y, detail, fontsize=8, color='#7f8c8d',
                    transform=ax.transAxes, va='center')

        # Separator line
        ax.plot([0.02, 0.98], [y - row_h * 0.65, y - row_h * 0.65],
                color=COLORS['light'], linewidth=0.5,
                transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def render_warnings(pdf: PdfPages, cap: dict):
    """Collect all warnings from capability profile and render."""
    warnings = []

    # EPI correction warnings
    ec = cap.get('epi_correction', {})
    if not ec.get('sdc_performed', True):
        warnings.append(('EPI Correction', 'WARN',
                         'No susceptibility distortion correction performed'))
    if ec.get('method') == 'lastresort_ants':
        warnings.append(('EPI Correction', 'WARN',
                         'Last-resort ANTs warp used — lower accuracy than Synb0'))

    # NODDI warnings
    nd = cap.get('noddi', {})
    if nd.get('warning'):
        warnings.append(('NODDI', 'INFO', nd['warning']))

    # Tensor fitting warnings
    tf = cap.get('tensor_fitting', {})
    if tf.get('dki_warning'):
        warnings.append(('Tensor Fitting', 'WARN',
                         'DKI run on single-shell b=1000 — high variance estimates'))
    if tf.get('mask_type') == 'brain_mask':
        warnings.append(('Tensor Fitting', 'INFO',
                         'Brain mask used for fitting (no FAST WM mask available)'))

    # T1w prep
    tp = cap.get('t1w_prep', {})
    if not tp:
        warnings.append(('T1w Prep', 'WARN', 'T1w not available — Synb0 and recon-all skipped'))

    # Tractography missing outputs
    tr = cap.get('tractography', {})
    if tr and not tr.get('connectome_fa'):
        warnings.append(('Tractography', 'INFO', 'Mean-FA connectome not generated'))

    if not warnings:
        return

    fig, ax = plt.subplots(figsize=(8.5, max(2.5, len(warnings) * 0.5 + 1.5)))
    ax.set_facecolor(COLORS['bg'])
    fig.patch.set_facecolor(COLORS['bg'])
    ax.axis('off')

    ax.text(0.02, 0.95, 'Warnings & Notes',
            fontsize=13, fontweight='bold', color=COLORS['header'],
            transform=ax.transAxes, va='top')

    row_h = 0.80 / max(len(warnings), 1)
    for i, (stage, level, msg) in enumerate(warnings):
        y = 0.88 - i * row_h
        color = COLORS['warn'] if level == 'WARN' else COLORS['accent']

        ax.text(0.02, y, f'[{level}]', fontsize=8, fontweight='bold',
                color=color, transform=ax.transAxes, va='center')
        ax.text(0.12, y, f'{stage}:', fontsize=8, fontweight='bold',
                color=COLORS['text'], transform=ax.transAxes, va='center')
        ax.text(0.28, y, msg, fontsize=8, color='#555',
                transform=ax.transAxes, va='center', wrap=True)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def render_image_slices(pdf: PdfPages, cap: dict, work_dir: str):
    """Axial slices of key output volumes."""
    if not NIBABEL_AVAILABLE:
        fig, ax = plt.subplots(figsize=(8.5, 1.5))
        ax.axis('off')
        ax.text(0.5, 0.5, 'Image slices unavailable — nibabel not installed',
                ha='center', va='center', color='#999', fontsize=10,
                transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        return

    tmi_dir = cap.get('tensor_fitting', {}).get('output_dir', '')
    noddi_dir = cap.get('noddi', {}).get('output_dir', '')
    b0_mean = os.path.join(work_dir, 'b0_mean.nii.gz')
    fa_map  = os.path.join(tmi_dir, 'fa_dti.nii') if tmi_dir else ''
    md_map  = os.path.join(tmi_dir, 'md_dti.nii') if tmi_dir else ''
    ndi_map = os.path.join(noddi_dir, 'NODDI_icvf.nii.gz') if noddi_dir else ''
    odi_map = os.path.join(noddi_dir, 'NODDI_odi.nii.gz') if noddi_dir else ''

    volumes = [
        (b0_mean,  'b0 (mean)',   'gray',    None, None),
        (fa_map,   'FA',          'RdYlGn',  0.0,  1.0),
        (md_map,   'MD',          'hot_r',   None, None),
        (ndi_map,  'NDI (NODDI)', 'viridis', 0.0,  1.0),
        (odi_map,  'ODI (NODDI)', 'plasma',  0.0,  1.0),
    ]
    # Filter to existing files
    volumes = [(p, l, c, mn, mx) for p, l, c, mn, mx in volumes
               if p and os.path.exists(p)]

    if not volumes:
        return

    N_SLICES = 5
    n_rows = len(volumes)
    fig = plt.figure(figsize=(8.5, n_rows * 1.4 + 0.8))
    fig.patch.set_facecolor(COLORS['bg'])
    fig.suptitle('Image Slices (axial)', fontsize=11, fontweight='bold',
                 color=COLORS['header'], y=0.98)

    gs = gridspec.GridSpec(n_rows, N_SLICES,
                           hspace=0.08, wspace=0.04,
                           top=0.92, bottom=0.04,
                           left=0.01, right=0.85)

    for row_i, (path, label, cmap, vmin, vmax) in enumerate(volumes):
        axes = [fig.add_subplot(gs[row_i, col]) for col in range(N_SLICES)]
        plot_slices(axes, path, label, cmap=cmap, vmin=vmin, vmax=vmax)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def render_metrics(pdf: PdfPages, cap: dict, work_dir: str):
    """DTI and NODDI metric summary in WM mask."""
    if not NIBABEL_AVAILABLE:
        return

    tf = cap.get('tensor_fitting', {})
    nd = cap.get('noddi', {})
    tmi_dir = tf.get('output_dir', '')
    noddi_dir = nd.get('output_dir', '')
    wm_mask_path = tf.get('mask_path', os.path.join(work_dir, 'wm_mask_dwi.nii.gz'))

    # Load WM mask
    mask = None
    if wm_mask_path and os.path.exists(wm_mask_path):
        try:
            mask = nib.load(wm_mask_path).get_fdata() > 0.5
        except Exception:
            mask = None

    def wm_stats(nii_path):
        if not nii_path or not os.path.exists(nii_path):
            return None, None
        try:
            data = nib.load(nii_path).get_fdata()
            if data.ndim == 4:
                data = data[..., 0]
            vals = data[mask] if mask is not None else data[data > 0]
            vals = vals[np.isfinite(vals)]
            return float(np.mean(vals)), float(np.std(vals))
        except Exception:
            return None, None

    metrics = {}
    for name, fname in [
        ('FA',    'fa_dti.nii'),
        ('MD',    'md_dti.nii'),
        ('AD',    'ad.nii'),
        ('RD',    'rd.nii'),
    ]:
        path = os.path.join(tmi_dir, fname) if tmi_dir else ''
        mu, sd = wm_stats(path)
        if mu is not None:
            metrics[name] = (mu, sd)

    for name, fname in [
        ('NDI',   'NODDI_icvf.nii.gz'),
        ('ODI',   'NODDI_odi.nii.gz'),
        ('ISOVF', 'NODDI_isovf.nii.gz'),
    ]:
        path = os.path.join(noddi_dir, fname) if noddi_dir else ''
        mu, sd = wm_stats(path)
        if mu is not None:
            metrics[name] = (mu, sd)

    if not metrics:
        return

    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    ax.axis('off')
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    ax.text(0.02, 0.95, 'White Matter Metrics (mean ± SD in WM mask)',
            fontsize=11, fontweight='bold', color=COLORS['header'],
            transform=ax.transAxes, va='top')

    col_labels = ['Metric', 'Mean', 'SD', 'Mask']
    mask_label = tf.get('mask_type', 'unknown')
    col_x = [0.05, 0.35, 0.55, 0.70]
    header_y = 0.80

    for cx, cl in zip(col_x, col_labels):
        ax.text(cx, header_y, cl, fontsize=9, fontweight='bold',
                color=COLORS['header'], transform=ax.transAxes)

    row_h = 0.62 / max(len(metrics), 1)
    for i, (name, (mu, sd)) in enumerate(metrics.items()):
        y = 0.73 - i * row_h
        bg_color = COLORS['light'] if i % 2 == 0 else 'white'
        bg = FancyBboxPatch((0.02, y - row_h * 0.4), 0.94, row_h * 0.85,
                             boxstyle='round,pad=0.005',
                             facecolor=bg_color, edgecolor='none',
                             transform=ax.transAxes)
        ax.add_patch(bg)
        ax.text(col_x[0], y, name, fontsize=9, color=COLORS['text'],
                transform=ax.transAxes, va='center', fontweight='bold')
        ax.text(col_x[1], y, f'{mu:.4f}', fontsize=9, color=COLORS['text'],
                transform=ax.transAxes, va='center')
        ax.text(col_x[2], y, f'{sd:.4f}', fontsize=9, color='#7f8c8d',
                transform=ax.transAxes, va='center')
        ax.text(col_x[3], y, mask_label, fontsize=8, color='#95a5a6',
                transform=ax.transAxes, va='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def render_connectome(pdf: PdfPages, cap: dict):
    """Connectome adjacency heatmap and summary statistics."""
    tr = cap.get('tractography', {})
    count_csv = tr.get('connectome_count', '')

    if not count_csv or not os.path.exists(count_csv):
        return

    try:
        matrix = np.loadtxt(count_csv, delimiter=',')
    except Exception:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
    fig.patch.set_facecolor(COLORS['bg'])
    fig.suptitle('Structural Connectome (SIFT2-weighted streamline count)',
                 fontsize=11, fontweight='bold', color=COLORS['header'])

    # Log-scale heatmap
    ax = axes[0]
    log_mat = np.log1p(matrix)
    im = ax.imshow(log_mat, cmap='viridis', aspect='auto', interpolation='none')
    ax.set_title('log(1 + count)', fontsize=9)
    ax.set_xlabel('Region index', fontsize=8)
    ax.set_ylabel('Region index', fontsize=8)
    ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Summary statistics
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_facecolor(COLORS['bg'])

    n_nodes = matrix.shape[0]
    upper = matrix[np.triu_indices(n_nodes, k=1)]
    n_edges = int(np.sum(upper > 0))
    density = n_edges / (n_nodes * (n_nodes - 1) / 2) * 100
    mean_str = float(np.mean(upper[upper > 0])) if np.any(upper > 0) else 0
    max_str  = float(np.max(matrix))

    # FA connectome stats if available
    fa_csv = tr.get('connectome_fa', '')
    fa_stats = ''
    if fa_csv and os.path.exists(fa_csv):
        try:
            fa_mat = np.loadtxt(fa_csv, delimiter=',')
            fa_upper = fa_mat[np.triu_indices(n_nodes, k=1)]
            fa_upper = fa_upper[fa_upper > 0]
            fa_stats = f"\nMean edge FA:  {np.mean(fa_upper):.3f} ± {np.std(fa_upper):.3f}"
        except Exception:
            pass

    stats_text = (
        f"N nodes:        {n_nodes}\n"
        f"N edges:        {n_edges}\n"
        f"Density:        {density:.1f}%\n"
        f"Mean strength:  {mean_str:.1f}\n"
        f"Max strength:   {max_str:.0f}"
        f"{fa_stats}"
    )
    ax2.text(0.1, 0.65, stats_text,
             fontsize=10, family='monospace', color=COLORS['text'],
             transform=ax2.transAxes, va='center',
             bbox=dict(boxstyle='round', facecolor=COLORS['light'],
                       edgecolor=COLORS['accent'], linewidth=1))
    ax2.set_title('Summary statistics', fontsize=9)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.capability_json):
        print(f"ERROR: capability.json not found: {args.capability_json}",
              file=sys.stderr)
        return 1

    cap = load_cap(args.capability_json)

    print(f"Generating QC report for {args.subject}...")

    with PdfPages(args.output) as pdf:
        # Set PDF metadata
        d = pdf.infodict()
        d['Title']   = f'dwiforge QC Report — {args.subject}'
        d['Author']  = f'dwiforge v{args.pipeline_version}'
        d['Subject'] = 'DWI preprocessing QC'
        d['CreationDate'] = datetime.datetime.now()

        render_header(pdf, args.subject, cap, args.pipeline_version)
        render_stage_status(pdf, cap)
        render_warnings(pdf, cap)
        render_image_slices(pdf, cap, args.work_dir)
        render_metrics(pdf, cap, args.work_dir)
        render_connectome(pdf, cap)

    print(f"Report saved: {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
