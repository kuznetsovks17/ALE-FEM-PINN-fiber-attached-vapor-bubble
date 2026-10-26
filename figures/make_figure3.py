#!/usr/bin/env python3
# Reviewer-revised Figure 3 generator
# Exported from a cleaned Jupyter notebook; notebook outputs are not embedded.

# %% [notebook code cell 1]

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ------------------------------------------------------------
# User controls
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data' / 'cross_method'
FIG_DIR = Path(__file__).resolve().parent
T_COMPARE = 0.30
N_ARC = 5001
EXCLUSION_PERCENT = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 15.0])
AF = 300.0 / 686.0

FEM_FILE = DATA_DIR / 'freefem_fiber_stage11_v8_POINT2_interfaceSnapshots.csv'
PINN_FILE = DATA_DIR / 'point2_PINN_interfaceSnapshots_ZERO_TRAINING.csv'

OUTPUT_PNG = FIG_DIR / 'figure3_interface_comparison_twopanel_equal_size.png'
OUTPUT_PDF = FIG_DIR / 'figure3_interface_comparison_twopanel_equal_size.pdf'
OUTPUT_CSV = DATA_DIR / 'figure3_distance_audit.csv'

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

missing = [p.name for p in (FEM_FILE, PINN_FILE) if not p.exists()]
if missing:
    print('Raw production contours are not yet available in this directory.')
    print('Add these files and rerun the notebook:')
    for name in missing:
        print('  -', name)
else:
    print('Found both raw production interface files.')

# %% [notebook code cell 2]

# ------------------------------------------------------------
# Interface extraction and arc-length utilities
# ------------------------------------------------------------

def _validate_interface_table(df, name='interface'):
    required = {'t', 'r', 'z'}
    absent = required.difference(df.columns)
    if absent:
        raise ValueError(f'{name}: missing required columns {sorted(absent)}')


def interface_points(df, t, af=AF):
    """Return one ordered, duplicate-free interface polyline at time t.

    The supplied export format lists successive edge endpoints, so consecutive
    duplicate points are removed. The orientation is then forced to run from
    the pinned rim toward the symmetry axis, which makes eta=0 the rim.
    """
    _validate_interface_table(df)
    tt = df['t'].to_numpy(float)
    sub = df[np.isclose(tt, float(t), atol=1e-10)]
    if len(sub) == 0:
        available = np.sort(df['t'].unique())
        raise ValueError(f'No interface snapshot at t={t}. Available times include {available[:10]} ...')

    arr = sub[['r', 'z']].to_numpy(float)
    keep = [arr[0]]
    for p in arr[1:]:
        if np.linalg.norm(p - keep[-1]) > 1e-12:
            keep.append(p)
    pts = np.asarray(keep)

    # The pinned point is nearest (af, 0). Orient the polyline so it is first.
    d_first = np.linalg.norm(pts[0] - np.array([af, 0.0]))
    d_last = np.linalg.norm(pts[-1] - np.array([af, 0.0]))
    if d_last < d_first:
        pts = pts[::-1].copy()

    return pts


def resample_by_arclength(pts, n=N_ARC):
    """Resample a polyline at uniform normalized meridional arc coordinate eta."""
    pts = np.asarray(pts, float)
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    if not np.isfinite(s[-1]) or s[-1] <= 0:
        raise ValueError('Degenerate interface polyline.')
    eta_nodes = s / s[-1]
    eta = np.linspace(0.0, 1.0, int(n))
    r = np.interp(eta, eta_nodes, pts[:, 0])
    z = np.interp(eta, eta_nodes, pts[:, 1])
    return eta, np.column_stack((r, z))


def normalized_contour(pts, n=N_ARC):
    eta, xy = resample_by_arclength(pts, n=n)
    Rmax = np.max(pts[:, 0])
    H = np.max(pts[:, 1])
    if Rmax <= 0 or H <= 0:
        raise ValueError('Rmax and H must be positive.')
    xhat = np.column_stack((xy[:, 0] / Rmax, xy[:, 1] / H))
    return eta, xhat, Rmax, H

# %% [notebook code cell 3]

# ------------------------------------------------------------
# Point-to-polyline distance routines
# ------------------------------------------------------------

def point_to_polyline_distances(points, polyline, chunk=256):
    """Euclidean distance from each point to a piecewise-linear polyline.

    Distances are exact with respect to the supplied line segments, apart from
    floating-point arithmetic. Chunking avoids a large temporary array.
    """
    P = np.asarray(points, float)
    Q = np.asarray(polyline, float)
    A = Q[:-1]
    B = Q[1:]
    V = B - A
    vv = np.sum(V * V, axis=1)
    vv = np.where(vv > 0, vv, 1.0)

    out = np.empty(len(P), float)
    for i0 in range(0, len(P), chunk):
        C = P[i0:i0+chunk]
        W = C[:, None, :] - A[None, :, :]
        tau = np.sum(W * V[None, :, :], axis=2) / vv[None, :]
        tau = np.clip(tau, 0.0, 1.0)
        proj = A[None, :, :] + tau[:, :, None] * V[None, :, :]
        d2 = np.sum((C[:, None, :] - proj) ** 2, axis=2)
        out[i0:i0+chunk] = np.sqrt(np.min(d2, axis=1))
    return out


def symmetric_distance_metrics(source_a, eta_a, source_b, eta_b, eta0=0.0):
    """Symmetric sampled distances with a source-side rim cutoff.

    For each direction, only source samples with eta >= eta0 are retained;
    the opposite target polyline remains complete, matching the manuscript
    definition. The symmetric mean is the average of the two directed means;
    the symmetric RMS is sqrt(mean of the two directed mean-square distances);
    the sampled maximum is the larger directed maximum.
    """
    ma = eta_a >= eta0
    mb = eta_b >= eta0

    da = point_to_polyline_distances(source_a[ma], source_b)
    db = point_to_polyline_distances(source_b[mb], source_a)

    mean = 0.5 * (da.mean() + db.mean())
    rms = np.sqrt(0.5 * (np.mean(da**2) + np.mean(db**2)))
    maximum = max(da.max(), db.max())
    return mean, rms, maximum

# %% [notebook code cell 4]

# ------------------------------------------------------------
# Load production contours and compute all Figure-3 quantities
# ------------------------------------------------------------

READY = FEM_FILE.exists() and PINN_FILE.exists()

if READY:
    fem_df = pd.read_csv(FEM_FILE)
    pinn_df = pd.read_csv(PINN_FILE)

    fem_raw = interface_points(fem_df, T_COMPARE)
    pinn_raw = interface_points(pinn_df, T_COMPARE)

    eta_fem, fem_xy = resample_by_arclength(fem_raw, N_ARC)
    eta_pinn, pinn_xy = resample_by_arclength(pinn_raw, N_ARC)

    _, fem_hat, Rf, Hf = normalized_contour(fem_raw, N_ARC)
    _, pinn_hat, Rp, Hp = normalized_contour(pinn_raw, N_ARC)

    rows = []
    for pct in EXCLUSION_PERCENT:
        mean, rms, maximum = symmetric_distance_metrics(
            fem_xy, eta_fem, pinn_xy, eta_pinn, eta0=pct/100.0
        )
        rows.append({
            'excluded_arc_percent': pct,
            'mean_over_R0': mean,
            'rms_over_R0': rms,
            'maximum_over_R0': maximum,
        })

    audit = pd.DataFrame(rows)
    audit.to_csv(OUTPUT_CSV, index=False)
    print(audit.to_string(index=False))

    print(f'FEM:  Rmax={Rf:.8f}, H={Hf:.8f}')
    print(f'PINN: Rmax={Rp:.8f}, H={Hp:.8f}')
else:
    print('Skipping production calculations because the two raw contour CSVs are missing.')

# %% [notebook code cell 5]

# ------------------------------------------------------------
# Reviewer-revised two-panel Figure 3
# SAME VISIBLE SUBFIGURE SIZE
# ------------------------------------------------------------

if READY:
    fig, axs = plt.subplots(
        1, 2,
        figsize=(12.8, 5.4),
        gridspec_kw={'width_ratios': [1.0, 1.0]},
        constrained_layout=True
    )

    # ============================================================
    # (a) Raw physical-coordinate overlay
    # ============================================================
    ax = axs[0]

    ax.plot(fem_xy[:, 0], fem_xy[:, 1], label='ALE-FEM')
    ax.plot(pinn_xy[:, 0], pinn_xy[:, 1], label='PINN')
    ax.plot(AF, 0.0, 'o', ms=4, label='pinned rim')

    ax.set_xlabel(r'$r/R_0$')
    ax.set_ylabel(r'$z/R_0$')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, loc='lower right')
    ax.set_title('(a) Unnormalized physical overlay')

    # ------------------------------------------------------------
    # IMPORTANT:
    # Keep the physical coordinates undistorted AND make panel (a)
    # occupy the same square axes box as panel (b).
    #
    # Because z spans much more than r, "aspect='equal'" would normally
    # make panel (a) visibly narrower. We therefore enlarge the r-range
    # to match the z-range. The extra right-hand space is then used for
    # the pinned-rim inset.
    # ------------------------------------------------------------
    zmin = -0.08
    zmax = max(fem_xy[:, 1].max(), pinn_xy[:, 1].max()) + 0.08
    zspan = zmax - zmin

    xmin_main = -0.02
    xmax_main = xmin_main + zspan

    ax.set_xlim(xmin_main, xmax_main)
    ax.set_ylim(zmin, zmax)

    # Preserve 1:1 physical scaling in r and z.
    ax.set_aspect('equal', adjustable='box')

    # Force the visible axes box itself to be square.
    ax.set_box_aspect(1)

    # ------------------------------------------------------------
    # Pinned-rim inset, placed in the intentionally reserved
    # blank space on the right side of panel (a).
    # ------------------------------------------------------------
    rim_fraction = 0.15
    mf = eta_fem <= rim_fraction
    mp = eta_pinn <= rim_fraction

    rim_pts = np.vstack((
        fem_xy[mf],
        pinn_xy[mp],
        np.array([[AF, 0.0]])
    ))

    xmin, ymin = rim_pts.min(axis=0)
    xmax, ymax_rim = rim_pts.max(axis=0)

    dx = max(xmax - xmin, 0.03)
    dy = max(ymax_rim - ymin, 0.03)

    axins = inset_axes(
        ax,
        width='38%',
        height='68%',
        loc='upper right',
        borderpad=1.5
    )

    axins.plot(fem_xy[:, 0], fem_xy[:, 1])
    axins.plot(pinn_xy[:, 0], pinn_xy[:, 1])
    axins.plot(AF, 0.0, 'o', ms=3)

    axins.set_xlim(
        xmin - 0.08 * dx,
        xmax + 0.08 * dx
    )
    axins.set_ylim(
        max(-0.01, ymin - 0.08 * dy),
        ymax_rim + 0.10 * dy
    )

    axins.tick_params(labelsize=7, direction='in')
    axins.grid(True, alpha=0.20)
    axins.set_title('pinned-rim detail', fontsize=8)

    # ============================================================
    # (b) Common-coordinate distances
    # ============================================================
    ax = axs[1]

    x = audit['excluded_arc_percent'].to_numpy()

    ax.plot(
        x,
        1e3 * audit['mean_over_R0'],
        marker='o',
        label='Mean'
    )
    ax.plot(
        x,
        1e3 * audit['rms_over_R0'],
        marker='s',
        linestyle='--',
        label='RMS'
    )
    ax.plot(
        x,
        1e3 * audit['maximum_over_R0'],
        marker='^',
        linestyle='-.',
        label='Maximum'
    )

    ax.set_xlabel('Excluded meridional arc [%]')
    ax.set_ylabel(r'Distance / $R_0$ ($\times 10^{-3}$)')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title('(b) Distances in common coordinates')

    # Same visible axes-box size as panel (a).
    ax.set_box_aspect(1)

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------
    fig.savefig(
        OUTPUT_PNG,
        dpi=400,
        bbox_inches='tight'
    )
    fig.savefig(
        OUTPUT_PDF,
        bbox_inches='tight'
    )

    plt.close(fig)

    print('Saved:')
    print(' ', OUTPUT_PNG)
    print(' ', OUTPUT_PDF)
    print(' ', OUTPUT_CSV)

else:
    print(
        'Figure not generated: add the two raw interface CSV files '
        'listed in the first cell.'
    )

# %% [notebook code cell 7]
from pathlib import Path

# ------------------------------------------------------------
# Save the two subfigures separately, without changing the plot
# ------------------------------------------------------------

OUT_A = FIG_DIR / 'figure3_panel_a_overlay.png'
OUT_B = FIG_DIR / 'figure3_panel_b_distances.png'

OUT_A_PDF = FIG_DIR / 'figure3_panel_a_overlay.pdf'
OUT_B_PDF = FIG_DIR / 'figure3_panel_b_distances.pdf'

# Make sure the figure is fully drawn, so tight bounding boxes are correct
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

def save_axis(fig, ax, png_path, pdf_path=None, pad_inches=0.03):
    # Tight bounding box of this axis, including labels, ticks, title, legend
    bbox = ax.get_tightbbox(renderer).expanded(1.02, 1.02)

    # Convert from display units to inches for savefig
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())

    fig.savefig(
        png_path,
        dpi=400,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches
    )

    if pdf_path is not None:
        fig.savefig(
            pdf_path,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches
        )

# Save panel (a)
save_axis(fig, axs[0], OUT_A, OUT_A_PDF)

# Save panel (b)
save_axis(fig, axs[1], OUT_B, OUT_B_PDF)

print('Saved separate subfigures:')
print(' ', OUT_A)
print(' ', OUT_A_PDF)
print(' ', OUT_B)
print(' ', OUT_B_PDF)
