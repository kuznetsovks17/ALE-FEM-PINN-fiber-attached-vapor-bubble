#!/usr/bin/env python3
# ALE-FEM and PINN method schematics
# Exported from a cleaned Jupyter notebook; notebook outputs are not embedded.

# %% [notebook code cell 1]

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import qmc
from matplotlib.lines import Line2D

# ----------------------------
# Typography
# ----------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 10.5,
})

# ----------------------------
# Geometry
# ----------------------------
af = 300.0 / 686.0
zc = np.sqrt(1.0 - af**2)
theta_rim = np.arccos(-zc)

R_VIEW = 0.62
Z_VIEW_MIN = 1.50
Z_VIEW_MAX = 2.00

def interface_initial(xi):
    xi = np.asarray(xi)
    th = theta_rim * (1.0 - xi)
    r = np.sin(th)
    z = zc + np.cos(th)
    return r, z

def interface_fallback_t03(xi):
    xi = np.asarray(xi)
    r0, z0 = interface_initial(xi)
    radial_shape = xi * (0.85 + 0.15*np.sin(np.pi*xi))
    r = r0 * (1.0 - 0.058 * radial_shape)
    H0 = zc + 1.0
    Btop = 2.0*xi - xi**2
    Blocal = xi * (1.0 - xi)**2
    z = z0 - 0.031 * H0 * Btop - 0.012 * Blocal
    return r, z

xi_dense = np.linspace(0.0, 1.0, 500)
r0_dense, z0_dense = interface_initial(xi_dense)
r1_dense, z1_dense = interface_fallback_t03(xi_dense)

def inside_initial_vapor(r, z):
    return (z >= 0.0) & ((r**2 + (z-zc)**2) <= 1.0 + 1e-10)

rng = np.random.default_rng(7)

# ----------------------------
# FEM mesh reconstruction
# ----------------------------
def build_phase_triangulation(phase="vapor", n_interior=1200):
    xi_b = np.linspace(0.0, 1.0, 150)
    rg, zg = interface_initial(xi_b)
    pts = []

    if phase == "vapor":
        pts += list(zip(rg, zg))
        pts += list(zip(np.linspace(0, af, 55), np.zeros(55)))
        pts += list(zip(np.zeros(75), np.linspace(0, zc+1.0, 75)))

        cand_r = rng.uniform(0, 1.05, n_interior*4)
        cand_z = rng.uniform(0, zc+1.02, n_interior*4)
        mask = inside_initial_vapor(cand_r, cand_z)
        pts += list(zip(cand_r[mask][:n_interior], cand_z[mask][:n_interior]))
    else:
        pts += list(zip(rg, zg))
        pts += list(zip(np.linspace(0.0, 0.95, 100), np.full(100, Z_VIEW_MAX)))
        pts += list(zip(np.full(80, 0.95), np.linspace(1.10, Z_VIEW_MAX, 80)))
        pts += list(zip(np.zeros(80), np.linspace(Z_VIEW_MAX, zc+1.0, 80)))

        cand_r = rng.uniform(0, 0.95, n_interior*5)
        cand_z = rng.uniform(1.10, Z_VIEW_MAX, n_interior*5)
        mask = ~inside_initial_vapor(cand_r, cand_z)
        pts += list(zip(cand_r[mask][:n_interior], cand_z[mask][:n_interior]))

    P = np.unique(np.round(np.asarray(pts, float), 8), axis=0)
    tri = Delaunay(P)
    T = tri.simplices.copy()
    C = P[T].mean(axis=1)

    if phase == "vapor":
        keep = inside_initial_vapor(C[:,0], C[:,1])
    else:
        keep = (
            (C[:,0] >= 0) & (C[:,0] <= 0.95) &
            (C[:,1] >= 1.10) & (C[:,1] <= Z_VIEW_MAX) &
            ~inside_initial_vapor(C[:,0], C[:,1])
        )
    return P, T[keep]

Pv, Tv = build_phase_triangulation("vapor", 900)
Pl, Tl = build_phase_triangulation("liquid", 900)

I0 = np.column_stack((r0_dense, z0_dense))
I1 = np.column_stack((r1_dense, z1_dense))
dI = I1 - I0
tree = cKDTree(I0)

def ale_displace_points(P, phase):
    P = np.asarray(P, float)
    dist, idx = tree.query(P, k=1)
    d = dI[idx]
    ell = 0.68 if phase == "liquid" else 0.95
    weight = np.exp(-(dist/ell)**2)

    r, z = P[:,0], P[:,1]
    if phase == "liquid":
        outer_r = np.clip((0.95-r)/(0.95-0.55), 0, 1)
        outer_top = np.clip((Z_VIEW_MAX-z)/(Z_VIEW_MAX-1.82), 0, 1)
        weight *= outer_r * outer_top
    else:
        tip_fix = np.clip(z/0.30, 0, 1)
        weight *= 0.30 + 0.70*tip_fix

    return P + weight[:,None] * d

Pv_ale = ale_displace_points(Pv, "vapor")
Pl_ale = ale_displace_points(Pl, "liquid")

def pick_arrow_points(P, zlow=1.57, zhigh=1.90, rmax=0.56, n=10):
    mask = (P[:,0] < rmax) & (P[:,1] > zlow) & (P[:,1] < zhigh)
    idx = np.where(mask)[0]
    if len(idx) > n:
        idx = idx[np.linspace(0, len(idx)-1, n, dtype=int)]
    return idx

arr_liq = pick_arrow_points(Pl, n=8)
arr_vap = pick_arrow_points(Pv, n=7)

# ----------------------------
# PINN point clouds
# ----------------------------
def interface_radius_from_profile(z, rg, zg):
    order = np.argsort(zg)
    return np.interp(np.asarray(z), zg[order], rg[order], left=0.0, right=0.0)

def inside_deformed_vapor(r, z):
    z = np.asarray(z)
    r = np.asarray(r)
    rb = interface_radius_from_profile(z, r1_dense, z1_dense)
    return (z >= 0.0) & (z <= np.max(z1_dense)+1e-10) & (r <= rb + 1e-10)

def sobol_in_box(n_total, lo, hi, seed=0):
    d = len(lo)
    m = int(np.ceil(np.log2(n_total)))
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    pts = sampler.random_base2(m)
    return qmc.scale(pts, lo, hi)[:n_total]

cand_l1 = sobol_in_box(512, [0.0, 1.10], [0.95, Z_VIEW_MAX], seed=17)
liq1_all = cand_l1[(~inside_deformed_vapor(cand_l1[:,0], cand_l1[:,1]))][:64]

# Remove liquid points from the immediate interface neighborhood
dist_to_interface = np.min(
    np.sqrt(
        (liq1_all[:,0,None]-r1_dense[None,:])**2 +
        (liq1_all[:,1,None]-z1_dense[None,:])**2
    ),
    axis=1
)
liq1 = liq1_all[dist_to_interface > 0.06]

sampler_v = qmc.Sobol(d=2, scramble=True, seed=23)
uv = sampler_v.random_base2(9)[:480]
eta = uv[:,0]
xi = uv[:,1]

def map_vapor_points(eta, xi, rg, zg):
    xi_prof = np.linspace(0.0, 1.0, len(rg))
    rG = np.interp(xi, xi_prof, rg)
    zG = np.interp(xi, xi_prof, zg)
    return np.column_stack((eta * rG, zG))

vap0 = map_vapor_points(eta, xi, r0_dense, z0_dense)
vap1 = map_vapor_points(eta, xi, r1_dense, z1_dense)

xi_if = np.linspace(0.0, 1.0, 60)
if0 = np.column_stack(interface_initial(xi_if))
if1 = np.column_stack(interface_fallback_t03(xi_if))

# ----------------------------
# Color system
# ----------------------------
initial_purple = "#7B2CBF"   # solid purple, not light purple
deformed_blue = "#2171B5"
initial_orange = "#F6A03C"
deformed_orange = "#E6550D"
arrow_black = "black"
arrow_red = "#CB181D"

def draw_tri_mesh(ax, P, T, color, lw=0.30, alpha=0.70):
    tri = mtri.Triangulation(P[:,0], P[:,1], T)
    ax.triplot(tri, lw=lw, alpha=alpha, color=color)

def finish_axis(ax, title):
    ax.set_xlim(0, R_VIEW)
    ax.set_ylim(Z_VIEW_MIN, Z_VIEW_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$z$")
#    ax.set_title(title)
    ax.tick_params(direction="in")

mesh_arrow_width = 0.0048
interface_arrow_width = 0.0060

# ============================================================
# FIGURE 1 — FEM
# ============================================================
fig1, ax = plt.subplots(figsize=(6.8, 6.0))

draw_tri_mesh(ax, Pl, Tl, color=initial_purple, lw=0.24, alpha=0.55)
draw_tri_mesh(ax, Pv, Tv, color=initial_purple, lw=0.24, alpha=0.55)
draw_tri_mesh(ax, Pl_ale, Tl, color=deformed_blue, lw=0.31, alpha=0.82)
draw_tri_mesh(ax, Pv_ale, Tv, color=deformed_blue, lw=0.31, alpha=0.82)

ax.plot(r0_dense, z0_dense, lw=1.8, color=initial_orange)
ax.plot(r1_dense, z1_dense, lw=2.2, color=deformed_orange)

dPl = Pl_ale - Pl
dPv = Pv_ale - Pv
ax.quiver(
    Pl[arr_liq,0], Pl[arr_liq,1], dPl[arr_liq,0], dPl[arr_liq,1],
    angles="xy", scale_units="xy", scale=1,
    width=mesh_arrow_width, color=arrow_black
)
ax.quiver(
    Pv[arr_vap,0], Pv[arr_vap,1], dPv[arr_vap,0], dPv[arr_vap,1],
    angles="xy", scale_units="xy", scale=1,
    width=mesh_arrow_width, color=arrow_black
)

idx_if_fem = np.where((if0[:,1] > 1.57) & (if0[:,1] < 1.90))[0]
idx_if_fem = idx_if_fem[
    np.linspace(0, len(idx_if_fem)-1, min(6, len(idx_if_fem)), dtype=int)
]
ax.quiver(
    if0[idx_if_fem,0], if0[idx_if_fem,1],
    (if1-if0)[idx_if_fem,0], (if1-if0)[idx_if_fem,1],
    angles="xy", scale_units="xy", scale=1,
    width=interface_arrow_width, color=arrow_red
)

finish_axis(ax, "FEM: ALE displacement near the moving interface")

handles_fem = [
    Line2D([0],[0], color=initial_purple, lw=1.8, label="initial mesh"),
    Line2D([0],[0], color=deformed_blue, lw=1.8, label="deformed mesh"),
    Line2D([0],[0], color=initial_orange, lw=1.8, label="initial interface"),
    Line2D([0],[0], color=deformed_orange, lw=1.8, label="deformed interface"),
    Line2D([0],[0], color=arrow_black, marker=r"$\rightarrow$", linestyle="None",
           markersize=12, label="mesh motion"),
    Line2D([0],[0], color=arrow_red, marker=r"$\rightarrow$", linestyle="None",
           markersize=12, label="interface motion"),
]
ax.legend(
    handles=handles_fem,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.17),
    ncol=3,
    frameon=False,
    handlelength=2.1,
    columnspacing=1.2,
    handletextpad=0.6,
    borderaxespad=0.0
)

fig1.subplots_adjust(bottom=0.27)
FIG_DIR = Path(__file__).resolve().parent
FEM_PNG = FIG_DIR / "FEM_ALE_interface_zoom_v5.png"
FEM_PDF = FIG_DIR / "FEM_ALE_interface_zoom_v5.pdf"
fig1.savefig(FEM_PNG, dpi=300, bbox_inches="tight")
fig1.savefig(FEM_PDF, bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 2 — PINN
# ============================================================
fig2, ax = plt.subplots(figsize=(6.8, 6.0))

pt_liq_size = 34
pt_vap_size = 24
pt_if_size = 42

ax.scatter(
    vap0[:,0], vap0[:,1],
    s=pt_vap_size, marker=".", alpha=0.72, color=initial_purple
)
ax.scatter(
    if0[:,0], if0[:,1],
    s=pt_if_size, marker="x", alpha=0.85, color=initial_orange
)
ax.plot(r0_dense, z0_dense, lw=1.8, color=initial_orange)

ax.scatter(
    liq1[:,0], liq1[:,1],
    s=pt_liq_size, marker="o", alpha=0.72, color="tab:green"
)
ax.scatter(
    vap1[:,0], vap1[:,1],
    s=pt_vap_size, marker=".", alpha=0.92, color=deformed_blue
)
ax.scatter(
    if1[:,0], if1[:,1],
    s=pt_if_size, marker="x", alpha=0.98, color=deformed_orange
)
ax.plot(r1_dense, z1_dense, lw=2.0, color=deformed_orange)

idx_move = np.where(
    (vap0[:,1] > 1.57) &
    (vap0[:,1] < 1.90) &
    (vap0[:,0] < 0.53)
)[0]
if len(idx_move) > 18:
    idx_move = idx_move[
        np.linspace(0, len(idx_move)-1, 18, dtype=int)
    ]
dvap = vap1 - vap0

ax.quiver(
    vap0[idx_move,0], vap0[idx_move,1],
    dvap[idx_move,0], dvap[idx_move,1],
    angles="xy", scale_units="xy", scale=1,
    width=mesh_arrow_width, color=arrow_black
)

idx_if_move = np.where(
    (if0[:,1] > 1.57) &
    (if0[:,1] < 1.90)
)[0]
idx_if_move = idx_if_move[
    np.linspace(0, len(idx_if_move)-1, min(6, len(idx_if_move)), dtype=int)
]
ax.quiver(
    if0[idx_if_move,0], if0[idx_if_move,1],
    (if1-if0)[idx_if_move,0], (if1-if0)[idx_if_move,1],
    angles="xy", scale_units="xy", scale=1,
    width=interface_arrow_width, color=arrow_red
)

finish_axis(ax, "PINN: moving collocation/sampling near the interface")

handles_pinn = [
    Line2D([0],[0], marker=".", color=initial_purple, linestyle="None",
           markersize=13, label="initial vapor points"),
    Line2D([0],[0], marker=".", color=deformed_blue, linestyle="None",
           markersize=13, label="deformed vapor points"),
    Line2D([0],[0], marker="o", color="tab:green", linestyle="None",
           markersize=7, label="liquid points"),
    Line2D([0],[0], color=initial_orange, lw=1.8, label="initial interface"),
    Line2D([0],[0], color=deformed_orange, lw=1.8, label="deformed interface"),
    Line2D([0],[0], color=arrow_black, marker=r"$\rightarrow$", linestyle="None",
           markersize=12, label="vapor-point motion"),
    Line2D([0],[0], color=arrow_red, marker=r"$\rightarrow$", linestyle="None",
           markersize=12, label="interface motion"),
]
ax.legend(
    handles=handles_pinn,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.17),
    ncol=3,
    frameon=False,
    handlelength=2.1,
    columnspacing=1.15,
    handletextpad=0.6,
    borderaxespad=0.0
)

fig2.subplots_adjust(bottom=0.29)
PINN_PNG = FIG_DIR / "PINN_moving_sampling_zoom_v5.png"
PINN_PDF = FIG_DIR / "PINN_moving_sampling_zoom_v5.pdf"
fig2.savefig(PINN_PNG, dpi=300, bbox_inches="tight")
fig2.savefig(PINN_PDF, bbox_inches="tight")
plt.close()

print("Saved:", FEM_PNG.resolve())
print("Saved:", FEM_PDF.resolve())
print("Saved:", PINN_PNG.resolve())
print("Saved:", PINN_PDF.resolve())

# %% [notebook code cell 3]
import tensorflow as tf
print(tf.__version__)

# %% [notebook code cell 4]

