#!/usr/bin/env python3
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "cross_method"
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
# Geometry-sensitive diagnostics
# Exported from a cleaned Jupyter notebook; notebook outputs are not embedded.

# %% [notebook code cell 1]

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

FILES = {
    "large_fem_interface": str(DATA_DIR / "freefem_fiber_stage11_v8_POINT2_interfaceSnapshots.csv"),
    "large_fem_metrics": str(DATA_DIR / "freefem_fiber_stage11_v8_POINT2_deformationMetrics.csv"),
    "large_pinn_interface": str(DATA_DIR / "point2_PINN_interfaceSnapshots_ZERO_TRAINING.csv"),
    "large_pinn_metrics": str(DATA_DIR / "point2_PINN_deformationMetrics_ZERO_TRAINING.csv"),
    "small_fem_interface": str(DATA_DIR / "freefem_fiber_point14_af005_POINT4_interfaceSnapshots_t030.csv"),
    "small_fem_metrics": str(DATA_DIR / "freefem_fiber_point14_af005_POINT4_deformationMetrics_t030.csv"),
}

for key, name in FILES.items():
    if not Path(name).exists():
        raise FileNotFoundError(f"Missing {key}: {name}")

large_fem_i = pd.read_csv(FILES["large_fem_interface"])
large_fem_m = pd.read_csv(FILES["large_fem_metrics"])
large_pinn_i = pd.read_csv(FILES["large_pinn_interface"])
large_pinn_m = pd.read_csv(FILES["large_pinn_metrics"])
small_fem_i = pd.read_csv(FILES["small_fem_interface"])
small_fem_m = pd.read_csv(FILES["small_fem_metrics"])

print("Loaded all existing Point-2/4 outputs. No simulation or training will be run.")

# %% [notebook code cell 2]

def interface_points(df, t):
    # Return ordered unique (r,z) points at time t.
    sub = df[np.isclose(df["t"].to_numpy(float), float(t), atol=1e-10)]
    if len(sub) == 0:
        raise ValueError(f"No interface snapshot at t={t}")
    arr = sub[["r","z"]].to_numpy(float)
    out = [arr[0]]
    for p in arr[1:]:
        if np.linalg.norm(p - out[-1]) > 1e-12:
            out.append(p)
    return np.asarray(out)

def normalized_interface(df, t, n=2000):
    # Reparameterize by normalized arc length eta and normalize r,z by Rmax,H.
    pts = interface_points(df, t)
    ds = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    eta_nodes = s / s[-1]
    eta = np.linspace(0.0, 1.0, n)
    r = np.interp(eta, eta_nodes, pts[:,0])
    z = np.interp(eta, eta_nodes, pts[:,1])
    Rmax = np.max(pts[:,0])
    H = np.max(pts[:,1])
    return eta, r/Rmax, z/H, Rmax, H

def shape_defect(df, t, eta0=0.05, n=2000):
    # Scale-free deviation from the case's own initial normalized shape.
    eta, r0, z0, _, _ = normalized_interface(df, 0.0, n=n)
    eta, r, z, _, _ = normalized_interface(df, t, n=n)
    d2 = (r-r0)**2 + (z-z0)**2
    mask = eta >= eta0
    rms = np.sqrt(np.trapezoid(d2[mask], eta[mask]) /
                  (eta[mask][-1] - eta[mask][0]))
    dmax = np.sqrt(d2[mask]).max()
    return rms, dmax

def shape_defect_series(df, eta0=0.05):
    times = np.sort(df["t"].unique())
    rows = []
    for t in times:
        rms, dmax = shape_defect(df, t, eta0=eta0)
        rows.append({"t": t, "Eshape": rms, "Eshape_max": dmax})
    return pd.DataFrame(rows)

ETA0 = 0.05
small_shape = shape_defect_series(small_fem_i, ETA0)
large_fem_shape = shape_defect_series(large_fem_i, ETA0)
large_pinn_shape = shape_defect_series(large_pinn_i, ETA0)

print("Default contact-line exclusion eta0 =", ETA0)

# %% [notebook code cell 3]

rows = []
for label, df in [
    ("small FEM", small_fem_i),
    ("large FEM", large_fem_i),
    ("large PINN", large_pinn_i),
]:
    rms, dmax = shape_defect(df, 0.3, eta0=ETA0)
    rows.append({"case": label, "t": 0.3, "Eshape": rms, "Eshape_max": dmax})

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))

lf = summary.loc[summary.case=="large FEM","Eshape"].iloc[0]
lp = summary.loc[summary.case=="large PINN","Eshape"].iloc[0]
print(f"Large-fiber FEM–PINN Eshape relative difference: {abs(lp-lf)/abs(lf)*100:.3f}%")

# %% [notebook code cell 4]

target_V = float(large_fem_m.loc[np.isclose(large_fem_m.t,0.3), "Vratio"].iloc[0])
idx = np.argmin(np.abs(small_fem_m["Vratio"].to_numpy() - target_V))
small_match = small_fem_m.iloc[idx]
t_small_match = float(small_match["t"])

E_small_match, E_small_match_max = shape_defect(
    small_fem_i, t_small_match, eta0=ETA0
)
E_large_final, E_large_final_max = shape_defect(
    large_fem_i, 0.3, eta0=ETA0
)

matched = pd.DataFrame([
    {
        "case": "small FEM nearest matched volume",
        "t": t_small_match,
        "Vratio": float(small_match["Vratio"]),
        "Eshape": E_small_match,
        "Eshape_max": E_small_match_max,
    },
    {
        "case": "large FEM final",
        "t": 0.3,
        "Vratio": target_V,
        "Eshape": E_large_final,
        "Eshape_max": E_large_final_max,
    },
])
print(matched.to_string(index=False))
print("Eshape ratio small/large at matched volume:",
      E_small_match/E_large_final)

# %% [notebook code cell 6]

fig, ax = plt.subplots(figsize=(6.5,4.5))
ax.plot(small_shape.t, small_shape.Eshape, marker="o", markevery=max(1,len(small_shape)//10), label="small fiber FEM")
ax.plot(large_fem_shape.t, large_fem_shape.Eshape, marker="s", markevery=max(1,len(large_fem_shape)//10), label="large fiber FEM")
ax.plot(large_pinn_shape.t, large_pinn_shape.Eshape, linestyle="--", label="large fiber PINN")
ax.set_xlabel("time")
ax.set_ylabel(r"$E_{\rm shape}$")
ax.set_title("Scale-free interface-shape evolution")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "point4_shape_self_similarity_defect_vs_time.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# %% [notebook code cell 7]

fig, ax = plt.subplots(figsize=(6.5,4.5))
ax.plot(small_fem_m.Vratio, small_fem_m.aspectRel, marker="o", markevery=max(1,len(small_fem_m)//10), label="small fiber FEM")
ax.plot(large_fem_m.Vratio, large_fem_m.aspectRel, marker="s", markevery=max(1,len(large_fem_m)//10), label="large fiber FEM")
ax.plot(large_pinn_m.Vratio, large_pinn_m.aspectRel, linestyle="--", label="large fiber PINN")
ax.set_xlabel(r"$V/V_0$")
ax.set_ylabel(r"$A_{\rm rel}$")
ax.set_title("Global deformation versus compression state")
ax.grid(True, alpha=0.3)
ax.legend()
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(FIG_DIR / "point4_global_aspect_vs_volume.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# %% [notebook code cell 8]

fig, ax = plt.subplots(figsize=(5.5,5.5))
for t in [0.0,0.1,0.2,0.3]:
    eta, rhat, zhat, _, _ = normalized_interface(large_fem_i, t)
    ax.plot(rhat, zhat, label=f"t={t:.1f}")
ax.set_xlabel(r"$r/R_{\max}(t)$")
ax.set_ylabel(r"$z/H(t)$")
ax.set_title("Large-fiber FEM: normalized shape evolution")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "point4_largeFEM_normalized_shape_evolution.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# %% [notebook code cell 9]

fig, ax = plt.subplots(figsize=(5.5,5.5))
for label, df in [("FEM", large_fem_i), ("PINN", large_pinn_i)]:
    eta, rhat, zhat, _, _ = normalized_interface(df, 0.3)
    ax.plot(rhat, zhat, label=label)
ax.set_xlabel(r"$r/R_{\max}(t)$")
ax.set_ylabel(r"$z/H(t)$")
ax.set_title("Large fiber at t=0.3: FEM–PINN normalized shape")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "point4_largeFEM_PINN_normalized_shape_t030.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# %% [notebook code cell 10]

fig, ax = plt.subplots(figsize=(5.5,5.5))
for label, df, t in [
    (f"small FEM, t={t_small_match:.2f}", small_fem_i, t_small_match),
    ("large FEM, t=0.30", large_fem_i, 0.3),
]:
    eta, rhat, zhat, _, _ = normalized_interface(df, t)
    ax.plot(rhat, zhat, label=label)
ax.set_xlabel(r"$r/R_{\max}(t)$")
ax.set_ylabel(r"$z/H(t)$")
ax.set_title("Normalized shape at approximately matched volume")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "point4_small_large_matchedVolume_normalized_shape.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# %% [notebook code cell 12]

def axis_circle_metrics(df, t, fit_zmin=0.20, shoulder_zmin=0.03, shoulder_zmax=0.20):
    pts = interface_points(df, t)
    r, z = pts[:,0], pts[:,1]
    H = z.max()
    fit = (z/H >= fit_zmin)

    def residual(x):
        zc, R = x
        return np.sqrt(r[fit]**2 + (z[fit]-zc)**2) - R

    x0 = [H/2, max(r.max(),H/2)]
    sol = least_squares(residual, x0)
    zc, R = sol.x
    delta = np.sqrt(r**2 + (z-zc)**2) - R

    bulk = fit
    shoulder = (z/H >= shoulder_zmin) & (z/H < shoulder_zmax)

    return {
        "t": t,
        "zc_fit": zc,
        "R_fit": R,
        "Ecirc_bulk": np.sqrt(np.mean(delta[bulk]**2))/R,
        "Eshoulder_rms": np.sqrt(np.mean(delta[shoulder]**2))/R,
        "Eshoulder_max": np.max(np.abs(delta[shoulder]))/R,
    }

circle_rows = []
for case, df in [
    ("small FEM", small_fem_i),
    ("large FEM", large_fem_i),
    ("large PINN", large_pinn_i),
]:
    for t in [0.1,0.2,0.3]:
        row = axis_circle_metrics(df,t)
        row["case"] = case
        circle_rows.append(row)

circle_table = pd.DataFrame(circle_rows)
print(circle_table.to_string(index=False))

# %% [notebook code cell 13]

shape_export = pd.concat([
    small_shape.assign(case="small FEM"),
    large_fem_shape.assign(case="large FEM"),
    large_pinn_shape.assign(case="large PINN"),
], ignore_index=True)

shape_export.to_csv(str(DATA_DIR / "point4_shape_self_similarity_metrics.csv"), index=False)
circle_table.to_csv(str(DATA_DIR / "point4_circle_shoulder_metrics.csv"), index=False)
matched.to_csv(str(DATA_DIR / "point4_matched_volume_shape_metrics.csv"), index=False)

print("Wrote:")
print("  point4_shape_self_similarity_metrics.csv")
print("  point4_circle_shoulder_metrics.csv")
print("  point4_matched_volume_shape_metrics.csv")
print("  plus five PNG figures")
