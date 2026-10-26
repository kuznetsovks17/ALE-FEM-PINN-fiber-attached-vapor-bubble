#!/usr/bin/env python3
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "cross_method"
# Reported-versus-reconstructed gV diagnostic
# Exported from a cleaned Jupyter notebook; notebook outputs are not embedded.

# %% [notebook code cell 1]

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fem_m = pd.read_csv(DATA_DIR / "freefem_fiber_stage11_v8_POINT2_deformationMetrics.csv")
fem_full = pd.read_csv(DATA_DIR / "freefem_fiber_stage11_v8_tangentialGaugeALE.csv")
pinn = pd.read_csv(DATA_DIR / "point2_PINN_deformationMetrics_ZERO_TRAINING.csv")

lookup = {round(float(t),12): float(g) for t,g in zip(fem_full.t, fem_full.gV)}
fem_g = np.array([0.0 if abs(t)<1e-14 else lookup[round(float(t),12)] for t in fem_m.t])

# %% [notebook code cell 2]

def bdf2_series(t, v):
    t=np.asarray(t,float); v=np.asarray(v,float)
    g=np.full_like(v,np.nan); g[0]=0.0
    h=t[1]-t[0]
    g[1]=(v[1]-v[0])/h/v[1]
    for n in range(2,len(v)):
        h=t[n]-t[n-1]
        g[n]=(3*v[n]-4*v[n-1]+v[n-2])/(2*h)/v[n]
    return g

def endpoint_bdf2_stride(t,v,stride):
    t=np.asarray(t,float); v=np.asarray(v,float)
    n=len(v)-1
    h=t[n]-t[n-stride]
    return (3*v[n]-4*v[n-stride]+v[n-2*stride])/(2*h)/v[n]

tf=fem_m.t.to_numpy(float); vf=fem_m.Vratio.to_numpy(float)
tp=pinn.t.to_numpy(float); vp=pinn.Vratio.to_numpy(float)

gf_rec=bdf2_series(tf,vf)
gp_rec=bdf2_series(tp,vp)
gp_ad=pinn.gV.to_numpy(float)

# %% [notebook code cell 3]

print("t=0.3")
print("FEM reported:", fem_g[-1])
print("FEM from V:", gf_rec[-1])
print("PINN AD:", gp_ad[-1])
print("PINN from V:", gp_rec[-1])

for stride in [1,2,3,4,5,6,8,10]:
    if 2*stride < len(tf):
        print(
            "stride", stride,
            "h=", tf[-1]-tf[-1-stride],
            "FEM", endpoint_bdf2_stride(tf,vf,stride),
            "PINN", endpoint_bdf2_stride(tp,vp,stride)
        )

# %% [notebook code cell 4]

fig, ax = plt.subplots(figsize=(7,4.8))
ax.plot(tf,fem_g,label="FEM reported")
ax.plot(tf,gf_rec,linestyle="--",label="FEM reconstructed from V")
ax.plot(tp,gp_ad,label="PINN AD")
ax.plot(tp,gp_rec,linestyle="--",label="PINN reconstructed from V")
ax.set_xlabel("time")
ax.set_ylabel(r"$g_V=\dot V/V$")
ax.grid(True,alpha=0.3)
ax.legend()
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
fig.savefig(FIG_DIR / "gv_reported_vs_reconstructed.png", dpi=300, bbox_inches="tight")
plt.close(fig)
