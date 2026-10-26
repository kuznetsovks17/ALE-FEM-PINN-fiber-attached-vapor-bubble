#!/usr/bin/env python3
"""Reproducible FEM-PINN interface-distance audit.

The FEM interface export stores duplicated endpoints for each boundary edge.
For each requested time, the code reconstructs the piecewise-linear FEM curve,
uses the dense PINN xi export as a piecewise-linear curve, resamples each curve
uniformly in normalized meridional arc length, and evaluates exact point-to-
segment distances to the opposite polyline. Symmetric mean/RMS statistics give
equal weight to the two directed curves. The symmetric maximum is the maximum
of the two directed maxima (a polyline Hausdorff-type diagnostic).
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA=ROOT/'data'/'cross_method'
FEM=DATA/'freefem_fiber_stage11_v8_POINT2_interfaceSnapshots.csv'
PINN=DATA/'point2_PINN_interfaceSnapshots_ZERO_TRAINING.csv'
N_ARC=5001
TIMES=(0.0,0.1,0.2,0.3)
CUTS=(0.0,0.02,0.05,0.075,0.10,0.15)

def fem_polyline(df,t):
    f=df[np.isclose(df.t,t)].sort_values(['edge','endpoint'])
    pairs=[]
    for _,g in f.groupby('edge',sort=True):
        g=g.sort_values('endpoint')
        pairs.append((g.iloc[0][['r','z']].to_numpy(float),
                      g.iloc[-1][['r','z']].to_numpy(float)))
    pts=[pairs[0][0]]
    for p0,p1 in pairs:
        # Follow whichever endpoint continues the current chain.
        nxt=p1 if np.linalg.norm(pts[-1]-p0)<=np.linalg.norm(pts[-1]-p1) else p0
        if np.linalg.norm(nxt-pts[-1])>1e-14:
            pts.append(nxt)
    return np.asarray(pts)

def pinn_polyline(df,t):
    return df[np.isclose(df.t,t)].sort_values('xi')[['r','z']].to_numpy(float)

def resample(poly,n=N_ARC):
    ds=np.linalg.norm(np.diff(poly,axis=0),axis=1)
    s=np.r_[0.0,np.cumsum(ds)]
    sq=np.linspace(0.0,s[-1],n)
    pts=np.c_[np.interp(sq,s,poly[:,0]),np.interp(sq,s,poly[:,1])]
    return pts,sq/s[-1]

def point_to_polyline(points,poly,chunk=300):
    a=poly[:-1]; b=poly[1:]; ab=b-a
    den=np.einsum('ij,ij->i',ab,ab)
    out=np.empty(points.shape[0])
    for i0 in range(0,points.shape[0],chunk):
        p=points[i0:i0+chunk]
        ap=p[:,None,:]-a[None,:,:]
        tau=np.einsum('nmi,mi->nm',ap,ab)/den[None,:]
        tau=np.clip(tau,0.0,1.0)
        proj=a[None,:,:]+tau[:,:,None]*ab[None,:,:]
        d2=np.einsum('nmi,nmi->nm',p[:,None,:]-proj,p[:,None,:]-proj)
        out[i0:i0+p.shape[0]]=np.sqrt(d2.min(axis=1))
    return out

def stats(d_pf,d_fp,eta_p,eta_f,eta0):
    a=d_pf[eta_p>=eta0]; b=d_fp[eta_f>=eta0]
    return {
        'eta0':eta0,
        'symmetric_mean':0.5*(a.mean()+b.mean()),
        'symmetric_rms':np.sqrt(0.5*(np.mean(a*a)+np.mean(b*b))),
        'symmetric_max':max(a.max(),b.max()),
        'PINN_to_FEM_mean':a.mean(),
        'FEM_to_PINN_mean':b.mean(),
        'PINN_to_FEM_rms':np.sqrt(np.mean(a*a)),
        'FEM_to_PINN_rms':np.sqrt(np.mean(b*b)),
        'PINN_to_FEM_max':a.max(),
        'FEM_to_PINN_max':b.max(),
    }

def main():
    fem=pd.read_csv(FEM); pinn=pd.read_csv(PINN)
    rows=[]
    for t in TIMES:
        F=fem_polyline(fem,t); P=pinn_polyline(pinn,t)
        Fr,eta_f=resample(F); Pr,eta_p=resample(P)
        d_pf=point_to_polyline(Pr,F)
        d_fp=point_to_polyline(Fr,P)
        for eta0 in CUTS:
            row={'t':t,'n_arc':N_ARC,'FEM_vertices':len(F),'PINN_export_points':len(P)}
            row.update(stats(d_pf,d_fp,eta_p,eta_f,eta0))
            rows.append(row)
    out=pd.DataFrame(rows)
    out.to_csv(DATA/'interface_distance_audit.csv',index=False)
    print(out.to_string(index=False))

if __name__=='__main__':
    main()
