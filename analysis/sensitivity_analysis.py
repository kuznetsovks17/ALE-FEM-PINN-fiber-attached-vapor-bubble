#!/usr/bin/env python3
"""Reproduce capillarity and outer-domain sensitivity tables/figures for v4 manuscript."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'; FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)

def poly(df,t):
    f=df[np.isclose(df.t,t)].sort_values(['edge','endpoint'])
    pairs=[]
    for _,g in f.groupby('edge',sort=True):
        g=g.sort_values('endpoint')
        pairs.append((g.iloc[0][['r','z']].to_numpy(float),g.iloc[-1][['r','z']].to_numpy(float)))
    pts=[pairs[0][0]]
    for p0,p1 in pairs:
        nxt=p1 if np.linalg.norm(pts[-1]-p0)<=np.linalg.norm(pts[-1]-p1) else p0
        if np.linalg.norm(nxt-pts[-1])>1e-14: pts.append(nxt)
    return np.asarray(pts)

def resamp(p,n=5001):
    ds=np.linalg.norm(np.diff(p,axis=0),axis=1); s=np.r_[0,np.cumsum(ds)]
    q=np.linspace(0,s[-1],n)
    return np.c_[np.interp(q,s,p[:,0]),np.interp(q,s,p[:,1])],q/s[-1]

def d2poly(P,Q,chunk=400):
    a=Q[:-1]; b=Q[1:]; ab=b-a; den=np.einsum('ij,ij->i',ab,ab); out=np.empty(len(P))
    for i in range(0,len(P),chunk):
        p=P[i:i+chunk]; ap=p[:,None,:]-a[None,:,:]
        tau=np.clip(np.einsum('nmi,mi->nm',ap,ab)/den[None,:],0,1)
        proj=a[None,:,:]+tau[:,:,None]*ab[None,:,:]
        d2=np.einsum('nmi,nmi->nm',p[:,None,:]-proj,p[:,None,:]-proj)
        out[i:i+len(p)]=np.sqrt(d2.min(axis=1))
    return out

def stats(A,B,eta0=0):
    Ar,ea=resamp(A); Br,eb=resamp(B); da=d2poly(Ar,B); db=d2poly(Br,A)
    a=da[ea>=eta0]; b=db[eb>=eta0]
    return .5*(a.mean()+b.mean()),np.sqrt(.5*(np.mean(a*a)+np.mean(b*b))),max(a.max(),b.max())

prod_s=pd.read_csv(DATA/'raw/production_scalars.csv'); prod_i=pd.read_csv(DATA/'raw/production_interface.csv')
P0=poly(prod_i,.3); sf0=prod_s[np.isclose(prod_s.t,.3)].iloc[0]; H0=P0[:,1].max(); R0=P0[:,0].max()
cap=[]; profiles={}
for we in [1000,770,500,300]:
    d=DATA/f'raw/capillary/We{we}'
    s=pd.read_csv(next(d.glob('*_state.csv'))); de=pd.read_csv(next(d.glob('*_deformation.csv'))); it=pd.read_csv(next(d.glob('*_interface.csv')))
    sf=s[np.isclose(s.t,.3)].iloc[0]; df=de[np.isclose(de.t,.3)].iloc[0]; P=poly(it,.3)
    m,r,x=stats(P0,P,0); _,_,x10=stats(P0,P,.10)
    Ar,e=resamp(P0); Br,_=resamp(P); profiles[we]=(e,.5*(d2poly(Ar,P)+d2poly(Br,P0)))
    cap.append(dict(We=we,Vratio=sf.Vratio,delta_Vratio_pct=100*(sf.Vratio-sf0.Vratio)/sf0.Vratio,
        Req=sf.Req,delta_Req_pct=100*(sf.Req-sf0.Req)/sf0.Req,Zcent=sf.Zcent,delta_Zcent_pct=100*(sf.Zcent-sf0.Zcent)/sf0.Zcent,
        H=df.H,delta_H_pct=100*(df.H-H0)/H0,Rmax=df.Rmax,delta_Rmax_pct=100*(df.Rmax-R0)/R0,
        interface_mean=m,interface_rms=r,interface_max=x,interface_max_eta_ge_0p10=x10,gV=sf.gV,gCorr=sf.gCorr,corrRel=sf.corrRel))
pd.DataFrame(cap).to_csv(DATA/'capillary_publication_summary.csv',index=False)

r5s=pd.read_csv(DATA/'raw/outer_domain/outer_stretch_r5_zm2p5_z5_scalars.csv'); r6s=pd.read_csv(DATA/'raw/outer_domain/outer_stretch_r6_zm3_z6_scalars.csv')
r5d=pd.read_csv(DATA/'raw/outer_domain/outer_stretch_r5_zm2p5_z5_deformationMetrics.csv'); r6d=pd.read_csv(DATA/'raw/outer_domain/outer_stretch_r6_zm3_z6_deformationMetrics.csv')
r5i=pd.read_csv(DATA/'raw/outer_domain/outer_stretch_r5_zm2p5_z5_interfaceSnapshots.csv'); r6i=pd.read_csv(DATA/'raw/outer_domain/outer_stretch_r6_zm3_z6_interfaceSnapshots.csv')
out=[]
for t in [.1,.2,.3]:
    a=prod_s[np.isclose(prod_s.t,t)].iloc[0]; b=r5s[np.isclose(r5s.t,t)].iloc[0]; c=r6s[np.isclose(r6s.t,t)].iloc[0]
    d5=r5d[np.isclose(r5d.t,t)].iloc[0]; d6=r6d[np.isclose(r6d.t,t)].iloc[0]; p4=poly(prod_i,t)
    out.append(dict(t=t,R4_Vratio=a.Vratio,R5_Vratio=b.Vratio,R6_Vratio=c.Vratio,R4_to_R5_Vratio_pct=100*(b.Vratio-a.Vratio)/a.Vratio,R5_to_R6_Vratio_pct=100*(c.Vratio-b.Vratio)/b.Vratio,
        R4_Req=a.Req,R5_Req=b.Req,R6_Req=c.Req,R4_to_R5_Req_pct=100*(b.Req-a.Req)/a.Req,R5_to_R6_Req_pct=100*(c.Req-b.Req)/b.Req,
        R4_Zcent=a.Zcent,R5_Zcent=b.Zcent,R6_Zcent=c.Zcent,R4_to_R5_Zcent_pct=100*(b.Zcent-a.Zcent)/a.Zcent,R5_to_R6_Zcent_pct=100*(c.Zcent-b.Zcent)/b.Zcent,
        R4_H=p4[:,1].max(),R5_H=d5.H,R6_H=d6.H,R4_Rmax=p4[:,0].max(),R5_Rmax=d5.Rmax,R6_Rmax=d6.Rmax))
pd.DataFrame(out).to_csv(DATA/'outer_domain_sensitivity_summary.csv',index=False)

iface=[]
for name,A,B in [('R4-R5',prod_i,r5i),('R5-R6',r5i,r6i),('R4-R6',prod_i,r6i)]:
    for t in [.1,.2,.3]:
        m,r,x=stats(poly(A,t),poly(B,t),0); _,_,x10=stats(poly(A,t),poly(B,t),.10)
        iface.append(dict(pair=name,t=t,interface_mean=m,interface_rms=r,interface_max=x,interface_max_eta_ge_0p10=x10))
pd.DataFrame(iface).to_csv(DATA/'outer_domain_interface_summary.csv',index=False)

plt.figure(figsize=(7,4.5)); plt.plot(prod_s.t,prod_s.Vratio,label='R4 production'); plt.plot(r5s.t,r5s.Vratio,label='R5 stretched'); plt.plot(r6s.t,r6s.Vratio,label='R6 stretched'); plt.xlabel('t'); plt.ylabel('$V_v/V_{v0}$'); plt.title('Outer-domain sensitivity of vapor-volume trajectory'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'outer_domain_volume_trajectories.png',dpi=220); plt.close()
