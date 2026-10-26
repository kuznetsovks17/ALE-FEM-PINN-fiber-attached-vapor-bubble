"""Rebuild three manuscript figures from supplied numeric exports.

Run from any directory with Python, numpy, pandas and matplotlib installed.
No simulations or PINN training are performed. Original data are read only.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
FIG=ROOT/'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':9,
    'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
    'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.7,
})

def finish(fig,stem):
    fig.savefig(FIG/f'{stem}.pdf',bbox_inches='tight',pad_inches=.07)
    fig.savefig(FIG/f'{stem}.png',dpi=240,bbox_inches='tight',pad_inches=.07)
    plt.close(fig)

base=pd.read_csv(DATA/'raw/production_scalars.csv')
expanded=pd.read_csv(DATA/'raw/outer_domain/outer_stretch_r5_zm2p5_z5_scalars.csv')
largest=pd.read_csv(DATA/'raw/outer_domain/outer_stretch_r6_zm3_z6_scalars.csv')
fig,ax=plt.subplots(figsize=(5.2,3.05),layout='constrained')
for df,label,color,ls in [(base,'Baseline domain','#0072B2','-'),
                         (expanded,'Expanded domain','#D55E00','--'),
                         (largest,'Largest domain','#009E73','-.')]:
    # The initial value is exactly 1 by the definition of normalized volume.
    ax.plot(np.r_[0,df.t],np.r_[1,df.Vratio],label=label,color=color,ls=ls,lw=1.6)
ax.set(xlim=(0,.30),ylim=(.83,1.005),xlabel='Nondimensional time, $t$',ylabel=r'Normalized volume, $V_v/V_{v0}$')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.grid(alpha=.18,lw=.6);ax.legend(frameon=False,loc='lower left')
finish(fig,'outer_domain_volume')

fig,ax=plt.subplots(figsize=(5.2,2.05),layout='constrained')
ax.plot(base.t,100*base.gCorr.abs()/(base.gV.abs()+1e-10),color='#0072B2',lw=1.4)
ax.set(xlim=(0,.30),ylim=(0,17.2),xlabel=r'Time, $t$',ylabel=r'Compatibility correction [%]')
ax.set_xticks([0,.10,.20,.30]);ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_yticks([0,5,10,15]);ax.grid(alpha=.18,lw=.6)
finish(fig,'ale_compatibility')

audit=pd.read_csv(DATA/'cross_method'/'interface_distance_audit.csv')
df=audit[np.isclose(audit.t,.30)].sort_values('eta0')
fig,ax=plt.subplots(figsize=(3.15,3.10),layout='constrained')
for field,label,color,marker,ls in [
    ('symmetric_mean','Mean','#0072B2','o','-'),
    ('symmetric_rms','RMS','#D55E00','s','--'),
    ('symmetric_max','Maximum','#009E73','^','-.')]:
    ax.plot(100*df.eta0,1000*df[field],label=label,color=color,marker=marker,ms=3.5,ls=ls,lw=1.25)
ax.set(xlim=(-.5,15.5),ylim=(0,23.5),xlabel='Excluded meridional arc [%]',ylabel=r'Distance / $R_0$ ($\times10^{-3}$)')
ax.set_xticks([0,5,10,15]);ax.set_yticks([0,5,10,15,20]);ax.grid(alpha=.18,lw=.6)
ax.legend(frameon=False,loc='upper right')
finish(fig,'interface_distance_cutoff')
print('Rebuilt outer-domain, compatibility, and interface-distance figures.')
