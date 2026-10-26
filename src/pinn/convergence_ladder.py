#!/usr/bin/env python3
# PINN convergence ladder
# Exported from a cleaned Jupyter notebook; notebook outputs are not embedded.

# %% [notebook code cell 1]
import os
import sys
import re
import json
import math
import subprocess
from pathlib import Path
import numpy as np

WORKDIR = Path.cwd()
FINAL_REPORT = WORKDIR / 'point16_pinn_convergence_report.txt'
LEVELS_TO_RUN = ['coarse','medium','fine']  # keep all three for the formal Point-16 ladder
KEEP_LEVEL_REPORTS = False

print('Point 16 workdir:', WORKDIR)
print('Every level launches in a fresh Python interpreter; no checkpoint can leak between levels.')

# %% [notebook code cell 2]
LEVEL_SPECS = [{'name': 'coarse', 'n_time_slices': 17, 'n_liquid_per_time': 32, 'n_vapor_per_time': 160, 'n_interface_per_time': 24, 'n_vapor_t0_fixed': 160, 'n_liquid_t0_fixed': 160, 'n_gauge_spatial': 48, 'n_boundary_per_time': 4, 'n_axis_per_time': 4, 'n_volume_xi': 48, 'liquid_width': 48, 'liquid_depth': 3, 'vapor_width': 48, 'vapor_depth': 3, 'interface_width': 40, 'interface_depth': 3, 'interface_top_width': 24, 'ff_features': 12}, {'name': 'medium', 'n_time_slices': 33, 'n_liquid_per_time': 48, 'n_vapor_per_time': 320, 'n_interface_per_time': 32, 'n_vapor_t0_fixed': 256, 'n_liquid_t0_fixed': 256, 'n_gauge_spatial': 64, 'n_boundary_per_time': 6, 'n_axis_per_time': 6, 'n_volume_xi': 64, 'liquid_width': 64, 'liquid_depth': 4, 'vapor_width': 60, 'vapor_depth': 3, 'interface_width': 48, 'interface_depth': 4, 'interface_top_width': 32, 'ff_features': 16}, {'name': 'fine', 'n_time_slices': 49, 'n_liquid_per_time': 64, 'n_vapor_per_time': 480, 'n_interface_per_time': 48, 'n_vapor_t0_fixed': 384, 'n_liquid_t0_fixed': 384, 'n_gauge_spatial': 96, 'n_boundary_per_time': 8, 'n_axis_per_time': 8, 'n_volume_xi': 96, 'liquid_width': 80, 'liquid_depth': 5, 'vapor_width': 80, 'vapor_depth': 4, 'interface_width': 64, 'interface_depth': 5, 'interface_top_width': 40, 'ff_features': 20}]

BASE_SOURCE = 'import time\nimport numpy as np\nimport tensorflow as tf\nimport matplotlib.pyplot as plt\nfrom scipy.stats import qmc\nfrom numpy.polynomial.legendre import leggauss\n\nnp.random.seed(1234)\ntf.random.set_seed(1234)\n\nmse=lambda x: tf.reduce_mean(tf.square(x))\n\ndef rms_np(x):\n    x=np.asarray(x).reshape(-1)\n    return float(np.sqrt(np.mean(x*x))) if x.size else np.nan\n\nclass ProductionConfig:\n    def __init__(self):\n        # Same nondimensional production physics.\n        self.Re=500.0\n        self.Pi=1.25\n        self.pv0=-1.0\n        self.pinf=0.0\n        self.gamma_v=1.0\n        self.rho_v0=2.0e-2\n        self.mu_ratio=2.0e-2\n\n        # Production geometry.\n        self.af=300.0/686.0\n        self.rmin=0.0\n        self.rmax=4.0\n        self.zmin=-2.0\n        self.zmax=4.0\n        self.zc=float(np.sqrt(1.0-self.af*self.af))\n        self.theta_rim=float(np.arccos(-self.zc))\n\n        # Architecture gate only.\n        self.tmin=0.0\n        self.tmax=0.10\n\n        # Frozen-G3 neural conditioning.\n        self.axis_eps=2.0e-3\n        self.lr=2.0e-4\n        self.beta1=0.99\n        self.seed=1234\n        self.ff_features=16\n        self.liquid_width=64\n        self.liquid_depth=4\n        self.vapor_width=60\n        self.vapor_depth=3\n        self.interface_top_width=32\n        self.ff_scale=1.0\n        self.chi_scale=3.0\n        self.p_scale=4.0\n        self.u_scale=3.0          # vapor velocity scale, as in G3\n        self.q_scale=1.0\n\n        # Frozen-G3 collocation structure.\n        self.n_time_slices=33\n        self.n_liquid_per_time=48       # 33*48=1584 ~ frozen G3\'s 1536 liquid points\n        self.n_vapor_per_time=320       # nonzero slices only: 32*320=10240\n        self.n_interface_per_time=32\n        self.n_vapor_t0_fixed=256\n        self.n_liquid_t0_fixed=256      # necessary replacement for unavailable spherical p0 aid\n        self.n_gauge_spatial=64\n        self.n_boundary_per_time=6      # 33*6=198 ~ frozen G3\'s 200 boundary points\n        self.n_axis_per_time=6\n        self.refresh_every=500\n        self.interface_margin=0.02\n\n        # Nonspherical geometry quadrature / interface network (new geometry only).\n        self.n_volume_xi=64\n        self.interface_width=48\n        self.interface_depth=4\n        self.geom_r_scale=0.20\n        self.geom_z_scale=0.20\n\n        # Frozen G3 force-balance weighting.\n        self.lambda_normal=10.0\n\n        # Training budget from final G3.\n        self.epochs=22000\n\ncfg=ProductionConfig()\nprint(\'af=\',cfg.af,\'zc=\',cfg.zc,\'theta_rim=\',cfg.theta_rim)\nprint(\'gate=\',(cfg.tmin,cfg.tmax),\'Chebyshev nodes=\',cfg.n_time_slices)\nprint(\'vapor points/nonzero slice=\',cfg.n_vapor_per_time,\'epochs=\',cfg.epochs)\n\ndef chebyshev_lobatto_times(n,tmin,tmax):\n    n=int(n)\n    j=np.arange(n,dtype=np.float64)\n    tau=0.5*(1.0-np.cos(np.pi*j/(n-1)))\n    return (tmin+(tmax-tmin)*tau).astype(np.float32)\n\nTnodes_np=chebyshev_lobatto_times(cfg.n_time_slices,cfg.tmin,cfg.tmax)\nTnodes=tf.constant(Tnodes_np[:,None],tf.float32)\nTvapor_np=Tnodes_np[1:]\n\n# Gauss-Legendre quadrature on [0,1] for differentiable geometric moments.\nxg,wg=leggauss(cfg.n_volume_xi)\nVOL_XI=tf.constant(((xg+1.0)/2.0).astype(np.float32)[:,None])\nVOL_W=tf.constant((wg/2.0).astype(np.float32)[:,None])\n\nprint(\'first/last Chebyshev nodes:\',Tnodes_np[:4],Tnodes_np[-4:])\n\n# -----------------------------------------------------------------------------\n# Frozen-G3 neural conditioning blocks.\n# -----------------------------------------------------------------------------\nclass GaussianFourierBlock(tf.keras.layers.Layer):\n    def __init__(self,n_features=16,scale=1.0,name=None):\n        super().__init__(name=name); self.n_features=int(n_features); self.scale=float(scale)\n    def build(self,input_shape):\n        self.B=self.add_weight(name=\'B\',shape=(int(input_shape[-1]),self.n_features),\n            initializer=tf.keras.initializers.RandomNormal(stddev=self.scale),trainable=False)\n    def call(self,x):\n        a=2.0*np.pi*tf.matmul(x,self.B)\n        return tf.concat([x,tf.sin(a),tf.cos(a)],axis=1)\n\nclass NormalizedInput(tf.keras.layers.Layer):\n    """Frozen-G3 smooth physical-coordinate input for high-order liquid AD."""\n    def __init__(self,cfg,radial_even=False,name=None):\n        super().__init__(name=name); self.cfg=cfg; self.radial_even=bool(radial_even)\n    def call(self,X):\n        X=tf.cast(X,tf.float32); r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]\n        rn=(r-self.cfg.rmin)/(self.cfg.rmax-self.cfg.rmin)\n        zn=(z-self.cfg.zmin)/(self.cfg.zmax-self.cfg.zmin)\n        tn=(t-self.cfg.tmin)/(self.cfg.tmax-self.cfg.tmin)\n        if self.radial_even: rn=tf.square(rn)\n        return tf.concat([rn,zn,tn],axis=1)\n\nclass VolumeScaledFourierInput(tf.keras.layers.Layer):\n    """Frozen-G3 vapor features generalized to actual volume and centroid."""\n    def __init__(self,cfg,name=None):\n        super().__init__(name=name); self.cfg=cfg\n        self.ff=GaussianFourierBlock(cfg.ff_features,cfg.ff_scale)\n    def call(self,inputs):\n        X,Lv,Zc=inputs\n        X=tf.cast(X,tf.float32); Lv=tf.cast(Lv,tf.float32); Zc=tf.cast(Zc,tf.float32)\n        r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]\n        Lsafe=tf.maximum(Lv,tf.cast(1e-5,Lv.dtype))\n        rn=r/Lsafe\n        zn=(z-Zc)/Lsafe\n        tn=(t-self.cfg.tmin)/(self.cfg.tmax-self.cfg.tmin)\n        return self.ff(tf.concat([rn,zn,tn],axis=1))\n\nclass LiquidScalarNetwork(tf.keras.Model):\n    def __init__(self,cfg,output_scale,layers=4,dim=64,bounded=True,name=\'scalar\'):\n        super().__init__(name=name); self.output_scale=float(output_scale); self.bounded=bool(bounded)\n        self.inp=NormalizedInput(cfg,radial_even=True,name=name+\'_input\')\n        self.hidden=[tf.keras.layers.Dense(dim,activation=\'tanh\') for _ in range(layers)]\n        self.out=tf.keras.layers.Dense(1,kernel_initializer=\'zeros\',bias_initializer=\'zeros\')\n    def call(self,X):\n        y=self.inp(tf.cast(X,tf.float32))\n        for layer in self.hidden: y=layer(y)\n        y=self.out(y)\n        if self.bounded: y=tf.tanh(y)\n        return tf.cast(self.output_scale,tf.float32)*y\n\nclass VaporNet(tf.keras.Model):\n    """Exact frozen-G3 vapor architecture: Fourier input, 3x60 trunk, three 60-wide heads."""\n    def __init__(self,cfg):\n        super().__init__(); self.cfg=cfg\n        self.ffe=VolumeScaledFourierInput(cfg,name=\'vapor_ffe\')\n        self.common=[tf.keras.layers.Dense(cfg.vapor_width,activation=\'tanh\') for _ in range(cfg.vapor_depth)]\n        self.heads=[]\n        for name in [\'urv\',\'uzv\',\'qv\']:\n            h=tf.keras.layers.Dense(cfg.vapor_width,activation=\'tanh\',name=name+\'_h\')\n            o=tf.keras.layers.Dense(1,kernel_initializer=\'zeros\',bias_initializer=\'zeros\',name=name+\'_out\')\n            self.heads.append((name,h,o))\n    def call(self,X,Lv,Zc):\n        x=self.ffe((X,Lv,Zc))\n        for h in self.common: x=h(x)\n        out=[]\n        for name,h,o in self.heads:\n            y=o(h(x)); out.append(tf.tanh(y) if name!=\'qv\' else y)\n        return tf.concat(out,axis=1)\n\nclass SmoothMLP(tf.keras.Model):\n    """Only the nonspherical interface needs a new network absent from spherical G3."""\n    def __init__(self,out_dim,width,depth,name=None):\n        super().__init__(name=name)\n        self.hidden=[tf.keras.layers.Dense(width,activation=\'tanh\') for _ in range(depth)]\n        self.out=tf.keras.layers.Dense(out_dim,kernel_initializer=\'zeros\',bias_initializer=\'zeros\')\n    def call(self,x):\n        for h in self.hidden: x=h(x)\n        return self.out(x)\n\nclass FrozenG3FiberPINN(tf.keras.Model):\n    def __init__(self,cfg):\n        super().__init__(); self.cfg=cfg\n        self.chi=LiquidScalarNetwork(cfg,cfg.chi_scale,layers=cfg.liquid_depth,dim=cfg.liquid_width,bounded=True,name=\'chi_l\')\n        self.pressure=LiquidScalarNetwork(cfg,cfg.p_scale,layers=cfg.liquid_depth,dim=cfg.liquid_width,bounded=False,name=\'p_l\')\n        self.vapor=VaporNet(cfg)\n        self.iface_shape=SmoothMLP(2,cfg.interface_width,cfg.interface_depth,name=\'iface_shape\')\n        self.iface_top=SmoothMLP(1,cfg.interface_top_width,2,name=\'iface_top\')\n        self.V0=None\n\n    def tau(self,t):\n        return (tf.cast(t,tf.float32)-self.cfg.tmin)/(self.cfg.tmax-self.cfg.tmin)\n\n    def interface_features(self,xi,t):\n        return tf.concat([2.0*xi-1.0,2.0*self.tau(t)-1.0],axis=1)\n\n    def interface(self,xi,t):\n        xi=tf.cast(xi,tf.float32); t=tf.cast(t,tf.float32); tau=self.tau(t)\n        th=tf.cast(self.cfg.theta_rim,xi.dtype)*(1.0-xi)\n        r0=tf.sin(th)\n        z0=tf.cast(self.cfg.zc,xi.dtype)+tf.cos(th)\n        raw=tf.tanh(self.iface_shape(self.interface_features(xi,t)))\n        rr,zz=tf.split(raw,2,axis=1)\n        top=tf.tanh(self.iface_top(2.0*tau-1.0))\n\n        # Exact initial shape, exact pinned rim, exact axis intersection, X_t(xi,0)=0.\n        r=r0*tf.exp(tf.cast(self.cfg.geom_r_scale,r0.dtype)*tf.square(tau)*xi*rr)\n        Btop=2.0*xi-tf.square(xi)\n        Bint=xi*tf.square(1.0-xi)\n        z=z0+tf.cast(self.cfg.geom_z_scale,z0.dtype)*tf.square(tau)*(Btop*top+Bint*zz)\n        return r,z\n\n    def volume_centroid(self,t):\n        t=tf.cast(t,tf.float32); nt=tf.shape(t)[0]; nq=tf.shape(VOL_XI)[0]\n        xi=tf.tile(VOL_XI,[nt,1])\n        tt=tf.reshape(tf.repeat(t,repeats=nq,axis=0),[-1,1])\n        with tf.GradientTape() as tape:\n            tape.watch(xi)\n            r,z=self.interface(xi,tt)\n        zxi=tape.gradient(z,xi)\n        base=tf.square(r)*zxi\n        base=tf.reshape(base,[nt,nq]); zmat=tf.reshape(z,[nt,nq])\n        w=tf.reshape(VOL_W,[1,-1])\n        den=tf.reduce_sum(base*w,axis=1,keepdims=True)\n        znum=tf.reduce_sum(zmat*base*w,axis=1,keepdims=True)\n        V=tf.cast(np.pi,base.dtype)*den\n        Zc=znum/(den+1e-12)\n        return V,Zc\n\n    def vapor_characteristics(self,t):\n        V,Zc=self.volume_centroid(t)\n        ratio=V/tf.cast(self.V0,V.dtype)\n        Lv=tf.pow(tf.maximum(ratio,1e-10),1.0/3.0)  # frozen G3 L_v=(V/V0)^(1/3)\n        return V,ratio,Zc,Lv\n\n    def liquid_scalars(self,X):\n        return self.chi(X),self.pressure(X)\n\n    def liquid_state(self,X):\n        """Minimal nonspherical generalization of frozen G3 streamfunction liquid.\n\n        Frozen G3 used sink + tau^2*chi correction.  The sink is spherical-only and removed.\n        Because production initial acceleration is unknown, tau (not tau^2) multiplies the full\n        streamfunction velocity, preserving u(0)=0 while leaving u_t(0) learnable.\n        """\n        X=tf.cast(X,tf.float32); r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]\n        with tf.GradientTape(persistent=True) as tape:\n            tape.watch([r,z]); chi=self.chi(tf.concat([r,z,t],axis=1)); p=self.pressure(tf.concat([r,z,t],axis=1))\n        chi_r=tape.gradient(chi,r); chi_z=tape.gradient(chi,z); del tape\n        tau=self.tau(t)\n        ur=-tau*r*chi_z\n        uz=tau*(2.0*chi+r*chi_r)\n        return ur,uz,p,chi\n\n    def vapor_hat(self,X,Lv,Zc):\n        return self.vapor(tf.cast(X,tf.float32),Lv,Zc)\n\n    def vapor_state_scaled(self,X,Lv,Zc):\n        X=tf.cast(X,tf.float32); t=X[:,2:3]\n        uh_r,uh_z,qh=tf.split(self.vapor_hat(X,Lv,Zc),3,axis=1)\n        tau=self.tau(t)\n        return tau*self.cfg.u_scale*uh_r,tau*self.cfg.u_scale*uh_z,self.cfg.q_scale*qh\n\n    def vapor_state_auto(self,X):\n        X=tf.cast(X,tf.float32); t=X[:,2:3]\n        _,_,Zc,Lv=self.vapor_characteristics(t)\n        return self.vapor_state_scaled(X,Lv,Zc)\n\nmodel=FrozenG3FiberPINN(cfg)\n# Build interface/liquid, then establish the exact geometric initial volume.\n_ = model.interface(tf.constant([[0.5]],tf.float32),tf.constant([[0.0]],tf.float32))\n_ = model.liquid_state(tf.constant([[1.0,1.0,0.0]],tf.float32))\nV0tmp,_=model.volume_centroid(tf.constant([[0.0]],tf.float32))\nmodel.V0=tf.stop_gradient(V0tmp)\n# Build vapor after V0 exists.\n_ = model.vapor_state_auto(tf.constant([[0.2,0.5,0.0]],tf.float32))\nprint(\'initial geometric V0=\',float(model.V0.numpy()[0,0]))\nprint(\'trainable parameters=\',sum(int(np.prod(v.shape)) for v in model.trainable_variables))\nprint(\'liquid: frozen-G3 chi architecture + learned nonspherical p_l; no spherical sink/pressure\')\n\n@tf.function(reduce_retracing=True)\ndef interface_kinematics(xi,t):\n    xi=tf.cast(xi,tf.float32); t=tf.cast(t,tf.float32)\n    with tf.GradientTape(persistent=True) as tape:\n        tape.watch([xi,t]); r,z=model.interface(xi,t)\n    rxi=tape.gradient(r,xi); zxi=tape.gradient(z,xi)\n    rt=tape.gradient(r,t); zt=tape.gradient(z,t); del tape\n    ds=tf.sqrt(rxi*rxi+zxi*zxi+1e-12)\n    tr=rxi/ds; tz=zxi/ds\n    nr=zxi/ds; nz=-rxi/ds   # vapor -> liquid\n    return r,z,rxi,zxi,rt,zt,nr,nz,tr,tz\n\n@tf.function(reduce_retracing=True)\ndef thermo_time_state(t):\n    t=tf.cast(t,tf.float32)\n    with tf.GradientTape(persistent=True) as tape:\n        tape.watch(t)\n        V,ratio,Zc,Lv=model.vapor_characteristics(t)\n    Vt=tape.gradient(V,t)\n    Zct=tape.gradient(Zc,t)\n    Lvt=tape.gradient(Lv,t)\n    del tape\n    gV=Vt/tf.maximum(V,1e-10)\n    Pv=(cfg.Pi+cfg.pv0)*tf.pow(tf.maximum(ratio,1e-10),-cfg.gamma_v)-cfg.Pi\n    rho=cfg.rho_v0/tf.maximum(ratio,1e-10)\n    return dict(V=V,ratio=ratio,Vt=Vt,gV=gV,Pv=Pv,rho=rho,Zc=Zc,Lv=Lv,Zct=Zct,Lvt=Lvt)\n\n@tf.function(reduce_retracing=True)\ndef geometry_observables(t):\n    S=thermo_time_state(t)\n    Req=tf.pow(3.0*S[\'V\']/(4.0*np.pi),1.0/3.0)\n    return dict(**S,Req=Req)\n\nS0=geometry_observables(tf.constant([[0.0]],tf.float32))\nprint(\'initial Req=\',float(S0[\'Req\'].numpy()[0,0]),\'Zcent=\',float(S0[\'Zc\'].numpy()[0,0]))\n\nclass ProductionSampler:\n    def __init__(self,cfg): self.cfg=cfg; self.seed=cfg.seed\n    def sobol(self,n,d):\n        m=int(np.ceil(np.log2(max(2,int(n)))))\n        s=qmc.Sobol(d=d,scramble=True,seed=self.seed); self.seed+=1\n        return s.random_base2(m)[:int(n)]\n\n    def interface_curve_np(self,t,n=501):\n        xi=np.linspace(0.0,1.0,n,dtype=np.float32)[:,None]\n        tt=np.full_like(xi,float(t))\n        r,z=model.interface(tf.constant(xi),tf.constant(tt))\n        r=r.numpy().reshape(-1); z=z.numpy().reshape(-1)\n        order=np.argsort(z)\n        return r[order],z[order]\n\n    def liquid_at_time(self,t,n,margin=None):\n        c=self.cfg; margin=c.interface_margin if margin is None else float(margin)\n        ri,zi=self.interface_curve_np(t); ztop=float(zi[-1]); out=[]; need=int(n)\n        while need>0:\n            U=self.sobol(max(512,4*need),2)\n            r=c.axis_eps+(c.rmax-c.axis_eps)*U[:,0]\n            z=c.zmin+(c.zmax-c.zmin)*U[:,1]\n            solid=(z<0.0)&(r<c.af+margin)\n            vap=np.zeros_like(solid)\n            band=(z>=0.0)&(z<=ztop)\n            if np.any(band):\n                rb=np.interp(z[band],zi,ri)\n                vap[band]=r[band]<(rb+margin)\n            keep=~(solid|vap)\n            X=np.c_[r[keep],z[keep],np.full(np.count_nonzero(keep),t)]\n            take=min(need,len(X))\n            if take:\n                out.append(X[:take]); need-=take\n        return np.vstack(out).astype(np.float32)\n\n    def liquid_cloud(self):\n        return np.vstack([self.liquid_at_time(float(t),self.cfg.n_liquid_per_time) for t in Tnodes_np]).astype(np.float32)\n\n    def vapor_reference_cloud(self):\n        rows=[]; idx=[]\n        # Exactly as frozen G3: full vapor PDE only on the 32 nonzero Chebyshev slices.\n        for j,t in enumerate(Tnodes_np[1:],start=1):\n            U=self.sobol(self.cfg.n_vapor_per_time,2)\n            eta=0.01+0.97*U[:,0:1]; xi=0.01+0.98*U[:,1:2]\n            rows.append(np.c_[eta,xi,np.full(len(U),t)])\n            idx.append(np.full(len(U),j,np.int32))\n        return np.vstack(rows).astype(np.float32),np.concatenate(idx)\n\n    def vapor_t0_reference(self):\n        U=self.sobol(self.cfg.n_vapor_t0_fixed,2)\n        return np.c_[0.01+0.97*U[:,0],0.01+0.98*U[:,1],np.zeros(len(U))].astype(np.float32)\n\n    def interface_cloud(self):\n        rows=[]; idx=[]\n        for j,t in enumerate(Tnodes_np):\n            U=self.sobol(self.cfg.n_interface_per_time,1)\n            xi=0.005+0.99*U[:,0:1]\n            rows.append(np.c_[xi,np.full(len(U),t)])\n            idx.append(np.full(len(U),j,np.int32))\n        return np.vstack(rows).astype(np.float32),np.concatenate(idx)\n\n    def boundary_cloud(self,kind):\n        rows=[]\n        for t in Tnodes_np:\n            U=self.sobol(self.cfg.n_boundary_per_time,1)[:,0]\n            if kind==\'fiber\':\n                r=np.full_like(U,self.cfg.af); z=self.cfg.zmin+(0.0-self.cfg.zmin)*U\n            elif kind==\'tip\':\n                r=self.cfg.af*U; z=np.zeros_like(U)\n            elif kind==\'farr\':\n                r=np.full_like(U,self.cfg.rmax); z=self.cfg.zmin+(self.cfg.zmax-self.cfg.zmin)*U\n            elif kind==\'farz\':\n                r=self.cfg.rmax*U; z=np.full_like(U,self.cfg.zmax)\n            elif kind==\'bottom\':\n                r=self.cfg.af+(self.cfg.rmax-self.cfg.af)*U; z=np.full_like(U,self.cfg.zmin)\n            else: raise ValueError(kind)\n            rows.append(np.c_[r,z,np.full_like(U,t)])\n        return np.vstack(rows).astype(np.float32)\n\n    def vapor_axis_reference(self):\n        rows=[]; idx=[]\n        for j,t in enumerate(Tnodes_np):\n            U=self.sobol(self.cfg.n_axis_per_time,1)[:,0]\n            rows.append(np.c_[0.01+0.98*U,np.full_like(U,t)])  # s,z fraction + time\n            idx.append(np.full(len(U),j,np.int32))\n        return np.vstack(rows).astype(np.float32),np.concatenate(idx)\n\nsampler=ProductionSampler(cfg)\nYvap_np,VapTimeIdx_np=sampler.vapor_reference_cloud()\nYv0_np=sampler.vapor_t0_reference()\nGiface_np,IfaceTimeIdx_np=sampler.interface_cloud()\nYaxis_np,AxisTimeIdx_np=sampler.vapor_axis_reference()\n\nYvap=tf.constant(Yvap_np); VapTimeIdx=tf.constant(VapTimeIdx_np,tf.int32)\nYv0=tf.constant(Yv0_np)\nGiface=tf.constant(Giface_np); IfaceTimeIdx=tf.constant(IfaceTimeIdx_np,tf.int32)\nYaxis=tf.constant(Yaxis_np); AxisTimeIdx=tf.constant(AxisTimeIdx_np,tf.int32)\n\nXfiber=tf.constant(sampler.boundary_cloud(\'fiber\'))\nXtip=tf.constant(sampler.boundary_cloud(\'tip\'))\nXfarr=tf.constant(sampler.boundary_cloud(\'farr\'))\nXfarz=tf.constant(sampler.boundary_cloud(\'farz\'))\nXbottom=tf.constant(sampler.boundary_cloud(\'bottom\'))\n# boundary_cloud is ordered by fixed Chebyshev time slice.\nTipTimeIdx=tf.constant(np.repeat(np.arange(cfg.n_time_slices,dtype=np.int32),cfg.n_boundary_per_time),tf.int32)\n\nXliquid=tf.Variable(sampler.liquid_cloud(),trainable=False)\nXliquid0=tf.Variable(sampler.liquid_at_time(0.0,cfg.n_liquid_t0_fixed,margin=0.01),trainable=False)\n\n# Fixed 64-point reference cloud for the gauge at every one of the 33 time nodes.\nUg=sampler.sobol(cfg.n_gauge_spatial,2).astype(np.float32)\nYgauge=tf.constant(np.c_[0.01+0.97*Ug[:,0],0.01+0.98*Ug[:,1]].astype(np.float32))\n\n# Frozen-style separate initial interface traction set.\nU0=sampler.sobol(cfg.n_interface_per_time,1).astype(np.float32)\nG0=tf.constant(np.c_[0.005+0.99*U0[:,0],np.zeros(len(U0))].astype(np.float32))\n\n\ndef refresh_spatial_collocation(verbose=True):\n    Xliquid.assign(sampler.liquid_cloud())\n    Xliquid0.assign(sampler.liquid_at_time(0.0,cfg.n_liquid_t0_fixed,margin=0.01))\n    if verbose: print(\'refreshed liquid clouds:\',Xliquid.shape,Xliquid0.shape)\n\nprint(\'cloud sizes: liquid=\',Xliquid.shape,\'vapor(nonzero)=\',Yvap.shape,\'interface=\',Giface.shape)\nprint(\'t0 liquid=\',Xliquid0.shape,\'t0 vapor=\',Yv0.shape,\'boundary each=\',Xfiber.shape)\n\n@tf.function(reduce_retracing=True)\ndef map_vapor_reference(Y):\n    eta,xi,t=Y[:,0:1],Y[:,1:2],Y[:,2:3]\n    rG,zG=model.interface(xi,t)\n    return tf.concat([tf.stop_gradient(eta*rG),tf.stop_gradient(zG),tf.stop_gradient(t)],axis=1)\n\n@tf.function(reduce_retracing=True)\ndef map_vapor_t0(Y):\n    return map_vapor_reference(Y)\n\n@tf.function(reduce_retracing=True)\ndef qbar_from_state(tnodes,Lv_nodes,Zc_nodes):\n    """Frozen-G3 gauge mean on 64 reference points at every supplied time node."""\n    tnodes=tf.cast(tnodes,tf.float32); nt=tf.shape(tnodes)[0]; ng=tf.shape(Ygauge)[0]\n    eta0=Ygauge[:,0:1]; xi0=Ygauge[:,1:2]\n    eta=tf.tile(eta0,[nt,1]); xi=tf.tile(xi0,[nt,1])\n    tt=tf.reshape(tf.repeat(tnodes,repeats=ng,axis=0),[-1,1])\n    with tf.GradientTape() as tape:\n        tape.watch(xi); rG,zG=model.interface(xi,tt)\n    zxi=tape.gradient(zG,xi)\n    r=tf.stop_gradient(eta*rG); z=tf.stop_gradient(zG)\n    X=tf.concat([r,z,tt],axis=1)\n    Lv=tf.reshape(tf.repeat(Lv_nodes,repeats=ng,axis=0),[-1,1])\n    Zc=tf.reshape(tf.repeat(Zc_nodes,repeats=ng,axis=0),[-1,1])\n    _,_,q=model.vapor_state_scaled(X,Lv,Zc)\n    w=tf.stop_gradient(eta*tf.square(rG)*zxi)\n    q=tf.reshape(q,[nt,ng]); w=tf.reshape(w,[nt,ng])\n    return tf.reduce_sum(q*w,axis=1)/(tf.reduce_sum(w,axis=1)+1e-12)\n\n@tf.function(reduce_retracing=True)\ndef qbar_at_nodes(tnodes):\n    """Diagnostic wrapper; training reuses its already-computed geometry state."""\n    S=thermo_time_state(tnodes)\n    return qbar_from_state(tnodes,S[\'Lv\'],S[\'Zc\'])\n\n@tf.function(reduce_retracing=True)\ndef liquid_terms(X):\n    """Frozen-G3 third-order liquid AD with learned nonspherical pressure."""\n    r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]\n    with tf.GradientTape(persistent=True) as tape3:\n        tape3.watch([r,z])\n        with tf.GradientTape(persistent=True) as tape2:\n            tape2.watch([r,z,t])\n            with tf.GradientTape(persistent=True) as tape1:\n                tape1.watch([r,z])\n                inp=tf.concat([r,z,t],axis=1)\n                chi=model.chi(inp); p=model.pressure(inp); tau=model.tau(t)\n            chi_r=tape1.gradient(chi,r); chi_z=tape1.gradient(chi,z); del tape1\n            ur=-tau*r*chi_z\n            uz=tau*(2.0*chi+r*chi_r)\n        ur_r=tape2.gradient(ur,r); ur_z=tape2.gradient(ur,z); ur_t=tape2.gradient(ur,t)\n        uz_r=tape2.gradient(uz,r); uz_z=tape2.gradient(uz,z); uz_t=tape2.gradient(uz,t)\n        p_r=tape2.gradient(p,r); p_z=tape2.gradient(p,z); del tape2\n    ur_rr=tape3.gradient(ur_r,r); ur_zz=tape3.gradient(ur_z,z)\n    uz_rr=tape3.gradient(uz_r,r); uz_zz=tape3.gradient(uz_z,z); del tape3\n    rs=tf.maximum(r,tf.cast(cfg.axis_eps,r.dtype))\n    div=ur_r+ur/rs+uz_z\n    lapr=ur_rr+ur_r/rs-ur/(rs*rs)+ur_zz\n    lapz=uz_rr+uz_r/rs+uz_zz\n    mr=ur_t+ur*ur_r+uz*ur_z+p_r-lapr/cfg.Re\n    mz=uz_t+ur*uz_r+uz*uz_z+p_z-lapz/cfg.Re\n    return dict(mr=mr,mz=mz,div=div,ur=ur,uz=uz,p=p)\n\n@tf.function(reduce_retracing=True)\ndef liquid_t0_momentum_terms(X):\n    """Necessary nonspherical replacement for G3\'s analytical initial-pressure aid.\n\n    At t=0, u=0 exactly, so convection and viscosity vanish.  With u=tau*U_chi,\n    u_t(0)=U_chi/tmax.  Enforce u_t + grad(p)=0 directly.\n    """\n    r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]\n    with tf.GradientTape(persistent=True) as tape:\n        tape.watch([r,z]); inp=tf.concat([r,z,t],axis=1); chi=model.chi(inp); p=model.pressure(inp)\n    chi_r=tape.gradient(chi,r); chi_z=tape.gradient(chi,z)\n    p_r=tape.gradient(p,r); p_z=tape.gradient(p,z); del tape\n    tau_t=tf.cast(1.0/(cfg.tmax-cfg.tmin),r.dtype)\n    ur_t=-tau_t*r*chi_z\n    uz_t=tau_t*(2.0*chi+r*chi_r)\n    return ur_t+p_r,uz_t+p_z\n\n@tf.function(reduce_retracing=True)\ndef vapor_terms(X,rho,gV,Lv,Zc,Lvt,Zct):\n    """Frozen-G3 primitive vapor PDE with generalized volume/centroid feature chain rule."""\n    r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]\n    with tf.GradientTape(persistent=True) as tape3:\n        tape3.watch([r,z])\n        with tf.GradientTape(persistent=True) as tape2:\n            tape2.watch([r,z,t,Lv,Zc])\n            ur,uz,q=model.vapor_state_scaled(tf.concat([r,z,t],axis=1),Lv,Zc)\n        ur_r=tape2.gradient(ur,r); ur_z=tape2.gradient(ur,z)\n        uz_r=tape2.gradient(uz,r); uz_z=tape2.gradient(uz,z)\n        q_r=tape2.gradient(q,r); q_z=tape2.gradient(q,z)\n        ur_t0=tape2.gradient(ur,t); uz_t0=tape2.gradient(uz,t)\n        ur_L=tape2.gradient(ur,Lv); uz_L=tape2.gradient(uz,Lv)\n        ur_Z=tape2.gradient(ur,Zc); uz_Z=tape2.gradient(uz,Zc)\n        del tape2\n        ur_t=ur_t0+ur_L*Lvt+ur_Z*Zct\n        uz_t=uz_t0+uz_L*Lvt+uz_Z*Zct\n    ur_rr=tape3.gradient(ur_r,r); ur_zz=tape3.gradient(ur_z,z)\n    uz_rr=tape3.gradient(uz_r,r); uz_zz=tape3.gradient(uz_z,z); del tape3\n    rs=tf.maximum(r,tf.cast(cfg.axis_eps,r.dtype))\n    div=ur_r+ur/rs+uz_z\n    cont=div-gV\n    lapr=ur_rr+ur_r/rs-ur/(rs*rs)+ur_zz\n    lapz=uz_rr+uz_r/rs+uz_zz\n    fac=(cfg.mu_ratio/cfg.Re)/rho\n    mr=ur_t+ur*ur_r+uz*ur_z+q_r-fac*lapr\n    mz=uz_t+ur*uz_r+uz*uz_z+q_z-fac*lapz\n    return dict(cont=cont,mr=mr,mz=mz,ur=ur,uz=uz,q=q)\n\n@tf.function(reduce_retracing=True)\ndef vapor_t0_momentum_terms(X,Lv,Zc):\n    """Frozen-G3 cheap t=0 vapor residual: u_v,t + grad(q_v)=0."""\n    r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]\n    with tf.GradientTape(persistent=True) as tape:\n        tape.watch([r,z]); raw=model.vapor_hat(tf.concat([r,z,t],axis=1),Lv,Zc); uh_r,uh_z,qh=tf.split(raw,3,axis=1)\n        q=cfg.q_scale*qh\n    q_r=tape.gradient(q,r); q_z=tape.gradient(q,z); del tape\n    tau_t=tf.cast(1.0/(cfg.tmax-cfg.tmin),r.dtype)\n    return cfg.u_scale*tau_t*uh_r+q_r,cfg.u_scale*tau_t*uh_z+q_z\n\n@tf.function(reduce_retracing=True)\ndef interface_residuals(G,time_idx,S):\n    xi,t=G[:,0:1],G[:,1:2]\n    r,z,rxi,zxi,rt,zt,nr,nz,tr,tz=interface_kinematics(xi,t)\n    Lv=tf.gather(S[\'Lv\'],time_idx); Zc=tf.gather(S[\'Zc\'],time_idx)\n    Pv=tf.gather(S[\'Pv\'],time_idx); rho=tf.gather(S[\'rho\'],time_idx)\n    with tf.GradientTape(persistent=True) as tape:\n        tape.watch([r,z]); X=tf.concat([r,z,t],axis=1)\n        ulr,ulz,p,_=model.liquid_state(X)\n        uvr,uvz,q=model.vapor_state_scaled(X,Lv,Zc)\n    ulr_r=tape.gradient(ulr,r); ulr_z=tape.gradient(ulr,z)\n    ulz_r=tape.gradient(ulz,r); ulz_z=tape.gradient(ulz,z)\n    uvr_r=tape.gradient(uvr,r); uvr_z=tape.gradient(uvr,z)\n    uvz_r=tape.gradient(uvz,r); uvz_z=tape.gradient(uvz,z); del tape\n\n    pvtot=Pv+rho*q\n    muL=1.0/cfg.Re; muV=cfg.mu_ratio/cfg.Re\n    srrL=-p+2.0*muL*ulr_r; szzL=-p+2.0*muL*ulz_z; srzL=muL*(ulr_z+ulz_r)\n    srrV=-pvtot+2.0*muV*uvr_r; szzV=-pvtot+2.0*muV*uvz_z; srzV=muV*(uvr_z+uvz_r)\n    tLr=srrL*nr+srzL*nz; tLz=srzL*nr+szzL*nz\n    tVr=srrV*nr+srzV*nz; tVz=srzV*nr+szzV*nz\n    dtr=tLr-tVr; dtz=tLz-tVz\n    normal=dtr*nr+dtz*nz\n    tangential=dtr*tr+dtz*tz\n\n    du_r=ulr-uvr; du_z=ulz-uvz\n    kin=(rt-ulr)*nr+(zt-ulz)*nz\n    tangauge=(rt-ulr)*tr+(zt-ulz)*tz\n    return dict(du_r=du_r,du_z=du_z,kin=kin,tangauge=tangauge,\n                normal=normal,tangential=tangential,p=p,pvtot=pvtot,q=q)\n\n@tf.function(reduce_retracing=True)\ndef weighted_interface_loss(G,time_idx,S):\n    I=interface_residuals(G,time_idx,S)\n    comps=(mse(I[\'du_r\']),mse(I[\'du_z\']),mse(I[\'kin\']),mse(I[\'tangauge\']),mse(I[\'normal\']),mse(I[\'tangential\']))\n    Ji=comps[0]+comps[1]+comps[2]+comps[3]+cfg.lambda_normal*comps[4]+comps[5]\n    return (Ji,)+comps\n\n@tf.function(reduce_retracing=True)\ndef initial_interface_traction_loss(S):\n    idx=tf.zeros((tf.shape(G0)[0],),tf.int32)\n    I=interface_residuals(G0,idx,S)\n    return cfg.lambda_normal*mse(I[\'normal\'])+mse(I[\'tangential\'])\n\n@tf.function(reduce_retracing=True)\ndef liquid_boundary_traction(X,nr,nz):\n    r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]; nr=tf.cast(nr,r.dtype); nz=tf.cast(nz,r.dtype)\n    with tf.GradientTape(persistent=True) as tape:\n        tape.watch([r,z]); ur,uz,p,_=model.liquid_state(tf.concat([r,z,t],axis=1))\n    ur_r=tape.gradient(ur,r); ur_z=tape.gradient(ur,z); uz_r=tape.gradient(uz,r); uz_z=tape.gradient(uz,z); del tape\n    mu=1.0/cfg.Re\n    srr=-p+2.0*mu*ur_r; szz=-p+2.0*mu*uz_z; srz=mu*(ur_z+uz_r)\n    return srr*nr+srz*nz+cfg.pinf*nr, srz*nr+szz*nz+cfg.pinf*nz\n\n@tf.function(reduce_retracing=True)\ndef solid_boundary_losses(S):\n    ulr,ulz,_,_=model.liquid_state(Xfiber)\n    Lv=tf.gather(S[\'Lv\'],TipTimeIdx); Zc=tf.gather(S[\'Zc\'],TipTimeIdx)\n    uvr,uvz,_=model.vapor_state_scaled(Xtip,Lv,Zc)\n    return mse(ulr)+mse(ulz),mse(uvr)+mse(uvz)\n\n@tf.function(reduce_retracing=True)\ndef far_traction_loss():\n    ar,az=liquid_boundary_traction(Xfarr,1.0,0.0)\n    br,bz=liquid_boundary_traction(Xfarz,0.0,1.0)\n    cr,cz=liquid_boundary_traction(Xbottom,0.0,-1.0)\n    return (mse(ar)+mse(az)+mse(br)+mse(bz)+mse(cr)+mse(cz))/3.0\n\n@tf.function(reduce_retracing=True)\ndef vapor_axis_loss(S):\n    s,t=Yaxis[:,0:1],Yaxis[:,1:2]\n    idx=AxisTimeIdx\n    xi1=tf.ones_like(t)\n    _,ztop=model.interface(xi1,t)\n    r=tf.zeros_like(t); z=tf.stop_gradient(s*ztop)\n    Lv=tf.gather(S[\'Lv\'],idx); Zc=tf.gather(S[\'Zc\'],idx)\n    with tf.GradientTape(persistent=True) as tape:\n        tape.watch(r); ur,uz,q=model.vapor_state_scaled(tf.concat([r,z,t],axis=1),Lv,Zc)\n    uz_r=tape.gradient(uz,r); q_r=tape.gradient(q,r); del tape\n    return mse(ur)+mse(uz_r)+mse(q_r)\n\n@tf.function(reduce_retracing=True)\ndef geometry_regularization():\n    nt=tf.shape(Tnodes)[0]; nq=tf.shape(VOL_XI)[0]\n    xi=tf.tile(VOL_XI,[nt,1]); tt=tf.reshape(tf.repeat(Tnodes,repeats=nq,axis=0),[-1,1])\n    with tf.GradientTape() as tape:\n        tape.watch(xi); r,z=model.interface(xi,tt)\n    zxi=tape.gradient(z,xi)\n    return mse(tf.nn.relu(1.0e-4-zxi))\n\noptimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr,beta_1=cfg.beta1)\n\n@tf.function(reduce_retracing=True)\ndef train_step():\n    Xv=map_vapor_reference(Yvap)\n    Xv0=map_vapor_t0(Yv0)\n    with tf.GradientTape() as tape:\n        S=thermo_time_state(Tnodes)\n\n        # Frozen-G3 liquid / vapor PDE structure plus necessary t=0 liquid substitute.\n        Tl=liquid_terms(Xliquid)\n        mlr0,mlz0=liquid_t0_momentum_terms(Xliquid0)\n\n        rhoV=tf.gather(S[\'rho\'],VapTimeIdx); gVV=tf.gather(S[\'gV\'],VapTimeIdx)\n        LvV=tf.gather(S[\'Lv\'],VapTimeIdx); ZcV=tf.gather(S[\'Zc\'],VapTimeIdx)\n        LvtV=tf.gather(S[\'Lvt\'],VapTimeIdx); ZctV=tf.gather(S[\'Zct\'],VapTimeIdx)\n        Tv=vapor_terms(Xv,rhoV,gVV,LvV,ZcV,LvtV,ZctV)\n\n        Lv0=tf.repeat(S[\'Lv\'][0:1],repeats=tf.shape(Xv0)[0],axis=0)\n        Zc0=tf.repeat(S[\'Zc\'][0:1],repeats=tf.shape(Xv0)[0],axis=0)\n        mvr0,mvz0=vapor_t0_momentum_terms(Xv0,Lv0,Zc0)\n\n        Jml=mse(Tl[\'mr\'])+mse(Tl[\'mz\'])\n        Jld=mse(Tl[\'div\'])  # diagnostic only; should be near hard-constraint floor\n        Jml0=mse(mlr0)+mse(mlz0)\n        Jcv=mse(Tv[\'cont\'])\n        Jmv=mse(Tv[\'mr\'])+mse(Tv[\'mz\'])\n        Jmv0=mse(mvr0)+mse(mvz0)\n\n        Ji,Jdu_r,Jdu_z,Jkin,JtanGauge,Jnormal,Jtang=weighted_interface_loss(Giface,IfaceTimeIdx,S)\n        Ji0=initial_interface_traction_loss(S)\n        Jfiber,Jtip=solid_boundary_losses(S)\n        Jfar=far_traction_loss()\n        qbars=qbar_from_state(Tnodes,S[\'Lv\'],S[\'Zc\'])\n        Jq=mse(qbars)\n        Jaxis=vapor_axis_loss(S)\n        Jgeom=geometry_regularization()\n\n        # Preserve frozen G3 additive loss philosophy; lambda_normal is internal to Ji/Ji0.\n        loss=(Jml+Jml0+Jcv+Jmv+Jmv0+Ji+Ji0+Jfiber+Jtip+Jfar+Jq+Jaxis+Jgeom)\n\n    grads=tape.gradient(loss,model.trainable_variables)\n    optimizer.apply_gradients([(g,v) for g,v in zip(grads,model.trainable_variables) if g is not None])\n    return (loss,Jml,Jld,Jml0,Jcv,Jmv,Jmv0,Ji,Ji0,Jfiber,Jtip,Jfar,Jq,Jaxis,Jgeom,\n            Jdu_r,Jdu_z,Jkin,JtanGauge,Jnormal,Jtang,tf.reduce_max(tf.abs(qbars)),S[\'ratio\'][-1,0])\n\n\ndef fit(epochs=None,refresh_every=None):\n    epochs=cfg.epochs if epochs is None else int(epochs)\n    refresh_every=cfg.refresh_every if refresh_every is None else int(refresh_every)\n    names=[\'loss\',\'Jml\',\'JldDiag\',\'Jml0\',\'Jcv\',\'Jmv\',\'Jmv0\',\'Ji(w)\',\'Ji0\',\'Jfiber\',\'Jtip\',\'Jfar\',\'Jq\',\'Jaxis\',\'Jgeom\',\n           \'Jdu_r\',\'Jdu_z\',\'Jkin\',\'JtanGauge\',\'Jnormal\',\'Jtang\',\'max|qbar|\',\'Vratio_end\']\n    print(\' \'.join(names))\n    history=[]; start=time.time(); mark=start; prev=-1\n    for it in range(epochs):\n        if refresh_every>0 and it>0 and it%refresh_every==0:\n            refresh_spatial_collocation(verbose=False)\n            print(f\'--- spatial refresh at {it}; {cfg.n_time_slices} Chebyshev nodes fixed; vapor points/slice={cfg.n_vapor_per_time} ---\')\n        vals=train_step()\n        if it%250==0 or it==epochs-1:\n            now=time.time(); nblock=max(1,it-prev); row=[float(v.numpy()) for v in vals]\n            history.append((it,row,now-start))\n            print(it,*[f\'{v:.3e}\' for v in row],f\'step={(now-mark)/nblock:.4f}s\',f\'elapsed={now-start:.1f}s\')\n            mark=now; prev=it\n    return names,history\n\nprint(\'STARTING FROM SCRATCH -- no checkpoint is loaded\')\ntrain_names,train_history=fit()\n\n# -----------------------------------------------------------------------------\n# Scalar geometry / thermodynamics diagnostics.\n# -----------------------------------------------------------------------------\ndef interface_snapshot(t,n=401):\n    xi=np.linspace(0.0,1.0,n,dtype=np.float32)[:,None]\n    tt=np.full_like(xi,float(t))\n    r,z=model.interface(tf.constant(xi),tf.constant(tt))\n    return xi.reshape(-1),r.numpy().reshape(-1),z.numpy().reshape(-1)\n\ndef qbar_single_time(t):\n    return float(qbar_at_nodes(tf.constant([[float(t)]],tf.float32)).numpy()[0])\n\ndef scalar_diagnostics(times=(0.0,0.0125,0.025,0.05,0.075,0.1)):\n    print(\'\\nSCALAR GEOMETRY / THERMODYNAMICS\')\n    rows=[]\n    xi0,r0,z0=interface_snapshot(0.0,401)\n    for tv in times:\n        tt=tf.constant([[tv]],tf.float32); O=geometry_observables(tt)\n        vals={k:float(v.numpy()[0,0]) for k,v in O.items()}\n        xi,r,z=interface_snapshot(tv,401)\n        disp=float(np.max(np.sqrt((r-r0)**2+(z-z0)**2)))\n        row=dict(t=float(tv),**vals,rmaxG=float(np.max(r)),ztop=float(z[-1]),qbar=qbar_single_time(tv),maxDisp=disp)\n        rows.append(row)\n        print(f"t={tv:7.4f} V/V0={vals[\'ratio\']:.8f} Req={vals[\'Req\']:.8f} Zc={vals[\'Zc\']:.8f} "\n              f"Pv={vals[\'Pv\']:+.8f} rho={vals[\'rho\']:.8f} gV={vals[\'gV\']:+.6e} "\n              f"rmaxG={row[\'rmaxG\']:.6f} ztop={row[\'ztop\']:.6f} maxDisp={disp:.3e} qbar={row[\'qbar\']:+.3e}")\n    return rows\n\nscalar_rows=scalar_diagnostics()\n\n# -----------------------------------------------------------------------------\n# Fresh-point validation, independent of training clouds.\n# -----------------------------------------------------------------------------\nrng_val=np.random.default_rng(20260831)\n\ndef fresh_liquid_points(t,n=2500):\n    return sampler.liquid_at_time(float(t),int(n),margin=0.01)\n\ndef fresh_vapor_reference(t,n=2500):\n    U=rng_val.random((int(n),2))\n    return np.c_[0.005+0.985*U[:,0],0.005+0.99*U[:,1],np.full(int(n),float(t))].astype(np.float32)\n\ndef fresh_interface(t,n=1200):\n    xi=np.linspace(0.002,0.998,int(n),dtype=np.float32)\n    return np.c_[xi,np.full(int(n),float(t))].astype(np.float32)\n\ndef validate_time(t):\n    # Liquid.\n    XL=tf.constant(fresh_liquid_points(t)); L=liquid_terms(XL)\n\n    # Vapor at one scalar time: compute geometry scale/time derivatives once, then tile.\n    Y=tf.constant(fresh_vapor_reference(t)); XV=map_vapor_reference(Y)\n    st=thermo_time_state(tf.constant([[float(t)]],tf.float32)); nv=tf.shape(XV)[0]\n    rho=tf.repeat(st[\'rho\'],nv,axis=0); gV=tf.repeat(st[\'gV\'],nv,axis=0)\n    Lv=tf.repeat(st[\'Lv\'],nv,axis=0); Zc=tf.repeat(st[\'Zc\'],nv,axis=0)\n    Lvt=tf.repeat(st[\'Lvt\'],nv,axis=0); Zct=tf.repeat(st[\'Zct\'],nv,axis=0)\n    V=vapor_terms(XV,rho,gV,Lv,Zc,Lvt,Zct)\n\n    # Interface.\n    G=tf.constant(fresh_interface(t)); nt=tf.shape(G)[0]\n    Sone={k:tf.repeat(v,nt,axis=0) if k not in [\'V\',\'ratio\',\'Vt\',\'gV\'] else v for k,v in st.items()}\n    # interface_residuals expects node arrays + integer indices; use a one-node dictionary.\n    Sif={k:v for k,v in st.items()}\n    idx=tf.zeros((tf.shape(G)[0],),tf.int32)\n    I=interface_residuals(G,idx,Sif)\n\n    row={\n        \'t\':float(t),\'liq_div\':rms_np(L[\'div\'].numpy()),\'liq_mom_r\':rms_np(L[\'mr\'].numpy()),\'liq_mom_z\':rms_np(L[\'mz\'].numpy()),\n        \'vap_cont\':rms_np(V[\'cont\'].numpy()),\'vap_mom_r\':rms_np(V[\'mr\'].numpy()),\'vap_mom_z\':rms_np(V[\'mz\'].numpy()),\n        \'if_du_r\':rms_np(I[\'du_r\'].numpy()),\'if_du_z\':rms_np(I[\'du_z\'].numpy()),\'if_kin\':rms_np(I[\'kin\'].numpy()),\n        \'if_tanGauge\':rms_np(I[\'tangauge\'].numpy()),\'if_normal\':rms_np(I[\'normal\'].numpy()),\'if_tangent\':rms_np(I[\'tangential\'].numpy())\n    }\n    print(f\'\\nFRESH VALIDATION t={t:.3f}\')\n    print(\' liquid RMS: div=\',row[\'liq_div\'],\'mom-r=\',row[\'liq_mom_r\'],\'mom-z=\',row[\'liq_mom_z\'])\n    print(\' vapor RMS: cont=\',row[\'vap_cont\'],\'mom-r=\',row[\'vap_mom_r\'],\'mom-z=\',row[\'vap_mom_z\'])\n    print(\' interface RMS: du_r=\',row[\'if_du_r\'],\'du_z=\',row[\'if_du_z\'],\'kin=\',row[\'if_kin\'],\n          \'tanGauge=\',row[\'if_tanGauge\'],\'normal=\',row[\'if_normal\'],\'tangent=\',row[\'if_tangent\'])\n    return row\n\nfresh_rows=[validate_time(tv) for tv in [0.025,0.05,0.10]]\n\n# Explicit initial-time checks that replace the unavailable spherical p0 aid.\nmlr0,mlz0=liquid_t0_momentum_terms(Xliquid0)\nSall=thermo_time_state(Tnodes)\nJi0_now=float(initial_interface_traction_loss(Sall).numpy())\nqbars_now=qbar_at_nodes(Tnodes).numpy()\ninitial_checks={\n    \'liq_t0_mom_r\':rms_np(mlr0.numpy()),\'liq_t0_mom_z\':rms_np(mlz0.numpy()),\n    \'Ji0\':Ji0_now,\'qbar_max\':float(np.max(np.abs(qbars_now)))\n}\nprint(\'\\nINITIAL / GAUGE CHECKS\',initial_checks)\n\n# -----------------------------------------------------------------------------\n# Self-contained text report. Upload only this file after the run.\n# -----------------------------------------------------------------------------\nreport_file=\'point15_frozenG3_transfer_gate_report.txt\'\nwith open(report_file,\'w\',encoding=\'utf-8\') as f:\n    f.write(\'POINT 15 PRODUCTION FIBER PINN -- FROZEN G3 TRANSFER BASELINE\\n\')\n    f.write(\'Starts from scratch. No FEM/spherical trajectory or spherical analytical pressure used in training.\\n\')\n    f.write(\'Frozen G3 retained: chi liquid, third-order AD, G3 vapor net, 33 Chebyshev nodes, 320 vapor/slice, lambda_n=10, t0 traction/vapor momentum, all-time gauge.\\n\')\n    f.write(\'Necessary production changes: learned nonspherical interface/volume/centroid, learned p_l, no spherical sink, explicit t0 liquid momentum, solid/open-boundary/tangential-gauge conditions.\\n\\n\')\n\n    f.write(\'CONFIG\\n\')\n    f.write(f\'tmax={cfg.tmax} epochs={cfg.epochs} lr={cfg.lr} beta1={cfg.beta1} seed={cfg.seed}\\n\')\n    f.write(f\'n_time={cfg.n_time_slices} n_liquid_per_time={cfg.n_liquid_per_time} n_vapor_per_nonzero_time={cfg.n_vapor_per_time} n_interface_per_time={cfg.n_interface_per_time}\\n\')\n    f.write(f\'n_liquid_t0={cfg.n_liquid_t0_fixed} n_vapor_t0={cfg.n_vapor_t0_fixed} n_gauge_spatial={cfg.n_gauge_spatial} lambda_normal={cfg.lambda_normal}\\n\')\n    f.write(f\'chi_scale={cfg.chi_scale} p_scale={cfg.p_scale} vapor_u_scale={cfg.u_scale} q_scale={cfg.q_scale}\\n\')\n    f.write(f\'liquid_width={cfg.liquid_width} liquid_depth={cfg.liquid_depth} vapor_width={cfg.vapor_width} vapor_depth={cfg.vapor_depth} interface_width={cfg.interface_width} interface_depth={cfg.interface_depth} ff_features={cfg.ff_features}\\n\')\n    f.write(f\'trainable_parameters={sum(int(np.prod(v.shape)) for v in model.trainable_variables)}\\n\\n\')\n\n    f.write(\'TRAINING HISTORY\\n\')\n    f.write(\'epoch \'+\' \'.join(train_names)+\' elapsed_s\\n\')\n    for epoch,row,elapsed in train_history:\n        f.write(str(epoch)+\' \'+\' \'.join(f\'{v:.9e}\' for v in row)+f\' {elapsed:.3f}\\n\')\n\n    f.write(\'\\nSCALAR GEOMETRY / THERMODYNAMICS\\n\')\n    f.write(\'t Vratio Req Zcent Pv rho gV rmaxG ztop maxDisp qbar\\n\')\n    for r in scalar_rows:\n        f.write(f"{r[\'t\']:.6f} {r[\'ratio\']:.9e} {r[\'Req\']:.9e} {r[\'Zc\']:.9e} {r[\'Pv\']:.9e} {r[\'rho\']:.9e} {r[\'gV\']:.9e} "\n                f"{r[\'rmaxG\']:.9e} {r[\'ztop\']:.9e} {r[\'maxDisp\']:.9e} {r[\'qbar\']:.9e}\\n")\n\n    f.write(\'\\nFRESH VALIDATION RMS\\n\')\n    keys=[\'t\',\'liq_div\',\'liq_mom_r\',\'liq_mom_z\',\'vap_cont\',\'vap_mom_r\',\'vap_mom_z\',\'if_du_r\',\'if_du_z\',\'if_kin\',\'if_tanGauge\',\'if_normal\',\'if_tangent\']\n    f.write(\' \'.join(keys)+\'\\n\')\n    for r in fresh_rows: f.write(\' \'.join(f\'{r[k]:.9e}\' for k in keys)+\'\\n\')\n\n    f.write(\'\\nINITIAL / GAUGE CHECKS\\n\')\n    for k,v in initial_checks.items(): f.write(f\'{k} {v:.9e}\\n\')\n\n    f.write(\'\\nDENSE INTERFACE SAMPLES\\n\')\n    f.write(\'t xi r z\\n\')\n    for tv in [0.025,0.05,0.10]:\n        xi_dense,r_dense,z_dense=interface_snapshot(tv,101)\n        for xx,rr,zz in zip(xi_dense,r_dense,z_dense):\n            f.write(f\'{tv:.6f} {xx:.9e} {rr:.9e} {zz:.9e}\\n\')\n\nprint(\'wrote self-contained report:\',report_file)'

# %% [notebook code cell 3]

def replace_assignment(src, name, value):
    """Replace `self.<name>=...` or `self.<name> = ...` in ProductionConfig."""
    pat = re.compile(rf"(\s*self\.{re.escape(name)}\s*=\s*)([^\n#]+)")
    m = pat.search(src)
    if not m:
        raise RuntimeError(f'Config assignment not found: {name}')
    return src[:m.start(2)] + repr(value) + src[m.end(2):]


def build_level_source(spec):
    s = BASE_SOURCE
    # Controlled numerical-resolution changes only.
    for key, val in spec.items():
        if key == 'name':
            continue
        s = replace_assignment(s, key, val)

    level = spec['name']
    # Every level has the same optimizer budget and starts from scratch.
    s = replace_assignment(s, 'epochs', 22000)
    s = replace_assignment(s, 'seed', 1234)

    tmp_report = f'point16_{level}_raw_report.txt'
    s = s.replace(
        "report_file='point15_frozenG3_transfer_gate_report.txt'",
        f"report_file={tmp_report!r}"
    )
    s = s.replace(
        "POINT 15 PRODUCTION FIBER PINN -- FROZEN G3 TRANSFER BASELINE",
        f"POINT 16 PRODUCTION FIBER PINN -- {level.upper()} FROM-SCRATCH LEVEL"
    )
    s = s.replace(
        "print('STARTING FROM SCRATCH -- no checkpoint is loaded')",
        f"print('POINT16 {level.upper()}: STARTING FROM SCRATCH -- no checkpoint is loaded')"
    )
    # Explicit level line in the report.
    marker = "    f.write('Starts from scratch. No FEM/spherical trajectory or spherical analytical pressure used in training.\\n')"
    s = s.replace(marker, marker + f"\n    f.write('resolution_level={level}\\n')")
    compile(s, f'<point16-{level}>', 'exec')
    return s, tmp_report


def parse_report(text):
    lines = [ln.strip() for ln in text.splitlines()]
    out = {}

    # Final training-history row.
    try:
        i0 = lines.index('TRAINING HISTORY')
        i1 = lines.index('SCALAR GEOMETRY / THERMODYNAMICS')
        hist = [ln for ln in lines[i0+2:i1] if ln and ln[0].isdigit()]
        out['final_history'] = [float(x) for x in hist[-1].split()]
    except Exception:
        out['final_history'] = None

    # Scalar rows.
    scalar = {}
    try:
        i0 = lines.index('SCALAR GEOMETRY / THERMODYNAMICS')
        i1 = lines.index('FRESH VALIDATION RMS')
        for ln in lines[i0+2:i1]:
            if not ln or not (ln[0].isdigit() or ln[0] == '-'):
                continue
            vals = [float(x) for x in ln.split()]
            scalar[round(vals[0],6)] = vals
    except Exception:
        pass
    out['scalar'] = scalar

    # Fresh validation rows.
    fresh = {}
    try:
        i0 = lines.index('FRESH VALIDATION RMS')
        i1 = lines.index('INITIAL / GAUGE CHECKS')
        for ln in lines[i0+2:i1]:
            if not ln or not (ln[0].isdigit() or ln[0] == '-'):
                continue
            vals = [float(x) for x in ln.split()]
            fresh[round(vals[0],6)] = vals
    except Exception:
        pass
    out['fresh'] = fresh

    # Initial checks.
    checks = {}
    try:
        i0 = lines.index('INITIAL / GAUGE CHECKS')
        i1 = lines.index('DENSE INTERFACE SAMPLES')
        for ln in lines[i0+1:i1]:
            if ln:
                p = ln.split()
                if len(p) == 2:
                    checks[p[0]] = float(p[1])
    except Exception:
        pass
    out['checks'] = checks

    # Dense interface.
    dense = {}
    try:
        i0 = lines.index('DENSE INTERFACE SAMPLES')
        for ln in lines[i0+2:]:
            if not ln:
                continue
            vals = [float(x) for x in ln.split()]
            t = round(vals[0],6)
            dense.setdefault(t, []).append(vals[1:4])
        for t in dense:
            dense[t] = np.asarray(dense[t], dtype=float)
    except Exception:
        pass
    out['dense'] = dense

    # Parameter count.
    m = re.search(r'trainable_parameters=(\d+)', text)
    out['params'] = int(m.group(1)) if m else None
    return out


def rel(a,b,eps=1e-14):
    return abs(a-b)/(abs(b)+eps)


def contour_rms(A,B):
    # columns are xi,r,z; xi grids are intentionally identical (101 uniform points).
    if A.shape != B.shape:
        return float('nan')
    return float(np.sqrt(np.mean((A[:,1]-B[:,1])**2 + (A[:,2]-B[:,2])**2)))



def run_level(spec):
    level = spec['name']
    source, report_name = build_level_source(spec)
    script_path = WORKDIR / f'._point16_{level}.py'
    report_path = WORKDIR / report_name
    script_path.write_text(source, encoding='utf-8')

    print('\n' + '='*78)
    print(f'POINT 16 LEVEL: {level.upper()}')
    print('='*78)

    try:
        proc = subprocess.Popen(
            [sys.executable, '-u', str(script_path)],
            cwd=str(WORKDIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end='')
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f'{level} subprocess failed with return code {rc}')
        if not report_path.exists():
            raise RuntimeError(f'{level} completed but report was not created: {report_path}')
        text = report_path.read_text(encoding='utf-8')
        return parse_report(text), text, report_path
    finally:
        if script_path.exists():
            script_path.unlink()

# %% [notebook code cell 4]
def write_combined_report(results, raw_reports):
    lines = []
    completed = [name for name in LEVELS_TO_RUN if name in results]
    lines.append('POINT 16 PRODUCTION FIBER PINN -- CONTROLLED FROM-SCRATCH CONVERGENCE LADDER')
    lines.append('All levels use identical physics and 22000 epochs; only representation/collocation resolution changes.')
    lines.append('No checkpoint, FEM trajectory, spherical trajectory, or spherical analytical pressure is used.')
    lines.append('')

    lines.append('LEVEL SUMMARY AT t=0.1')
    lines.append('level params Vratio Req Zcent Pv rho gV rmaxG ztop maxDisp '
                 'liq_div liq_mom_r liq_mom_z vap_cont vap_mom_r vap_mom_z '
                 'if_du_r if_du_z if_kin if_tanGauge if_normal if_tangent '
                 't0_liq_mom_r t0_liq_mom_z qbar_max')
    for name in completed:
        P = results[name]
        s = P['scalar'][0.1]    # t,Vratio,Req,Zc,Pv,rho,gV,rmaxG,ztop,maxDisp,qbar
        f = P['fresh'][0.1]     # t + 12 residuals
        c = P['checks']
        vals = [
            name, str(P.get('params')),
            *[f'{x:.9e}' for x in s[1:10]],
            *[f'{x:.9e}' for x in f[1:]],
            f"{c.get('liq_t0_mom_r',float('nan')):.9e}",
            f"{c.get('liq_t0_mom_z',float('nan')):.9e}",
            f"{c.get('qbar_max',float('nan')):.9e}",
        ]
        lines.append(' '.join(vals))

    if 'medium' in results and 'fine' in results:
        m = results['medium']; f = results['fine']
        sm = m['scalar'][0.1]; sf = f['scalar'][0.1]
        labels = ['Vratio','Req','Zcent','Pv','rho','gV','rmaxG','ztop','maxDisp']
        lines.append('')
        lines.append('MEDIUM_TO_FINE_RELATIVE_DIFFERENCE_AT_t=0.1')
        for j,lab in enumerate(labels, start=1):
            lines.append(f'{lab} {rel(sm[j],sf[j]):.9e}')

        lines.append('')
        lines.append('DENSE_INTERFACE_RMS_DIFFERENCE_MEDIUM_TO_FINE')
        for tv in [0.025,0.05,0.10]:
            A=m['dense'].get(round(tv,6)); B=f['dense'].get(round(tv,6))
            val=contour_rms(A,B) if A is not None and B is not None else float('nan')
            lines.append(f'{tv:.6f} {val:.9e}')

    lines.append('')
    lines.append('RAW LEVEL REPORTS FOLLOW')
    for name in completed:
        if name in raw_reports:
            lines.append('')
            lines.append('='*80)
            lines.append(f'BEGIN RAW REPORT: {name.upper()}')
            lines.append('='*80)
            lines.append(raw_reports[name].rstrip())
            lines.append('='*80)
            lines.append(f'END RAW REPORT: {name.upper()}')
            lines.append('='*80)

    FINAL_REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('updated combined report:', FINAL_REPORT)


# %% [notebook code cell 5]

results = {}
raw_reports = {}
level_report_paths = []

for spec in LEVEL_SPECS:
    if spec['name'] not in LEVELS_TO_RUN:
        continue
    parsed, raw, report_path = run_level(spec)
    results[spec['name']] = parsed
    raw_reports[spec['name']] = raw
    level_report_paths.append(report_path)
    write_combined_report(results, raw_reports)

print('\nALL REQUESTED POINT-16 LEVELS COMPLETED.')
print('Final combined report:', FINAL_REPORT)

if not KEEP_LEVEL_REPORTS:
    for p in level_report_paths:
        if p.exists():
            p.unlink()
    print('temporary per-level reports removed; only the combined report is needed.')

# %% [notebook code cell 6]
for spec in LEVEL_SPECS:
    if spec['name'] not in LEVELS_TO_RUN:
        continue
    parsed, raw, report_path = run_level(spec)
    results[spec['name']] = parsed
    raw_reports[spec['name']] = raw
    level_report_paths.append(report_path)
    write_combined_report(results, raw_reports)

# %% [notebook code cell 7]

# Compact on-screen comparison after all levels finish.
print('\nPOINT 16 SUMMARY @ t=0.1')
print('level        V/V0        Req          liq_mom(r,z)          vap_cont      if_kin       if_normal')
for name in LEVELS_TO_RUN:
    P=results[name]
    s=P['scalar'][0.1]
    f=P['fresh'][0.1]
    print(f"{name:7s}  {s[1]:.8f}  {s[2]:.8f}  "
          f"({f[2]:.3e},{f[3]:.3e})  {f[4]:.3e}  {f[9]:.3e}  {f[11]:.3e}")

if 'medium' in results and 'fine' in results:
    sm=results['medium']['scalar'][0.1]
    sf=results['fine']['scalar'][0.1]
    print('\nmedium -> fine relative changes:')
    for j,lab in enumerate(['V/V0','Req','Zcent','Pv','rho','gV','rmaxG','ztop','maxDisp'],start=1):
        print(f'  {lab:8s}: {rel(sm[j],sf[j]):.3e}')
    print('\nmedium -> fine dense-interface RMS differences:')
    for tv in [0.025,0.05,0.10]:
        print(f'  t={tv:.3f}: {contour_rms(results["medium"]["dense"][tv],results["fine"]["dense"][tv]):.3e}')

print('\nUpload only:', FINAL_REPORT.name)

# %% [notebook code cell 8]

