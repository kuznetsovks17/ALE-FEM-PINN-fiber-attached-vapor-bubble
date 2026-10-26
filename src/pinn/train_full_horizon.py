#!/usr/bin/env python3
# Production full-horizon PINN
# Exported from a cleaned Jupyter notebook; notebook outputs are not embedded.

# %% [notebook code cell 1]
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import qmc
from numpy.polynomial.legendre import leggauss

np.random.seed(1234)
tf.random.set_seed(1234)

mse=lambda x: tf.reduce_mean(tf.square(x))

def rms_np(x):
    x=np.asarray(x).reshape(-1)
    return float(np.sqrt(np.mean(x*x))) if x.size else np.nan

class ProductionConfig:
    def __init__(self):
        # Same nondimensional production physics.
        self.Re=500.0
        self.Pi=1.25
        self.pv0=-1.0
        self.pinf=0.0
        self.gamma_v=1.0
        self.rho_v0=2.0e-2
        self.mu_ratio=2.0e-2

        # Production geometry.
        self.af=300.0/686.0
        self.rmin=0.0
        self.rmax=4.0
        self.zmin=-2.0
        self.zmax=4.0
        self.zc=float(np.sqrt(1.0-self.af*self.af))
        self.theta_rim=float(np.arccos(-self.zc))

        # Architecture gate only.
        self.tmin=0.0
        self.tmax=0.3

        self.hard_time_scale=0.10  # fixed to validated Point-16 gate scale
        # Frozen-G3 neural conditioning.
        self.axis_eps=2.0e-3
        self.lr=2.0e-4
        self.beta1=0.99
        self.seed=1234
        self.ff_features=20
        self.liquid_width=80
        self.liquid_depth=5
        self.vapor_width=80
        self.vapor_depth=4
        self.interface_top_width=40
        self.ff_scale=1.0
        self.chi_scale=3.0
        self.p_scale=4.0
        self.u_scale=3.0          # vapor velocity scale, as in G3
        self.q_scale=1.0

        # Frozen-G3 collocation structure.
        self.n_time_slices=97
        self.n_liquid_per_time=64# 33*48=1584 ~ frozen G3's 1536 liquid points
        self.n_vapor_per_time=480# nonzero slices only: 32*320=10240
        self.n_interface_per_time=48
        self.n_vapor_t0_fixed=384
        self.n_liquid_t0_fixed=384# necessary replacement for unavailable spherical p0 aid
        self.n_gauge_spatial=96
        self.n_boundary_per_time=8# 33*6=198 ~ frozen G3's 200 boundary points
        self.n_axis_per_time=8
        self.refresh_every=500
        self.interface_margin=0.02

        # Nonspherical geometry quadrature / interface network (new geometry only).
        self.n_volume_xi=96
        self.interface_width=64
        self.interface_depth=5
        self.geom_r_scale=0.20
        self.geom_z_scale=0.20

        # Frozen G3 force-balance weighting.
        self.lambda_normal=10.0

        # Training budget from final G3.
        self.epochs=22000

cfg=ProductionConfig()
print('af=',cfg.af,'zc=',cfg.zc,'theta_rim=',cfg.theta_rim)
print('gate=',(cfg.tmin,cfg.tmax),'Chebyshev nodes=',cfg.n_time_slices)
print('vapor points/nonzero slice=',cfg.n_vapor_per_time,'epochs=',cfg.epochs)

def chebyshev_lobatto_times(n,tmin,tmax):
    n=int(n)
    j=np.arange(n,dtype=np.float64)
    tau=0.5*(1.0-np.cos(np.pi*j/(n-1)))
    return (tmin+(tmax-tmin)*tau).astype(np.float32)

Tnodes_np=chebyshev_lobatto_times(cfg.n_time_slices,cfg.tmin,cfg.tmax)
Tnodes=tf.constant(Tnodes_np[:,None],tf.float32)
Tvapor_np=Tnodes_np[1:]

# Gauss-Legendre quadrature on [0,1] for differentiable geometric moments.
xg,wg=leggauss(cfg.n_volume_xi)
VOL_XI=tf.constant(((xg+1.0)/2.0).astype(np.float32)[:,None])
VOL_W=tf.constant((wg/2.0).astype(np.float32)[:,None])

print('first/last Chebyshev nodes:',Tnodes_np[:4],Tnodes_np[-4:])

# -----------------------------------------------------------------------------
# Frozen-G3 neural conditioning blocks.
# -----------------------------------------------------------------------------

# %% [notebook code cell 2]
class GaussianFourierBlock(tf.keras.layers.Layer):
    def __init__(self,n_features=16,scale=1.0,name=None):
        super().__init__(name=name); self.n_features=int(n_features); self.scale=float(scale)
    def build(self,input_shape):
        self.B=self.add_weight(name='B',shape=(int(input_shape[-1]),self.n_features),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.scale),trainable=False)
    def call(self,x):
        a=2.0*np.pi*tf.matmul(x,self.B)
        return tf.concat([x,tf.sin(a),tf.cos(a)],axis=1)

class NormalizedInput(tf.keras.layers.Layer):
    """Frozen-G3 smooth physical-coordinate input for high-order liquid AD."""
    def __init__(self,cfg,radial_even=False,name=None):
        super().__init__(name=name); self.cfg=cfg; self.radial_even=bool(radial_even)
    def call(self,X):
        X=tf.cast(X,tf.float32); r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]
        rn=(r-self.cfg.rmin)/(self.cfg.rmax-self.cfg.rmin)
        zn=(z-self.cfg.zmin)/(self.cfg.zmax-self.cfg.zmin)
        tn=(t-self.cfg.tmin)/(self.cfg.tmax-self.cfg.tmin)
        if self.radial_even: rn=tf.square(rn)
        return tf.concat([rn,zn,tn],axis=1)

class VolumeScaledFourierInput(tf.keras.layers.Layer):
    """Frozen-G3 vapor features generalized to actual volume and centroid."""
    def __init__(self,cfg,name=None):
        super().__init__(name=name); self.cfg=cfg
        self.ff=GaussianFourierBlock(cfg.ff_features,cfg.ff_scale)
    def call(self,inputs):
        X,Lv,Zc=inputs
        X=tf.cast(X,tf.float32); Lv=tf.cast(Lv,tf.float32); Zc=tf.cast(Zc,tf.float32)
        r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]
        Lsafe=tf.maximum(Lv,tf.cast(1e-5,Lv.dtype))
        rn=r/Lsafe
        zn=(z-Zc)/Lsafe
        tn=(t-self.cfg.tmin)/(self.cfg.tmax-self.cfg.tmin)
        return self.ff(tf.concat([rn,zn,tn],axis=1))

class LiquidScalarNetwork(tf.keras.Model):
    def __init__(self,cfg,output_scale,layers=4,dim=64,bounded=True,name='scalar'):
        super().__init__(name=name); self.output_scale=float(output_scale); self.bounded=bool(bounded)
        self.inp=NormalizedInput(cfg,radial_even=True,name=name+'_input')
        self.hidden=[tf.keras.layers.Dense(dim,activation='tanh') for _ in range(layers)]
        self.out=tf.keras.layers.Dense(1,kernel_initializer='zeros',bias_initializer='zeros')
    def call(self,X):
        y=self.inp(tf.cast(X,tf.float32))
        for layer in self.hidden: y=layer(y)
        y=self.out(y)
        if self.bounded: y=tf.tanh(y)
        return tf.cast(self.output_scale,tf.float32)*y

class VaporNet(tf.keras.Model):
    """Exact frozen-G3 vapor architecture: Fourier input, 3x60 trunk, three 60-wide heads."""
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.ffe=VolumeScaledFourierInput(cfg,name='vapor_ffe')
        self.common=[tf.keras.layers.Dense(cfg.vapor_width,activation='tanh') for _ in range(cfg.vapor_depth)]
        self.heads=[]
        for name in ['urv','uzv','qv']:
            h=tf.keras.layers.Dense(cfg.vapor_width,activation='tanh',name=name+'_h')
            o=tf.keras.layers.Dense(1,kernel_initializer='zeros',bias_initializer='zeros',name=name+'_out')
            self.heads.append((name,h,o))
    def call(self,X,Lv,Zc):
        x=self.ffe((X,Lv,Zc))
        for h in self.common: x=h(x)
        out=[]
        for name,h,o in self.heads:
            y=o(h(x)); out.append(tf.tanh(y) if name!='qv' else y)
        return tf.concat(out,axis=1)

class SmoothMLP(tf.keras.Model):
    """Only the nonspherical interface needs a new network absent from spherical G3."""
    def __init__(self,out_dim,width,depth,name=None):
        super().__init__(name=name)
        self.hidden=[tf.keras.layers.Dense(width,activation='tanh') for _ in range(depth)]
        self.out=tf.keras.layers.Dense(out_dim,kernel_initializer='zeros',bias_initializer='zeros')
    def call(self,x):
        for h in self.hidden: x=h(x)
        return self.out(x)

class FrozenG3FiberPINN(tf.keras.Model):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.chi=LiquidScalarNetwork(cfg,cfg.chi_scale,layers=cfg.liquid_depth,dim=cfg.liquid_width,bounded=True,name='chi_l')
        self.pressure=LiquidScalarNetwork(cfg,cfg.p_scale,layers=cfg.liquid_depth,dim=cfg.liquid_width,bounded=False,name='p_l')
        self.vapor=VaporNet(cfg)
        self.iface_shape=SmoothMLP(2,cfg.interface_width,cfg.interface_depth,name='iface_shape')
        self.iface_top=SmoothMLP(1,cfg.interface_top_width,2,name='iface_top')
        self.V0=None

    def tau(self,t):
        # Feature normalization over the current training horizon.
        return (tf.cast(t,tf.float32)-self.cfg.tmin)/(self.cfg.tmax-self.cfg.tmin)

    def hardtime(self,t):
        # Hard IC prefactor uses a FIXED physical scale.  At tmax=0.1 this
        # exactly equals the old tau, so the validated Point-16 gate is unchanged.
        return (tf.cast(t,tf.float32)-self.cfg.tmin)/self.cfg.hard_time_scale

    def interface_features(self,xi,t):
        return tf.concat([2.0*xi-1.0,2.0*self.tau(t)-1.0],axis=1)

    def interface(self,xi,t):
        xi=tf.cast(xi,tf.float32); t=tf.cast(t,tf.float32); tau=self.tau(t); ht=self.hardtime(t)
        th=tf.cast(self.cfg.theta_rim,xi.dtype)*(1.0-xi)
        r0=tf.sin(th)
        z0=tf.cast(self.cfg.zc,xi.dtype)+tf.cos(th)
        raw=tf.tanh(self.iface_shape(self.interface_features(xi,t)))
        rr,zz=tf.split(raw,2,axis=1)
        top=tf.tanh(self.iface_top(2.0*tau-1.0))

        # Exact initial shape, exact pinned rim, exact axis intersection, X_t(xi,0)=0.
        r=r0*tf.exp(tf.cast(self.cfg.geom_r_scale,r0.dtype)*tf.square(ht)*xi*rr)
        Btop=2.0*xi-tf.square(xi)
        Bint=xi*tf.square(1.0-xi)
        z=z0+tf.cast(self.cfg.geom_z_scale,z0.dtype)*tf.square(ht)*(Btop*top+Bint*zz)
        return r,z

    def volume_centroid(self,t):
        t=tf.cast(t,tf.float32); nt=tf.shape(t)[0]; nq=tf.shape(VOL_XI)[0]
        xi=tf.tile(VOL_XI,[nt,1])
        tt=tf.reshape(tf.repeat(t,repeats=nq,axis=0),[-1,1])
        with tf.GradientTape() as tape:
            tape.watch(xi)
            r,z=self.interface(xi,tt)
        zxi=tape.gradient(z,xi)
        base=tf.square(r)*zxi
        base=tf.reshape(base,[nt,nq]); zmat=tf.reshape(z,[nt,nq])
        w=tf.reshape(VOL_W,[1,-1])
        den=tf.reduce_sum(base*w,axis=1,keepdims=True)
        znum=tf.reduce_sum(zmat*base*w,axis=1,keepdims=True)
        V=tf.cast(np.pi,base.dtype)*den
        Zc=znum/(den+1e-12)
        return V,Zc

    def vapor_characteristics(self,t):
        V,Zc=self.volume_centroid(t)
        ratio=V/tf.cast(self.V0,V.dtype)
        Lv=tf.pow(tf.maximum(ratio,1e-10),1.0/3.0)  # frozen G3 L_v=(V/V0)^(1/3)
        return V,ratio,Zc,Lv

    def liquid_scalars(self,X):
        return self.chi(X),self.pressure(X)

    def liquid_state(self,X):
        """Minimal nonspherical generalization of frozen G3 streamfunction liquid.

        Frozen G3 used sink + tau^2*chi correction.  The sink is spherical-only and removed.
        Because production initial acceleration is unknown, tau (not tau^2) multiplies the full
        streamfunction velocity, preserving u(0)=0 while leaving u_t(0) learnable.
        """
        X=tf.cast(X,tf.float32); r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([r,z]); chi=self.chi(tf.concat([r,z,t],axis=1)); p=self.pressure(tf.concat([r,z,t],axis=1))
        chi_r=tape.gradient(chi,r); chi_z=tape.gradient(chi,z); del tape
        ht=self.hardtime(t)
        ur=-ht*r*chi_z
        uz=ht*(2.0*chi+r*chi_r)
        return ur,uz,p,chi

    def vapor_hat(self,X,Lv,Zc):
        return self.vapor(tf.cast(X,tf.float32),Lv,Zc)

    def vapor_state_scaled(self,X,Lv,Zc):
        X=tf.cast(X,tf.float32); t=X[:,2:3]
        uh_r,uh_z,qh=tf.split(self.vapor_hat(X,Lv,Zc),3,axis=1)
        ht=self.hardtime(t)
        return ht*self.cfg.u_scale*uh_r,ht*self.cfg.u_scale*uh_z,self.cfg.q_scale*qh

    def vapor_state_auto(self,X):
        X=tf.cast(X,tf.float32); t=X[:,2:3]
        _,_,Zc,Lv=self.vapor_characteristics(t)
        return self.vapor_state_scaled(X,Lv,Zc)

model=FrozenG3FiberPINN(cfg)
# Build interface/liquid, then establish the exact geometric initial volume.
_ = model.interface(tf.constant([[0.5]],tf.float32),tf.constant([[0.0]],tf.float32))
_ = model.liquid_state(tf.constant([[1.0,1.0,0.0]],tf.float32))
V0tmp,_=model.volume_centroid(tf.constant([[0.0]],tf.float32))
model.V0=tf.stop_gradient(V0tmp)
# Build vapor after V0 exists.
_ = model.vapor_state_auto(tf.constant([[0.2,0.5,0.0]],tf.float32))
print('initial geometric V0=',float(model.V0.numpy()[0,0]))
print('trainable parameters=',sum(int(np.prod(v.shape)) for v in model.trainable_variables))
print('liquid: frozen-G3 chi architecture + learned nonspherical p_l; no spherical sink/pressure')

@tf.function(reduce_retracing=True)
def interface_kinematics(xi,t):
    xi=tf.cast(xi,tf.float32); t=tf.cast(t,tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([xi,t]); r,z=model.interface(xi,t)
    rxi=tape.gradient(r,xi); zxi=tape.gradient(z,xi)
    rt=tape.gradient(r,t); zt=tape.gradient(z,t); del tape
    ds=tf.sqrt(rxi*rxi+zxi*zxi+1e-12)
    tr=rxi/ds; tz=zxi/ds
    nr=zxi/ds; nz=-rxi/ds   # vapor -> liquid
    return r,z,rxi,zxi,rt,zt,nr,nz,tr,tz

@tf.function(reduce_retracing=True)
def thermo_time_state(t):
    t=tf.cast(t,tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        V,ratio,Zc,Lv=model.vapor_characteristics(t)
    Vt=tape.gradient(V,t)
    Zct=tape.gradient(Zc,t)
    Lvt=tape.gradient(Lv,t)
    del tape
    gV=Vt/tf.maximum(V,1e-10)
    Pv=(cfg.Pi+cfg.pv0)*tf.pow(tf.maximum(ratio,1e-10),-cfg.gamma_v)-cfg.Pi
    rho=cfg.rho_v0/tf.maximum(ratio,1e-10)
    return dict(V=V,ratio=ratio,Vt=Vt,gV=gV,Pv=Pv,rho=rho,Zc=Zc,Lv=Lv,Zct=Zct,Lvt=Lvt)

@tf.function(reduce_retracing=True)
def geometry_observables(t):
    S=thermo_time_state(t)
    Req=tf.pow(3.0*S['V']/(4.0*np.pi),1.0/3.0)
    return dict(**S,Req=Req)

S0=geometry_observables(tf.constant([[0.0]],tf.float32))
print('initial Req=',float(S0['Req'].numpy()[0,0]),'Zcent=',float(S0['Zc'].numpy()[0,0]))


# %% [notebook code cell 3]
class ProductionSampler:
    def __init__(self,cfg): self.cfg=cfg; self.seed=cfg.seed
    def sobol(self,n,d):
        m=int(np.ceil(np.log2(max(2,int(n)))))
        s=qmc.Sobol(d=d,scramble=True,seed=self.seed); self.seed+=1
        return s.random_base2(m)[:int(n)]

    def interface_curve_np(self,t,n=501):
        xi=np.linspace(0.0,1.0,n,dtype=np.float32)[:,None]
        tt=np.full_like(xi,float(t))
        r,z=model.interface(tf.constant(xi),tf.constant(tt))
        r=r.numpy().reshape(-1); z=z.numpy().reshape(-1)
        order=np.argsort(z)
        return r[order],z[order]

    def liquid_at_time(self,t,n,margin=None):
        c=self.cfg; margin=c.interface_margin if margin is None else float(margin)
        ri,zi=self.interface_curve_np(t); ztop=float(zi[-1]); out=[]; need=int(n)
        while need>0:
            U=self.sobol(max(512,4*need),2)
            r=c.axis_eps+(c.rmax-c.axis_eps)*U[:,0]
            z=c.zmin+(c.zmax-c.zmin)*U[:,1]
            solid=(z<0.0)&(r<c.af+margin)
            vap=np.zeros_like(solid)
            band=(z>=0.0)&(z<=ztop)
            if np.any(band):
                rb=np.interp(z[band],zi,ri)
                vap[band]=r[band]<(rb+margin)
            keep=~(solid|vap)
            X=np.c_[r[keep],z[keep],np.full(np.count_nonzero(keep),t)]
            take=min(need,len(X))
            if take:
                out.append(X[:take]); need-=take
        return np.vstack(out).astype(np.float32)

    def liquid_cloud(self):
        return np.vstack([self.liquid_at_time(float(t),self.cfg.n_liquid_per_time) for t in Tnodes_np]).astype(np.float32)

    def vapor_reference_cloud(self):
        rows=[]; idx=[]
        # Exactly as frozen G3: full vapor PDE only on the 32 nonzero Chebyshev slices.
        for j,t in enumerate(Tnodes_np[1:],start=1):
            U=self.sobol(self.cfg.n_vapor_per_time,2)
            eta=0.01+0.97*U[:,0:1]; xi=0.01+0.98*U[:,1:2]
            rows.append(np.c_[eta,xi,np.full(len(U),t)])
            idx.append(np.full(len(U),j,np.int32))
        return np.vstack(rows).astype(np.float32),np.concatenate(idx)

    def vapor_t0_reference(self):
        U=self.sobol(self.cfg.n_vapor_t0_fixed,2)
        return np.c_[0.01+0.97*U[:,0],0.01+0.98*U[:,1],np.zeros(len(U))].astype(np.float32)

    def interface_cloud(self):
        rows=[]; idx=[]
        for j,t in enumerate(Tnodes_np):
            U=self.sobol(self.cfg.n_interface_per_time,1)
            xi=0.005+0.99*U[:,0:1]
            rows.append(np.c_[xi,np.full(len(U),t)])
            idx.append(np.full(len(U),j,np.int32))
        return np.vstack(rows).astype(np.float32),np.concatenate(idx)

    def boundary_cloud(self,kind):
        rows=[]
        for t in Tnodes_np:
            U=self.sobol(self.cfg.n_boundary_per_time,1)[:,0]
            if kind=='fiber':
                r=np.full_like(U,self.cfg.af); z=self.cfg.zmin+(0.0-self.cfg.zmin)*U
            elif kind=='tip':
                r=self.cfg.af*U; z=np.zeros_like(U)
            elif kind=='farr':
                r=np.full_like(U,self.cfg.rmax); z=self.cfg.zmin+(self.cfg.zmax-self.cfg.zmin)*U
            elif kind=='farz':
                r=self.cfg.rmax*U; z=np.full_like(U,self.cfg.zmax)
            elif kind=='bottom':
                r=self.cfg.af+(self.cfg.rmax-self.cfg.af)*U; z=np.full_like(U,self.cfg.zmin)
            else: raise ValueError(kind)
            rows.append(np.c_[r,z,np.full_like(U,t)])
        return np.vstack(rows).astype(np.float32)

    def vapor_axis_reference(self):
        rows=[]; idx=[]
        for j,t in enumerate(Tnodes_np):
            U=self.sobol(self.cfg.n_axis_per_time,1)[:,0]
            rows.append(np.c_[0.01+0.98*U,np.full_like(U,t)])  # s,z fraction + time
            idx.append(np.full(len(U),j,np.int32))
        return np.vstack(rows).astype(np.float32),np.concatenate(idx)

sampler=ProductionSampler(cfg)
Yvap_np,VapTimeIdx_np=sampler.vapor_reference_cloud()
Yv0_np=sampler.vapor_t0_reference()
Giface_np,IfaceTimeIdx_np=sampler.interface_cloud()
Yaxis_np,AxisTimeIdx_np=sampler.vapor_axis_reference()

Yvap=tf.constant(Yvap_np); VapTimeIdx=tf.constant(VapTimeIdx_np,tf.int32)
Yv0=tf.constant(Yv0_np)
Giface=tf.constant(Giface_np); IfaceTimeIdx=tf.constant(IfaceTimeIdx_np,tf.int32)
Yaxis=tf.constant(Yaxis_np); AxisTimeIdx=tf.constant(AxisTimeIdx_np,tf.int32)

Xfiber=tf.constant(sampler.boundary_cloud('fiber'))
Xtip=tf.constant(sampler.boundary_cloud('tip'))
Xfarr=tf.constant(sampler.boundary_cloud('farr'))
Xfarz=tf.constant(sampler.boundary_cloud('farz'))
Xbottom=tf.constant(sampler.boundary_cloud('bottom'))
# boundary_cloud is ordered by fixed Chebyshev time slice.
TipTimeIdx=tf.constant(np.repeat(np.arange(cfg.n_time_slices,dtype=np.int32),cfg.n_boundary_per_time),tf.int32)

Xliquid=tf.Variable(sampler.liquid_cloud(),trainable=False)
Xliquid0=tf.Variable(sampler.liquid_at_time(0.0,cfg.n_liquid_t0_fixed,margin=0.01),trainable=False)

# Fixed 64-point reference cloud for the gauge at every one of the 33 time nodes.
Ug=sampler.sobol(cfg.n_gauge_spatial,2).astype(np.float32)
Ygauge=tf.constant(np.c_[0.01+0.97*Ug[:,0],0.01+0.98*Ug[:,1]].astype(np.float32))

# Frozen-style separate initial interface traction set.
U0=sampler.sobol(cfg.n_interface_per_time,1).astype(np.float32)
G0=tf.constant(np.c_[0.005+0.99*U0[:,0],np.zeros(len(U0))].astype(np.float32))


def refresh_spatial_collocation(verbose=True):
    Xliquid.assign(sampler.liquid_cloud())
    Xliquid0.assign(sampler.liquid_at_time(0.0,cfg.n_liquid_t0_fixed,margin=0.01))
    if verbose: print('refreshed liquid clouds:',Xliquid.shape,Xliquid0.shape)

print('cloud sizes: liquid=',Xliquid.shape,'vapor(nonzero)=',Yvap.shape,'interface=',Giface.shape)
print('t0 liquid=',Xliquid0.shape,'t0 vapor=',Yv0.shape,'boundary each=',Xfiber.shape)


# %% [notebook code cell 4]
@tf.function(reduce_retracing=True)
def map_vapor_reference(Y):
    eta,xi,t=Y[:,0:1],Y[:,1:2],Y[:,2:3]
    rG,zG=model.interface(xi,t)
    return tf.concat([tf.stop_gradient(eta*rG),tf.stop_gradient(zG),tf.stop_gradient(t)],axis=1)

@tf.function(reduce_retracing=True)
def map_vapor_t0(Y):
    return map_vapor_reference(Y)

@tf.function(reduce_retracing=True)
def qbar_from_state(tnodes,Lv_nodes,Zc_nodes):
    """Frozen-G3 gauge mean on 64 reference points at every supplied time node."""
    tnodes=tf.cast(tnodes,tf.float32); nt=tf.shape(tnodes)[0]; ng=tf.shape(Ygauge)[0]
    eta0=Ygauge[:,0:1]; xi0=Ygauge[:,1:2]
    eta=tf.tile(eta0,[nt,1]); xi=tf.tile(xi0,[nt,1])
    tt=tf.reshape(tf.repeat(tnodes,repeats=ng,axis=0),[-1,1])
    with tf.GradientTape() as tape:
        tape.watch(xi); rG,zG=model.interface(xi,tt)
    zxi=tape.gradient(zG,xi)
    r=tf.stop_gradient(eta*rG); z=tf.stop_gradient(zG)
    X=tf.concat([r,z,tt],axis=1)
    Lv=tf.reshape(tf.repeat(Lv_nodes,repeats=ng,axis=0),[-1,1])
    Zc=tf.reshape(tf.repeat(Zc_nodes,repeats=ng,axis=0),[-1,1])
    _,_,q=model.vapor_state_scaled(X,Lv,Zc)
    w=tf.stop_gradient(eta*tf.square(rG)*zxi)
    q=tf.reshape(q,[nt,ng]); w=tf.reshape(w,[nt,ng])
    return tf.reduce_sum(q*w,axis=1)/(tf.reduce_sum(w,axis=1)+1e-12)

@tf.function(reduce_retracing=True)
def qbar_at_nodes(tnodes):
    """Diagnostic wrapper; training reuses its already-computed geometry state."""
    S=thermo_time_state(tnodes)
    return qbar_from_state(tnodes,S['Lv'],S['Zc'])

@tf.function(reduce_retracing=True)
def liquid_terms(X):
    """Frozen-G3 third-order liquid AD with learned nonspherical pressure."""
    r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]
    with tf.GradientTape(persistent=True) as tape3:
        tape3.watch([r,z])
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([r,z,t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([r,z])
                inp=tf.concat([r,z,t],axis=1)
                chi=model.chi(inp); p=model.pressure(inp); ht=model.hardtime(t)
            chi_r=tape1.gradient(chi,r); chi_z=tape1.gradient(chi,z); del tape1
            ur=-ht*r*chi_z
            uz=ht*(2.0*chi+r*chi_r)
        ur_r=tape2.gradient(ur,r); ur_z=tape2.gradient(ur,z); ur_t=tape2.gradient(ur,t)
        uz_r=tape2.gradient(uz,r); uz_z=tape2.gradient(uz,z); uz_t=tape2.gradient(uz,t)
        p_r=tape2.gradient(p,r); p_z=tape2.gradient(p,z); del tape2
    ur_rr=tape3.gradient(ur_r,r); ur_zz=tape3.gradient(ur_z,z)
    uz_rr=tape3.gradient(uz_r,r); uz_zz=tape3.gradient(uz_z,z); del tape3
    rs=tf.maximum(r,tf.cast(cfg.axis_eps,r.dtype))
    div=ur_r+ur/rs+uz_z
    lapr=ur_rr+ur_r/rs-ur/(rs*rs)+ur_zz
    lapz=uz_rr+uz_r/rs+uz_zz
    mr=ur_t+ur*ur_r+uz*ur_z+p_r-lapr/cfg.Re
    mz=uz_t+ur*uz_r+uz*uz_z+p_z-lapz/cfg.Re
    return dict(mr=mr,mz=mz,div=div,ur=ur,uz=uz,p=p)

@tf.function(reduce_retracing=True)
def liquid_t0_momentum_terms(X):
    """Necessary nonspherical replacement for G3's analytical initial-pressure aid.

    At t=0, u=0 exactly, so convection and viscosity vanish.  With the
    horizon-portable hard prefactor u=(t/hard_time_scale)*U_chi,
    u_t(0)=U_chi/hard_time_scale.  Enforce u_t + grad(p)=0 directly.
    """
    r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([r,z]); inp=tf.concat([r,z,t],axis=1); chi=model.chi(inp); p=model.pressure(inp)
    chi_r=tape.gradient(chi,r); chi_z=tape.gradient(chi,z)
    p_r=tape.gradient(p,r); p_z=tape.gradient(p,z); del tape
    tau_t=tf.cast(1.0/cfg.hard_time_scale,r.dtype)
    ur_t=-tau_t*r*chi_z
    uz_t=tau_t*(2.0*chi+r*chi_r)
    return ur_t+p_r,uz_t+p_z

@tf.function(reduce_retracing=True)
def vapor_terms(X,rho,gV,Lv,Zc,Lvt,Zct):
    """Frozen-G3 primitive vapor PDE with generalized volume/centroid feature chain rule."""
    r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]
    with tf.GradientTape(persistent=True) as tape3:
        tape3.watch([r,z])
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([r,z,t,Lv,Zc])
            ur,uz,q=model.vapor_state_scaled(tf.concat([r,z,t],axis=1),Lv,Zc)
        ur_r=tape2.gradient(ur,r); ur_z=tape2.gradient(ur,z)
        uz_r=tape2.gradient(uz,r); uz_z=tape2.gradient(uz,z)
        q_r=tape2.gradient(q,r); q_z=tape2.gradient(q,z)
        ur_t0=tape2.gradient(ur,t); uz_t0=tape2.gradient(uz,t)
        ur_L=tape2.gradient(ur,Lv); uz_L=tape2.gradient(uz,Lv)
        ur_Z=tape2.gradient(ur,Zc); uz_Z=tape2.gradient(uz,Zc)
        del tape2
        ur_t=ur_t0+ur_L*Lvt+ur_Z*Zct
        uz_t=uz_t0+uz_L*Lvt+uz_Z*Zct
    ur_rr=tape3.gradient(ur_r,r); ur_zz=tape3.gradient(ur_z,z)
    uz_rr=tape3.gradient(uz_r,r); uz_zz=tape3.gradient(uz_z,z); del tape3
    rs=tf.maximum(r,tf.cast(cfg.axis_eps,r.dtype))
    div=ur_r+ur/rs+uz_z
    cont=div-gV
    lapr=ur_rr+ur_r/rs-ur/(rs*rs)+ur_zz
    lapz=uz_rr+uz_r/rs+uz_zz
    fac=(cfg.mu_ratio/cfg.Re)/rho
    mr=ur_t+ur*ur_r+uz*ur_z+q_r-fac*lapr
    mz=uz_t+ur*uz_r+uz*uz_z+q_z-fac*lapz
    return dict(cont=cont,mr=mr,mz=mz,ur=ur,uz=uz,q=q)

@tf.function(reduce_retracing=True)
def vapor_t0_momentum_terms(X,Lv,Zc):
    """Frozen-G3 cheap t=0 vapor residual: u_v,t + grad(q_v)=0."""
    r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([r,z]); raw=model.vapor_hat(tf.concat([r,z,t],axis=1),Lv,Zc); uh_r,uh_z,qh=tf.split(raw,3,axis=1)
        q=cfg.q_scale*qh
    q_r=tape.gradient(q,r); q_z=tape.gradient(q,z); del tape
    tau_t=tf.cast(1.0/cfg.hard_time_scale,r.dtype)
    return cfg.u_scale*tau_t*uh_r+q_r,cfg.u_scale*tau_t*uh_z+q_z

@tf.function(reduce_retracing=True)
def interface_residuals(G,time_idx,S):
    xi,t=G[:,0:1],G[:,1:2]
    r,z,rxi,zxi,rt,zt,nr,nz,tr,tz=interface_kinematics(xi,t)
    Lv=tf.gather(S['Lv'],time_idx); Zc=tf.gather(S['Zc'],time_idx)
    Pv=tf.gather(S['Pv'],time_idx); rho=tf.gather(S['rho'],time_idx)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([r,z]); X=tf.concat([r,z,t],axis=1)
        ulr,ulz,p,_=model.liquid_state(X)
        uvr,uvz,q=model.vapor_state_scaled(X,Lv,Zc)
    ulr_r=tape.gradient(ulr,r); ulr_z=tape.gradient(ulr,z)
    ulz_r=tape.gradient(ulz,r); ulz_z=tape.gradient(ulz,z)
    uvr_r=tape.gradient(uvr,r); uvr_z=tape.gradient(uvr,z)
    uvz_r=tape.gradient(uvz,r); uvz_z=tape.gradient(uvz,z); del tape

    pvtot=Pv+rho*q
    muL=1.0/cfg.Re; muV=cfg.mu_ratio/cfg.Re
    srrL=-p+2.0*muL*ulr_r; szzL=-p+2.0*muL*ulz_z; srzL=muL*(ulr_z+ulz_r)
    srrV=-pvtot+2.0*muV*uvr_r; szzV=-pvtot+2.0*muV*uvz_z; srzV=muV*(uvr_z+uvz_r)
    tLr=srrL*nr+srzL*nz; tLz=srzL*nr+szzL*nz
    tVr=srrV*nr+srzV*nz; tVz=srzV*nr+szzV*nz
    dtr=tLr-tVr; dtz=tLz-tVz
    normal=dtr*nr+dtz*nz
    tangential=dtr*tr+dtz*tz

    du_r=ulr-uvr; du_z=ulz-uvz
    kin=(rt-ulr)*nr+(zt-ulz)*nz
    tangauge=(rt-ulr)*tr+(zt-ulz)*tz
    return dict(du_r=du_r,du_z=du_z,kin=kin,tangauge=tangauge,
                normal=normal,tangential=tangential,p=p,pvtot=pvtot,q=q)

@tf.function(reduce_retracing=True)
def weighted_interface_loss(G,time_idx,S):
    I=interface_residuals(G,time_idx,S)
    comps=(mse(I['du_r']),mse(I['du_z']),mse(I['kin']),mse(I['tangauge']),mse(I['normal']),mse(I['tangential']))
    Ji=comps[0]+comps[1]+comps[2]+comps[3]+cfg.lambda_normal*comps[4]+comps[5]
    return (Ji,)+comps

@tf.function(reduce_retracing=True)
def initial_interface_traction_loss(S):
    idx=tf.zeros((tf.shape(G0)[0],),tf.int32)
    I=interface_residuals(G0,idx,S)
    return cfg.lambda_normal*mse(I['normal'])+mse(I['tangential'])

@tf.function(reduce_retracing=True)
def liquid_boundary_traction(X,nr,nz):
    r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]; nr=tf.cast(nr,r.dtype); nz=tf.cast(nz,r.dtype)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([r,z]); ur,uz,p,_=model.liquid_state(tf.concat([r,z,t],axis=1))
    ur_r=tape.gradient(ur,r); ur_z=tape.gradient(ur,z); uz_r=tape.gradient(uz,r); uz_z=tape.gradient(uz,z); del tape
    mu=1.0/cfg.Re
    srr=-p+2.0*mu*ur_r; szz=-p+2.0*mu*uz_z; srz=mu*(ur_z+uz_r)
    return srr*nr+srz*nz+cfg.pinf*nr, srz*nr+szz*nz+cfg.pinf*nz

@tf.function(reduce_retracing=True)
def solid_boundary_losses(S):
    ulr,ulz,_,_=model.liquid_state(Xfiber)
    Lv=tf.gather(S['Lv'],TipTimeIdx); Zc=tf.gather(S['Zc'],TipTimeIdx)
    uvr,uvz,_=model.vapor_state_scaled(Xtip,Lv,Zc)
    return mse(ulr)+mse(ulz),mse(uvr)+mse(uvz)

@tf.function(reduce_retracing=True)
def far_traction_loss():
    ar,az=liquid_boundary_traction(Xfarr,1.0,0.0)
    br,bz=liquid_boundary_traction(Xfarz,0.0,1.0)
    cr,cz=liquid_boundary_traction(Xbottom,0.0,-1.0)
    return (mse(ar)+mse(az)+mse(br)+mse(bz)+mse(cr)+mse(cz))/3.0

@tf.function(reduce_retracing=True)
def vapor_axis_loss(S):
    s,t=Yaxis[:,0:1],Yaxis[:,1:2]
    idx=AxisTimeIdx
    xi1=tf.ones_like(t)
    _,ztop=model.interface(xi1,t)
    r=tf.zeros_like(t); z=tf.stop_gradient(s*ztop)
    Lv=tf.gather(S['Lv'],idx); Zc=tf.gather(S['Zc'],idx)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(r); ur,uz,q=model.vapor_state_scaled(tf.concat([r,z,t],axis=1),Lv,Zc)
    uz_r=tape.gradient(uz,r); q_r=tape.gradient(q,r); del tape
    return mse(ur)+mse(uz_r)+mse(q_r)

@tf.function(reduce_retracing=True)
def geometry_regularization():
    nt=tf.shape(Tnodes)[0]; nq=tf.shape(VOL_XI)[0]
    xi=tf.tile(VOL_XI,[nt,1]); tt=tf.reshape(tf.repeat(Tnodes,repeats=nq,axis=0),[-1,1])
    with tf.GradientTape() as tape:
        tape.watch(xi); r,z=model.interface(xi,tt)
    zxi=tape.gradient(z,xi)
    return mse(tf.nn.relu(1.0e-4-zxi))


# %% [notebook code cell 5]

# -----------------------------------------------------------------------------
# STATIC-SHAPE CAUSAL CURRICULUM -- FROM SCRATCH
# -----------------------------------------------------------------------------
from pathlib import Path

REPORT_FILE = Path('point16_fine_t030_causal_STATIC_fromScratch_report.txt')

print('STATIC CAUSAL CURRICULUM STARTS FROM SCRATCH -- no checkpoint is loaded.')

# One optimizer for Stage 1 -> Stage 2 -> Stage 3, so Adam state is preserved across all stage transitions.
optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr, beta_1=cfg.beta1)
try:
    optimizer.build(model.trainable_variables)
except Exception:
    # Older TF/Keras versions lazily create optimizer slots on first update.
    pass

TRAIN_NAMES=[
    'loss','Jml','JldDiag','Jml0','Jcv','Jmv','Jmv0','Ji(w)','Ji0',
    'Jfiber','Jtip','Jfar','Jq','Jaxis','Jgeom',
    'Jdu_r','Jdu_z','Jkin','JtanGauge','Jnormal','Jtang',
    'max|qbar|','Vratio_active_end'
]

def robust_active_node_count(cutoff):
    cutoff=float(cutoff)
    if abs(cutoff-float(cfg.tmax)) < 1.0e-10:
        return int(cfg.n_time_slices)  # exact final endpoint: all 97 nodes
    return int(np.count_nonzero(
        Tnodes_np.astype(np.float64) <= cutoff + 1.0e-7
    ))

STAGES=[
    dict(name='stage1_t010', cutoff=0.10, epochs=22000),
    dict(name='stage2_t020', cutoff=0.20, epochs=22000),
    dict(name='stage3_t030', cutoff=0.30, epochs=22000),
]
for st in STAGES:
    st['active_n']=robust_active_node_count(st['cutoff'])

assert STAGES[0]['active_n'] == 38, STAGES[0]
assert STAGES[1]['active_n'] == 59, STAGES[1]
assert STAGES[2]['active_n'] == cfg.n_time_slices == 97, STAGES[2]

print('\nSTATIC CAUSAL STAGES')
for st in STAGES:
    print(st['name'],
          'cutoff=',st['cutoff'],
          'active_nodes=',st['active_n'],'/',cfg.n_time_slices,
          'epochs=',st['epochs'])


def liquid_cloud_for_active_nodes(active_n):
    """Eager sampler. Output shape is fixed for a given causal stage."""
    rows=[
        sampler.liquid_at_time(float(t),cfg.n_liquid_per_time)
        for t in Tnodes_np[:int(active_n)]
    ]
    return np.vstack(rows).astype(np.float32)


def build_static_stage(stage):
    """
    Build all stage slices eagerly using Python integer sizes.
    train_step() closes over these concrete-shape tensors/variables.
    """
    n=int(stage['active_n'])
    nl=n*cfg.n_liquid_per_time
    nv=(n-1)*cfg.n_vapor_per_time
    ni=n*cfg.n_interface_per_time
    nb=n*cfg.n_boundary_per_time
    na=n*cfg.n_axis_per_time

    data={
        'name':stage['name'],
        'cutoff':float(stage['cutoff']),
        'active_n':n,
        'epochs':int(stage['epochs']),
        'tnodes':tf.constant(Tnodes_np[:n,None],tf.float32),

        # Moving-domain liquid clouds: fixed SHAPE, refreshable VALUES.
        'Xl':tf.Variable(
            liquid_cloud_for_active_nodes(n),
            trainable=False,
            shape=tf.TensorShape([nl,3]),
            name=f"{stage['name']}_Xl"
        ),
        'Xl0':tf.Variable(
            sampler.liquid_at_time(0.0,cfg.n_liquid_t0_fixed,margin=0.01),
            trainable=False,
            shape=tf.TensorShape([cfg.n_liquid_t0_fixed,3]),
            name=f"{stage['name']}_Xl0"
        ),

        # All remaining stage slices are immutable fixed-shape tensors.
        'Yv':tf.constant(Yvap_np[:nv],tf.float32),
        'vidx':tf.constant(VapTimeIdx_np[:nv],tf.int32),
        'Gi':tf.constant(Giface_np[:ni],tf.float32),
        'iidx':tf.constant(IfaceTimeIdx_np[:ni],tf.int32),

        'Xfib':tf.constant(Xfiber.numpy()[:nb],tf.float32),
        'Xtp':tf.constant(Xtip.numpy()[:nb],tf.float32),
        'tipidx':tf.constant(TipTimeIdx.numpy()[:nb],tf.int32),
        'Xfr':tf.constant(Xfarr.numpy()[:nb],tf.float32),
        'Xfz':tf.constant(Xfarz.numpy()[:nb],tf.float32),
        'Xbot':tf.constant(Xbottom.numpy()[:nb],tf.float32),

        'Ya':tf.constant(Yaxis_np[:na],tf.float32),
        'aidx':tf.constant(AxisTimeIdx_np[:na],tf.int32),
    }

    print(
        f"built {stage['name']}:",
        'tnodes=',data['tnodes'].shape,
        'liquid=',data['Xl'].shape,
        'vapor=',data['Yv'].shape,
        'interface=',data['Gi'].shape,
        'boundary=',data['Xfib'].shape,
        'axis=',data['Ya'].shape
    )
    return data


def refresh_static_stage_liquid(data,verbose=False):
    """Refresh only the fixed-size active liquid cloud."""
    n=int(data['active_n'])
    data['Xl'].assign(liquid_cloud_for_active_nodes(n))
    data['Xl0'].assign(
        sampler.liquid_at_time(0.0,cfg.n_liquid_t0_fixed,margin=0.01)
    )
    if verbose:
        print('refreshed static liquid:',data['name'],data['Xl'].shape)


def make_static_train_step(data):
    """
    Compile a dedicated graph for this exact stage shape.
    No active_n Tensor argument and no variable-length slicing occur here.
    """
    tnodes=data['tnodes']
    Xl=data['Xl']
    Xl0=data['Xl0']
    Yv=data['Yv']
    vidx=data['vidx']
    Gi=data['Gi']
    iidx=data['iidx']
    Xfib=data['Xfib']
    Xtp=data['Xtp']
    tipidx=data['tipidx']
    Xfr=data['Xfr']
    Xfz=data['Xfz']
    Xbot=data['Xbot']
    Ya=data['Ya']
    aidx=data['aidx']

    nt=int(data['active_n'])
    nq=int(cfg.n_volume_xi)

    @tf.function(reduce_retracing=True)
    def train_step():
        Xv=map_vapor_reference(Yv)
        Xv0=map_vapor_t0(Yv0)

        with tf.GradientTape() as tape:
            S=thermo_time_state(tnodes)

            # Liquid PDE: fixed [nl,3] stage variable.
            Tl=liquid_terms(Xl)
            mlr0,mlz0=liquid_t0_momentum_terms(Xl0)

            # Vapor PDE: fixed [nv,3] tensors.
            rhoV=tf.gather(S['rho'],vidx)
            gVV=tf.gather(S['gV'],vidx)
            LvV=tf.gather(S['Lv'],vidx)
            ZcV=tf.gather(S['Zc'],vidx)
            LvtV=tf.gather(S['Lvt'],vidx)
            ZctV=tf.gather(S['Zct'],vidx)
            Tv=vapor_terms(Xv,rhoV,gVV,LvV,ZcV,LvtV,ZctV)

            Lv0=tf.repeat(S['Lv'][0:1],repeats=cfg.n_vapor_t0_fixed,axis=0)
            Zc0=tf.repeat(S['Zc'][0:1],repeats=cfg.n_vapor_t0_fixed,axis=0)
            mvr0,mvz0=vapor_t0_momentum_terms(Xv0,Lv0,Zc0)

            Jml=mse(Tl['mr'])+mse(Tl['mz'])
            Jld=mse(Tl['div'])  # diagnostic only
            Jml0=mse(mlr0)+mse(mlz0)
            Jcv=mse(Tv['cont'])
            Jmv=mse(Tv['mr'])+mse(Tv['mz'])
            Jmv0=mse(mvr0)+mse(mvz0)

            Ji,Jdu_r,Jdu_z,Jkin,JtanGauge,Jnormal,Jtang=weighted_interface_loss(
                Gi,iidx,S
            )
            Ji0=initial_interface_traction_loss(S)

            # Solid BCs -- all fixed shapes.
            ulr,ulz,_,_=model.liquid_state(Xfib)
            Lvtip=tf.gather(S['Lv'],tipidx)
            Zctip=tf.gather(S['Zc'],tipidx)
            uvr,uvz,_=model.vapor_state_scaled(Xtp,Lvtip,Zctip)
            Jfiber=mse(ulr)+mse(ulz)
            Jtip=mse(uvr)+mse(uvz)

            # Far traction -- all fixed shapes.
            ar,az=liquid_boundary_traction(Xfr,1.0,0.0)
            br,bz=liquid_boundary_traction(Xfz,0.0,1.0)
            cr,cz=liquid_boundary_traction(Xbot,0.0,-1.0)
            Jfar=(mse(ar)+mse(az)+mse(br)+mse(bz)+mse(cr)+mse(cz))/3.0

            # Gauge on every active fixed time node.
            qbars=qbar_from_state(tnodes,S['Lv'],S['Zc'])
            Jq=mse(qbars)

            # Vapor axis -- fixed stage shape.
            ss,t=Ya[:,0:1],Ya[:,1:2]
            xi1=tf.ones_like(t)
            _,ztop=model.interface(xi1,t)
            r=tf.zeros_like(t)
            z=tf.stop_gradient(ss*ztop)
            Lva=tf.gather(S['Lv'],aidx)
            Zca=tf.gather(S['Zc'],aidx)
            with tf.GradientTape(persistent=True) as atape:
                atape.watch(r)
                ura,uza,qa=model.vapor_state_scaled(
                    tf.concat([r,z,t],axis=1),Lva,Zca
                )
            uza_r=atape.gradient(uza,r)
            qa_r=atape.gradient(qa,r)
            del atape
            Jaxis=mse(ura)+mse(uza_r)+mse(qa_r)

            # Geometry regularization with Python-static nt/nq.
            xi=tf.tile(VOL_XI,[nt,1])
            tt=tf.reshape(
                tf.repeat(tnodes,repeats=nq,axis=0),
                [nt*nq,1]
            )
            with tf.GradientTape() as gtape:
                gtape.watch(xi)
                rg,zg=model.interface(xi,tt)
            zxi=gtape.gradient(zg,xi)
            Jgeom=mse(tf.nn.relu(1.0e-4-zxi))

            loss=(
                Jml+Jml0+Jcv+Jmv+Jmv0+Ji+Ji0+
                Jfiber+Jtip+Jfar+Jq+Jaxis+Jgeom
            )

        grads=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients([
            (g,v) for g,v in zip(grads,model.trainable_variables)
            if g is not None
        ])

        return (
            loss,Jml,Jld,Jml0,Jcv,Jmv,Jmv0,Ji,Ji0,
            Jfiber,Jtip,Jfar,Jq,Jaxis,Jgeom,
            Jdu_r,Jdu_z,Jkin,JtanGauge,Jnormal,Jtang,
            tf.reduce_max(tf.abs(qbars)),S['ratio'][-1,0]
        )

    return train_step


def append_history_to_report(stage,data,history,compile_seconds):
    with REPORT_FILE.open('a',encoding='utf-8') as f:
        f.write(f'\nTRAINING HISTORY {stage["name"]}\n')
        f.write(
            f'cutoff={stage["cutoff"]:.6f} '
            f'active_nodes={stage["active_n"]} '
            f'epochs={stage["epochs"]} '
            f'compile_first_step_s={compile_seconds:.6f}\n'
        )
        f.write('effective_epoch '+' '.join(TRAIN_NAMES)+' elapsed_stage_s\n')
        for epoch,row,elapsed in history:
            f.write(
                str(epoch)+' '+
                ' '.join(f'{v:.9e}' for v in row)+
                f' {elapsed:.3f}\n'
            )


def fit_static_stage(stage,data,train_step,effective_offset):
    epochs=int(stage['epochs'])
    refresh_every=int(cfg.refresh_every)

    print('\n'+'='*80)
    print(
        'STATIC',stage['name'].upper(),
        'cutoff=',stage['cutoff'],
        'active_nodes=',stage['active_n'],
        'epochs=',epochs
    )
    print('='*80)
    print(' '.join(TRAIN_NAMES))

    history=[]
    stage_start=time.perf_counter()

    # First update includes tracing/compilation; log it separately.
    compile_start=time.perf_counter()
    vals=train_step()
    compile_seconds=time.perf_counter()-compile_start

    row=[float(v.numpy()) for v in vals]
    history.append((effective_offset,row,time.perf_counter()-stage_start))
    print(
        effective_offset,stage['name'],
        *[f'{v:.3e}' for v in row],
        f'first_step_with_compile={compile_seconds:.3f}s'
    )

    mark=time.perf_counter()
    previous_it=0

    for it in range(1,epochs):
        if refresh_every>0 and it%refresh_every==0:
            rs=time.perf_counter()
            refresh_static_stage_liquid(data,verbose=False)
            refresh_s=time.perf_counter()-rs
            print(
                f'--- static liquid refresh stage_epoch={it} '
                f'time={refresh_s:.3f}s ---'
            )

        vals=train_step()

        if it%250==0 or it==epochs-1:
            now=time.perf_counter()
            row=[float(v.numpy()) for v in vals]
            nstep=it-previous_it
            step_s=(now-mark)/max(1,nstep)
            elapsed=now-stage_start
            effective_epoch=effective_offset+it
            history.append((effective_epoch,row,elapsed))
            print(
                effective_epoch,stage['name'],
                *[f'{v:.3e}' for v in row],
                f'step={step_s:.4f}s',
                f'elapsed_stage={elapsed:.1f}s'
            )
            mark=now
            previous_it=it

    append_history_to_report(stage,data,history,compile_seconds)
    return history

# %% [notebook code cell 6]

# -----------------------------------------------------------------------------
# Initialize the self-contained report BEFORE expensive training.
# Each completed stage appends its training history immediately.
# -----------------------------------------------------------------------------
with REPORT_FILE.open('w',encoding='utf-8') as f:
    f.write('POINT 16 PRODUCTION FIBER PINN -- STATIC-SHAPE CAUSAL FROM SCRATCH\n')
    f.write('Starts from scratch. No model checkpoint or optimizer checkpoint is loaded.\n')
    f.write('One persistent Adam optimizer is used through Stage 1 -> Stage 2 -> Stage 3.\n')
    f.write('No tensor-dependent slicing inside compiled train_step graphs.\n')
    f.write('Liquid PDE and liquid_state both use hardtime=t/0.1.\n')
    f.write('Stage 3 explicitly includes all 97 Chebyshev nodes including t=0.3.\n\n')
    f.write('CONFIG\n')
    f.write(f'tmax={cfg.tmax} lr={cfg.lr} beta1={cfg.beta1} seed={cfg.seed}\n')
    f.write(f'hard_time_scale={cfg.hard_time_scale}\n')
    f.write(f'n_time_final={cfg.n_time_slices} '
            f'n_liquid_per_time={cfg.n_liquid_per_time} '
            f'n_vapor_per_nonzero_time={cfg.n_vapor_per_time} '
            f'n_interface_per_time={cfg.n_interface_per_time}\n')
    f.write(f'trainable_parameters={sum(int(np.prod(v.shape)) for v in model.trainable_variables)}\n')
    f.write('checkpoint_loaded=NONE\n')
    for st in STAGES:
        f.write(
            f'{st["name"]} cutoff={st["cutoff"]:.6f} '
            f'active_nodes={st["active_n"]} epochs={st["epochs"]}\n'
        )
    f.write(f'effective_total_epochs={sum(int(st["epochs"]) for st in STAGES)}\n')

print('\nFROM-SCRATCH INITIAL STATE')
for tv in [0.00,0.05,0.10]:
    O=geometry_observables(tf.constant([[tv]],tf.float32))
    print(
        f't={tv:.3f}',
        f'V/V0={float(O["ratio"].numpy()[0,0]):.9f}',
        f'Req={float(O["Req"].numpy()[0,0]):.9f}',
        f'gV={float(O["gV"].numpy()[0,0]):+.6e}'
    )

all_stage_history={}
effective_offset=0

for stage in STAGES:
    # Proactive Stage-3 graph-shape cleanup. The historical notebook needed a
    # recovery cell here after TensorFlow cached nested tf.function shapes.
    if stage['name'] == 'stage3_t030':
        _nested_helpers = [
            'interface_kinematics','thermo_time_state','map_vapor_reference',
            'map_vapor_t0','qbar_from_state','liquid_terms',
            'liquid_t0_momentum_terms','vapor_terms','vapor_t0_momentum_terms',
            'interface_residuals','weighted_interface_loss',
            'initial_interface_traction_loss','liquid_boundary_traction',
        ]
        for _name in _nested_helpers:
            _fn = globals()[_name]
            if hasattr(_fn, 'python_function'):
                globals()[_name] = _fn.python_function

    # Build fixed-shape stage data only when the stage begins.
    # This lets the moving-domain liquid cloud reflect the geometry learned
    # by the previous stage while keeping graph shapes static.
    data=build_static_stage(stage)
    train_step=make_static_train_step(data)

    hist=fit_static_stage(
        stage,
        data,
        train_step,
        effective_offset=effective_offset
    )
    all_stage_history[stage['name']]=hist
    effective_offset += int(stage['epochs'])

    ckpt=Path(f'point16_causal_STATIC_fromScratch_{stage["name"]}.weights.h5')
    model.built = True
    model.save_weights(str(ckpt))
    print('saved:',ckpt)

    with REPORT_FILE.open('a',encoding='utf-8') as f:
        f.write(f'checkpoint_after_{stage["name"]}={ckpt.name}\n')

    # Diagnostic-only stage state; never used to alter training.
    print('\nSTAGE STATE CHECK:',stage['name'])
    check_times=[0.05,0.10]
    if stage['cutoff'] >= 0.20:
        check_times += [0.15,0.20]
    if stage['cutoff'] >= 0.30:
        check_times += [0.25,0.30]
    for tv in check_times:
        O=geometry_observables(tf.constant([[tv]],tf.float32))
        print(
            f't={tv:.3f}',
            f'V/V0={float(O["ratio"].numpy()[0,0]):.9f}',
            f'Req={float(O["Req"].numpy()[0,0]):.9f}',
            f'gV={float(O["gV"].numpy()[0,0]):+.6e}'
        )

    del train_step
    del data

print('\nSTATIC CAUSAL FROM-SCRATCH TRAINING COMPLETE')
print('effective model updates =',effective_offset)

# %% [notebook code cell 7]
# -----------------------------------------------------------------------------
# Final diagnostics after STATIC Stage 2 -> Stage 3 training.
# -----------------------------------------------------------------------------
def interface_snapshot(t,n=401):
    xi=np.linspace(0.0,1.0,n,dtype=np.float32)[:,None]
    tt=np.full_like(xi,float(t))
    r,z=model.interface(tf.constant(xi),tf.constant(tt))
    return xi.reshape(-1),r.numpy().reshape(-1),z.numpy().reshape(-1)

def qbar_single_time(t):
    return float(qbar_at_nodes(tf.constant([[float(t)]],tf.float32)).numpy()[0])

def scalar_diagnostics(times=(0.0,0.025,0.05,0.075,0.10,0.15,0.20,0.25,0.30)):
    print('\nSCALAR GEOMETRY / THERMODYNAMICS')
    rows=[]
    xi0,r0,z0=interface_snapshot(0.0,401)
    for tv in times:
        tt=tf.constant([[tv]],tf.float32); O=geometry_observables(tt)
        vals={k:float(v.numpy()[0,0]) for k,v in O.items()}
        xi,r,z=interface_snapshot(tv,401)
        disp=float(np.max(np.sqrt((r-r0)**2+(z-z0)**2)))
        row=dict(t=float(tv),**vals,rmaxG=float(np.max(r)),ztop=float(z[-1]),qbar=qbar_single_time(tv),maxDisp=disp)
        rows.append(row)
        print(f"t={tv:7.4f} V/V0={vals['ratio']:.8f} Req={vals['Req']:.8f} Zc={vals['Zc']:.8f} "
              f"Pv={vals['Pv']:+.8f} rho={vals['rho']:.8f} gV={vals['gV']:+.6e} "
              f"rmaxG={row['rmaxG']:.6f} ztop={row['ztop']:.6f} maxDisp={disp:.3e} qbar={row['qbar']:+.3e}")
    return rows

scalar_rows=scalar_diagnostics()

# -----------------------------------------------------------------------------
# Fresh-point validation, independent of training clouds. Uses the same corrected hard-time liquid PDE.
# -----------------------------------------------------------------------------
rng_val=np.random.default_rng(20260831)

def fresh_liquid_points(t,n=2500):
    return sampler.liquid_at_time(float(t),int(n),margin=0.01)

def fresh_vapor_reference(t,n=2500):
    U=rng_val.random((int(n),2))
    return np.c_[0.005+0.985*U[:,0],0.005+0.99*U[:,1],np.full(int(n),float(t))].astype(np.float32)

def fresh_interface(t,n=1200):
    xi=np.linspace(0.002,0.998,int(n),dtype=np.float32)
    return np.c_[xi,np.full(int(n),float(t))].astype(np.float32)

def validate_time(t):
    # Liquid.
    XL=tf.constant(fresh_liquid_points(t)); L=liquid_terms(XL)

    # Vapor at one scalar time: compute geometry scale/time derivatives once, then tile.
    Y=tf.constant(fresh_vapor_reference(t)); XV=map_vapor_reference(Y)
    st=thermo_time_state(tf.constant([[float(t)]],tf.float32)); nv=tf.shape(XV)[0]
    rho=tf.repeat(st['rho'],nv,axis=0); gV=tf.repeat(st['gV'],nv,axis=0)
    Lv=tf.repeat(st['Lv'],nv,axis=0); Zc=tf.repeat(st['Zc'],nv,axis=0)
    Lvt=tf.repeat(st['Lvt'],nv,axis=0); Zct=tf.repeat(st['Zct'],nv,axis=0)
    V=vapor_terms(XV,rho,gV,Lv,Zc,Lvt,Zct)

    # Interface.
    G=tf.constant(fresh_interface(t)); nt=tf.shape(G)[0]
    Sone={k:tf.repeat(v,nt,axis=0) if k not in ['V','ratio','Vt','gV'] else v for k,v in st.items()}
    # interface_residuals expects node arrays + integer indices; use a one-node dictionary.
    Sif={k:v for k,v in st.items()}
    idx=tf.zeros((tf.shape(G)[0],),tf.int32)
    I=interface_residuals(G,idx,Sif)

    row={
        't':float(t),'liq_div':rms_np(L['div'].numpy()),'liq_mom_r':rms_np(L['mr'].numpy()),'liq_mom_z':rms_np(L['mz'].numpy()),
        'vap_cont':rms_np(V['cont'].numpy()),'vap_mom_r':rms_np(V['mr'].numpy()),'vap_mom_z':rms_np(V['mz'].numpy()),
        'if_du_r':rms_np(I['du_r'].numpy()),'if_du_z':rms_np(I['du_z'].numpy()),'if_kin':rms_np(I['kin'].numpy()),
        'if_tanGauge':rms_np(I['tangauge'].numpy()),'if_normal':rms_np(I['normal'].numpy()),'if_tangent':rms_np(I['tangential'].numpy())
    }
    print(f'\nFRESH VALIDATION t={t:.3f}')
    print(' liquid RMS: div=',row['liq_div'],'mom-r=',row['liq_mom_r'],'mom-z=',row['liq_mom_z'])
    print(' vapor RMS: cont=',row['vap_cont'],'mom-r=',row['vap_mom_r'],'mom-z=',row['vap_mom_z'])
    print(' interface RMS: du_r=',row['if_du_r'],'du_z=',row['if_du_z'],'kin=',row['if_kin'],
          'tanGauge=',row['if_tanGauge'],'normal=',row['if_normal'],'tangent=',row['if_tangent'])
    return row

fresh_rows=[validate_time(tv) for tv in [0.025,0.05,0.10,0.15,0.20,0.25,0.30]]

# Explicit initial-time checks that replace the unavailable spherical p0 aid.
mlr0,mlz0=liquid_t0_momentum_terms(Xliquid0)
Sall=thermo_time_state(Tnodes)
Ji0_now=float(initial_interface_traction_loss(Sall).numpy())
qbars_now=qbar_at_nodes(Tnodes).numpy()
initial_checks={
    'liq_t0_mom_r':rms_np(mlr0.numpy()),'liq_t0_mom_z':rms_np(mlz0.numpy()),
    'Ji0':Ji0_now,'qbar_max':float(np.max(np.abs(qbars_now)))
}
print('\nINITIAL / GAUGE CHECKS',initial_checks)


# %% [notebook code cell 8]

# -----------------------------------------------------------------------------
# Append final diagnostics to the already-populated self-contained report.
# -----------------------------------------------------------------------------
with REPORT_FILE.open('a',encoding='utf-8') as f:
    f.write('\nSCALAR GEOMETRY / THERMODYNAMICS\n')
    f.write('t Vratio Req Zcent Pv rho gV rmaxG ztop maxDisp qbar\n')
    for r in scalar_rows:
        f.write(
            f"{r['t']:.6f} {r['ratio']:.9e} {r['Req']:.9e} "
            f"{r['Zc']:.9e} {r['Pv']:.9e} {r['rho']:.9e} "
            f"{r['gV']:.9e} {r['rmaxG']:.9e} {r['ztop']:.9e} "
            f"{r['maxDisp']:.9e} {r['qbar']:.9e}\n"
        )

    f.write('\nFRESH VALIDATION RMS\n')
    keys=[
        't','liq_div','liq_mom_r','liq_mom_z',
        'vap_cont','vap_mom_r','vap_mom_z',
        'if_du_r','if_du_z','if_kin','if_tanGauge',
        'if_normal','if_tangent'
    ]
    f.write(' '.join(keys)+'\n')
    for r in fresh_rows:
        f.write(' '.join(f'{r[k]:.9e}' for k in keys)+'\n')

    f.write('\nINITIAL / GAUGE CHECKS\n')
    for k,v in initial_checks.items():
        f.write(f'{k} {v:.9e}\n')

    f.write('\nDENSE INTERFACE SAMPLES\n')
    f.write('t xi r z\n')
    for tv in [0.025,0.05,0.10,0.15,0.20,0.25,0.30]:
        xi_dense,r_dense,z_dense=interface_snapshot(tv,101)
        for xx,rr,zz in zip(xi_dense,r_dense,z_dense):
            f.write(f'{tv:.6f} {xx:.9e} {rr:.9e} {zz:.9e}\n')

# Diagnostic-only reference to the independent Point-16 fine short-horizon run.
POINT16_FINE_T010_REF={
    'ratio':9.814348221e-01,
    'Req':9.913324118e-01,
    'Zc':9.053351283e-01,
    'Pv':-9.952709079e-01,
    'rho':2.037832700e-02,
    'gV':-3.803551793e-01,
    'rmaxG':9.935166836e-01,
    'ztop':1.892302990e+00,
    'maxDisp':7.349455729e-03,
}
row010=min(scalar_rows,key=lambda r:abs(r['t']-0.10))

with REPORT_FILE.open('a',encoding='utf-8') as f:
    f.write('\nPOINT16_FINE_T010_REFERENCE_COMPARISON_DIAGNOSTIC_ONLY\n')
    f.write('quantity static_causal_fromScratch point16_fine_ref relative_difference\n')
    for k in ['ratio','Req','Zc','Pv','rho','gV','rmaxG','ztop','maxDisp']:
        a=float(row010[k]); b=float(POINT16_FINE_T010_REF[k])
        rel=abs(a-b)/(abs(b)+1e-14)
        f.write(f'{k} {a:.9e} {b:.9e} {rel:.9e}\n')

final_weights=Path('point16_fine_t030_causal_STATIC_fromScratch.weights.h5')
model.save_weights(str(final_weights))

print('saved final weights:',final_weights)
print('wrote self-contained report:',REPORT_FILE)
print('UPLOAD ONLY:',REPORT_FILE.name)

# %% [notebook code cell 9]

