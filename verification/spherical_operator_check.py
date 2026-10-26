#!/usr/bin/env python3
# Exact spherical operator verification
# Exported from a cleaned Jupyter notebook; notebook outputs are not embedded.

# %% [notebook code cell 2]
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.stats import qmc

np.random.seed(1234)
tf.random.set_seed(1234)

mse = lambda x: tf.reduce_mean(tf.square(x))

def weighted_mse(x, w, eps=1e-12):
    w=tf.stop_gradient(tf.cast(w,x.dtype))
    return tf.reduce_sum(w*tf.square(x))/(tf.reduce_sum(w)+eps)

def masked_mse(x, mask, eps=1e-12):
    mask=tf.stop_gradient(tf.cast(mask,x.dtype))
    return tf.reduce_sum(mask*tf.square(x))/(tf.reduce_sum(mask)+eps)

def np_rms(x):
    x=np.asarray(x)
    return float(np.sqrt(np.mean(x*x))) if x.size else np.nan

def corr_np(a,b):
    a=np.asarray(a).reshape(-1); b=np.asarray(b).reshape(-1)
    if a.size<2 or np.std(a)==0 or np.std(b)==0:
        return np.nan
    return float(np.corrcoef(a,b)[0,1])

# %% [notebook code cell 3]
class BenchmarkConfig:
    def __init__(self):
        # Same nondimensional physical parameters as the attached model.
        self.Re=500.0
        self.Pi=1.25
        self.pv0=-1.0
        self.pinf=0.0
        self.gamma_v=1.0
        self.rho_v_ratio=2.0e-2
        self.mu_v_ratio=2.0e-2
        self.abs_pressure_floor=1e-5

        # Half-domain spherical benchmark. Start with collapse phase only.
        self.rmin=0.0; self.rmax=4.0
        self.zmin=0.0; self.zmax=4.0
        self.tmin=0.0; self.tmax=0.90
        # After a benchmark passes at 0.90, raise tmax to 1.20 to include rebound.

        self.axis_eps=2e-3
        self.pde_interface_halfwidth=0.01
        self.lr=2e-4
        self.beta1=0.99
        self.seed=1234

cfg=BenchmarkConfig()

class RPReference:
    """High-accuracy Rayleigh-Plesset reference plus differentiable cubic-spline evaluation."""
    def __init__(self,cfg,n_knots=2401):
        self.cfg=cfg
        def pv_R(R):
            return (cfg.Pi+cfg.pv0)*R**(-3.0*cfg.gamma_v)-cfg.Pi
        def rhs(t,y):
            R,V=y
            P=pv_R(R)
            A=(P-cfg.pinf-1.5*V*V-(4.0/cfg.Re)*V/R)/R
            return [V,A]
        sol=solve_ivp(rhs,(cfg.tmin,cfg.tmax),[1.0,0.0],rtol=1e-11,atol=1e-13,
                      max_step=max(2e-4,cfg.tmax/(n_knots-1)))
        if not sol.success:
            raise RuntimeError(sol.message)
        x=np.linspace(cfg.tmin,cfg.tmax,n_knots)
        y=sol.sol(x) if sol.sol is not None else None
        if y is None:
            # Re-solve with dense output if a SciPy build did not retain it.
            sol=solve_ivp(rhs,(cfg.tmin,cfg.tmax),[1.0,0.0],rtol=1e-11,atol=1e-13,
                          max_step=max(2e-4,cfg.tmax/(n_knots-1)),dense_output=True)
            y=sol.sol(x)
        R=y[0]
        self.cs=CubicSpline(x,R,bc_type='not-a-knot')
        self.x=x.astype(np.float64)
        self.c=self.cs.c.astype(np.float64)  # c0 dx^3 + c1 dx^2 + c2 dx + c3
        self.x_tf=tf.constant(self.x,dtype=tf.float32)
        self.c_tf=tf.constant(self.c,dtype=tf.float32)

    def pv_np(self,R):
        c=self.cfg
        return (c.Pi+c.pv0)*np.asarray(R)**(-3.0*c.gamma_v)-c.Pi

    def rva_np(self,t):
        t=np.asarray(t)
        return self.cs(t), self.cs(t,1), self.cs(t,2)

    def _interval_tf(self,t):
        flat=tf.reshape(tf.cast(t,tf.float32),[-1])
        idx=tf.searchsorted(self.x_tf[1:],flat,side='right')
        idx=tf.clip_by_value(idx,0,tf.shape(self.x_tf)[0]-2)
        dx=flat-tf.gather(self.x_tf,idx)
        cc=tf.gather(self.c_tf,idx,axis=1)
        return flat,idx,dx,cc,tf.shape(t)

    def rva_tf(self,t):
        flat,idx,dx,cc,shape=self._interval_tf(t)
        c0,c1,c2,c3=cc[0],cc[1],cc[2],cc[3]
        R=((c0*dx+c1)*dx+c2)*dx+c3
        V=(3.0*c0*dx+2.0*c1)*dx+c2
        A=6.0*c0*dx+2.0*c1
        return tf.reshape(R,shape),tf.reshape(V,shape),tf.reshape(A,shape)

    def pv_tf(self,R):
        c=self.cfg
        return (c.Pi+c.pv0)*tf.pow(R,-3.0*c.gamma_v)-c.Pi

    def interface_pressure_tf(self,t):
        R,V,_=self.rva_tf(t)
        return self.pv_tf(R)-(4.0/self.cfg.Re)*V/R

    def interface_pressure_np(self,t):
        R,V,_=self.rva_np(t)
        return self.pv_np(R)-(4.0/self.cfg.Re)*V/R

    def fields_tf(self,r,z,t):
        R,V,A=self.rva_tf(t)
        s=tf.sqrt(r*r+z*z+tf.cast(1e-12,r.dtype))
        C=R*R*V
        Cdot=2.0*R*V*V+R*R*A
        ur=C*r/tf.pow(s,3)
        uz=C*z/tf.pow(s,3)
        p=Cdot/s-0.5*C*C/tf.pow(s,4)
        return ur,uz,p

    def fields_np(self,r,z,t):
        r=np.asarray(r); z=np.asarray(z); t=np.asarray(t)
        R,V,A=self.rva_np(t)
        s=np.sqrt(r*r+z*z+1e-30)
        C=R*R*V
        Cdot=2.0*R*V*V+R*R*A
        ur=C*r/s**3
        uz=C*z/s**3
        p=Cdot/s-0.5*C*C/s**4
        return ur,uz,p

    def psi_tf(self,r,z,t):
        R,V,_=self.rva_tf(t)
        s=tf.sqrt(r*r+z*z+tf.cast(1e-12,r.dtype))
        C=R*R*V
        return C*(1.0-z/s)

ref=RPReference(cfg)

print('Rayleigh-Plesset reference:')
for tt in [0.0,0.25,0.60,min(0.90,cfg.tmax)]:
    if tt<=cfg.tmax:
        R,V,A=ref.rva_np(tt)
        print(f't={tt:4.2f}  R={R:.6f}  Rdot={V:.6f}  V/V0={R**3:.6f}  Pv={ref.pv_np(R):+.6f}')

# %% [notebook code cell 4]
class BenchmarkSampler:
    def __init__(self,cfg,ref):
        self.cfg=cfg; self.ref=ref; self.seed=cfg.seed

    def sobol(self,n,d):
        m=int(np.ceil(np.log2(max(2,int(n)))))
        s=qmc.Sobol(d=d,scramble=True,seed=self.seed)
        self.seed+=1
        return s.random_base2(m)[:int(n)]

    def liquid(self,n,margin=0.02):
        c=self.cfg; out=[]; need=int(n)
        while need>0:
            U=self.sobol(max(2048,int(1.5*need)),3)
            r=c.axis_eps+(c.rmax-c.axis_eps)*U[:,0]
            z=c.zmin+(c.zmax-c.zmin)*U[:,1]
            t=c.tmin+(c.tmax-c.tmin)*U[:,2]
            R=self.ref.rva_np(t)[0]
            keep=np.sqrt(r*r+z*z)>R+margin
            X=np.c_[r[keep],z[keep],t[keep]]
            take=min(need,len(X))
            if take:
                out.append(X[:take]); need-=take
        return np.vstack(out).astype(np.float32)

    def box(self,n):
        c=self.cfg; U=self.sobol(n,3)
        return np.c_[c.axis_eps+(c.rmax-c.axis_eps)*U[:,0],
                     c.zmin+(c.zmax-c.zmin)*U[:,1],
                     c.tmin+(c.tmax-c.tmin)*U[:,2]].astype(np.float32)

    def interface(self,n):
        c=self.cfg; U=self.sobol(n,2)
        t=c.tmin+(c.tmax-c.tmin)*U[:,0]
        th=0.5*np.pi*U[:,1]
        R=self.ref.rva_np(t)[0]
        return np.c_[R*np.sin(th),R*np.cos(th),t].astype(np.float32)

    def far_r(self,n):
        c=self.cfg; U=self.sobol(n,2)
        return np.c_[np.full(n,c.rmax), c.zmin+(c.zmax-c.zmin)*U[:,0],
                     c.tmin+(c.tmax-c.tmin)*U[:,1]].astype(np.float32)

    def far_top(self,n):
        c=self.cfg; U=self.sobol(n,2)
        return np.c_[c.axis_eps+(c.rmax-c.axis_eps)*U[:,0], np.full(n,c.zmax),
                     c.tmin+(c.tmax-c.tmin)*U[:,1]].astype(np.float32)

    def equator(self,n,margin=0.02):
        c=self.cfg; U=self.sobol(n,2)
        t=c.tmin+(c.tmax-c.tmin)*U[:,0]
        R=self.ref.rva_np(t)[0]
        r=R+margin+(c.rmax-R-margin)*U[:,1]
        return np.c_[r,np.zeros(n),t].astype(np.float32)

    def axis(self,n,margin=0.02):
        c=self.cfg; U=self.sobol(n,2)
        t=c.tmin+(c.tmax-c.tmin)*U[:,0]
        R=self.ref.rva_np(t)[0]
        z=R+margin+(c.zmax-R-margin)*U[:,1]
        return np.c_[np.zeros(n),z,t].astype(np.float32)

sampler=BenchmarkSampler(cfg,ref)

# %% [notebook code cell 5]
@tf.function
def exact_operator_terms(X):
    r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([r,z])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([r,z,t])
            ur,uz,p=ref.fields_tf(r,z,t)
        ur_r=tape1.gradient(ur,r); ur_z=tape1.gradient(ur,z); ur_t=tape1.gradient(ur,t)
        uz_r=tape1.gradient(uz,r); uz_z=tape1.gradient(uz,z); uz_t=tape1.gradient(uz,t)
        p_r=tape1.gradient(p,r); p_z=tape1.gradient(p,z)
        del tape1
    ur_rr=tape2.gradient(ur_r,r); ur_zz=tape2.gradient(ur_z,z)
    uz_rr=tape2.gradient(uz_r,r); uz_zz=tape2.gradient(uz_z,z)
    del tape2
    rs=tf.maximum(r,tf.cast(cfg.axis_eps,r.dtype))
    div=ur_r+ur/rs+uz_z
    adv_r=ur_t+ur*ur_r+uz*ur_z
    adv_z=uz_t+ur*uz_r+uz*uz_z
    lap_r=ur_rr+ur_r/rs-ur/(rs*rs)+ur_zz
    lap_z=uz_rr+uz_r/rs+uz_zz
    mom_r=adv_r+p_r-lap_r/cfg.Re
    mom_z=adv_z+p_z-lap_z/cfg.Re
    return div,adv_r,adv_z,p_r,p_z,lap_r,lap_z,mom_r,mom_z

X=tf.convert_to_tensor(sampler.liquid(40000),tf.float32)
vals=[v.numpy().reshape(-1) for v in exact_operator_terms(X)]
labels=['div','inertia_r','inertia_z','p_r','p_z','lap_r','lap_z','mom_r','mom_z']
for k,v in zip(labels,vals):
    print(f'{k:10s} RMS = {np_rms(v):.6e}')
print('corr(inertia_r,p_r)=',corr_np(vals[1],vals[3]))
print('corr(inertia_z,p_z)=',corr_np(vals[2],vals[4]))

# %% [notebook code cell 6]
# Exact t=0 pressure/acceleration cancellation.
U=sampler.sobol(20000,2)
r=cfg.axis_eps+(cfg.rmax-cfg.axis_eps)*U[:,0]
z=1.02+(cfg.zmax-1.02)*U[:,1]
X0=tf.convert_to_tensor(np.c_[r,z,np.zeros_like(r)].astype(np.float32))
_,ar,az,pr,pz,_,_,rr,rz=[v.numpy().reshape(-1) for v in exact_operator_terms(X0)]
print('t=0 corr(a_r,p_r) =',corr_np(ar,pr))
print('t=0 corr(a_z,p_z) =',corr_np(az,pz))
print('t=0 momentum RMS r/z =',np_rms(rr),np_rms(rz))
