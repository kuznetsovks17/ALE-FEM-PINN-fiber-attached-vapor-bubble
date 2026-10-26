#!/usr/bin/env python3
# Spherical G3 reference PINN
# Exported from a cleaned Jupyter notebook; notebook outputs are not embedded.

# %% [notebook code cell 2]
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.stats import qmc
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

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
        self.Re=500.0
        self.Pi=1.25
        self.pv0=-1.0
        self.pinf=0.0
        self.gamma_v=1.0
        self.rho_v_ratio=2.0e-2
        self.mu_v_ratio=2.0e-2

        self.rmin=0.0; self.rmax=4.0
        self.zmin=0.0; self.zmax=4.0
        self.tmin=0.0; self.tmax=0.90
        self.R0=1.0

        self.axis_eps=2e-3
        self.pde_interface_halfwidth=0.01
        self.lr=2e-4
        self.beta1=0.99
        self.seed=1234

        self.ff_features=16
        self.ff_scale=1.0
        self.chi_scale=3.0
        self.p_scale=4.0
        self.u_scale=3.0
        self.q_scale=1.0
        self.radius_log_scale=2.0

        # Deterministic full-set training with a PERMANENT Chebyshev--Lobatto
        # time grid.  There is no stochastic minibatching.
        self.n_liquid_fixed=1536
        self.n_time_slices=33

        # Moving-vapor PDE resolution.  t=0 is handled separately, leaving 32
        # nonzero time slices.  Every slice receives the same 64*5 = 320 interior
        # points: 32 * 320 = 10,240 vapor PDE points per optimizer step.
        self.n_vapor_per_time=64*5

        self.n_interface_per_time=32   # 33 slices x 32 = 1056 interface points
        self.n_vapor_t0_fixed=256
        self.n_gauge_spatial=64        # evaluated at every one of the 33 time nodes
        self.n_boundary_1d=200
        self.n_p0_radial=1201

        # Refresh only spatial/reference coordinates.  The time grid and per-slice
        # counts never change, and every epoch evaluates the complete set.
        self.refresh_every=500

        # Interface weighting: normal traction transmits the collapse-driving force.
        self.lambda_normal=10.0

cfg=BenchmarkConfig()


def pv_from_R_np(R,cfg=cfg):
    R=np.asarray(R)
    return (cfg.Pi+cfg.pv0)*R**(-3.0*cfg.gamma_v)-cfg.Pi


def rho_from_R_np(R,cfg=cfg):
    # Closed spherical vapor: rho*R^3 = rho_v0.
    return cfg.rho_v_ratio*np.asarray(R)**(-3.0)


class RPReference:
    """Coupled spherical reference consistent with vapor inertia and full normal traction."""
    def __init__(self,cfg,n_knots=2401):
        self.cfg=cfg
        def rhs(t,y):
            R,V=y
            Pv=pv_from_R_np(R,cfg)
            rho=rho_from_R_np(R,cfg)
            visc=((4.0+2.0*cfg.mu_v_ratio)/cfg.Re)*V/R
            A=(Pv-cfg.pinf-1.5*V*V-visc)/(R*(1.0+rho/5.0))
            return [V,A]
        sol=solve_ivp(rhs,(cfg.tmin,cfg.tmax),[cfg.R0,0.0],rtol=1e-11,atol=1e-13,
                      max_step=max(2e-4,cfg.tmax/(n_knots-1)),dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        x=np.linspace(cfg.tmin,cfg.tmax,n_knots)
        R=sol.sol(x)[0]
        self.cs=CubicSpline(x,R,bc_type='not-a-knot')
        self.x=x.astype(np.float64)
        self.c=self.cs.c.astype(np.float64)
        self.x_tf=tf.constant(self.x,dtype=tf.float32)
        self.c_tf=tf.constant(self.c,dtype=tf.float32)

    def pv_np(self,R): return pv_from_R_np(R,self.cfg)
    def rho_np(self,R): return rho_from_R_np(R,self.cfg)

    def rva_np(self,t):
        t=np.asarray(t)
        return self.cs(t),self.cs(t,1),self.cs(t,2)

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
        return (self.cfg.Pi+self.cfg.pv0)*tf.pow(R,-3.0*self.cfg.gamma_v)-self.cfg.Pi

    def rho_tf(self,R):
        return self.cfg.rho_v_ratio*tf.pow(R,-3.0)

    def interface_pressure_np(self,t):
        R,V,A=self.rva_np(t)
        return self.cfg.pinf+R*A+1.5*V*V

    def fields_tf(self,r,z,t):
        R,V,A=self.rva_tf(t)
        s=tf.sqrt(r*r+z*z+tf.cast(1e-12,r.dtype))
        C=R*R*V
        Cdot=2.0*R*V*V+R*R*A
        ur=C*r/tf.pow(s,3)
        uz=C*z/tf.pow(s,3)
        p=tf.cast(self.cfg.pinf,r.dtype)+Cdot/s-0.5*C*C/tf.pow(s,4)
        return ur,uz,p

    def fields_np(self,r,z,t):
        r=np.asarray(r); z=np.asarray(z); t=np.asarray(t)
        R,V,A=self.rva_np(t); s=np.sqrt(r*r+z*z+1e-30)
        C=R*R*V; Cdot=2.0*R*V*V+R*R*A
        return C*r/s**3,C*z/s**3,self.cfg.pinf+Cdot/s-0.5*C*C/s**4

    def psi_tf(self,r,z,t):
        R,V,_=self.rva_tf(t); s=tf.sqrt(r*r+z*z+tf.cast(1e-12,r.dtype)); C=R*R*V
        return C*(1.0-z/s)

ref=RPReference(cfg)


class InitialLiquidPressure:
    """Solve the t=0 spherical liquid pressure Laplace problem once.

    p'' + 2 p'/s = 0,  s in [R0,s_outer]
    p(R0) = p_Gamma(0),
    p'(s_outer) + (p-p_inf)/s_outer = 0.

    The interface value includes the small low-density vapor-inertia correction
    implied by the zero-mean q_v gauge. At t=0 viscous normal stresses vanish.
    """
    def __init__(self,cfg):
        self.cfg=cfg
        rho0=cfg.rho_v_ratio
        self.a0=(cfg.pv0-cfg.pinf)/(cfg.R0*(1.0+rho0/5.0))
        self.p_gamma0=cfg.pinf+cfg.R0*self.a0
        self.s_outer=float(np.hypot(cfg.rmax,cfg.zmax))
        s=np.linspace(cfg.R0,self.s_outer,cfg.n_p0_radial)
        h=s[1]-s[0]; n=len(s)
        A=lil_matrix((n,n),dtype=np.float64); b=np.zeros(n)
        A[0,0]=1.0; b[0]=self.p_gamma0
        for i in range(1,n-1):
            si=s[i]
            A[i,i-1]=1.0/h**2-1.0/(si*h)
            A[i,i]=-2.0/h**2
            A[i,i+1]=1.0/h**2+1.0/(si*h)
        # second-order backward derivative + asymptotic 1/s Robin condition
        A[-1,-1]=3.0/(2.0*h)+1.0/s[-1]
        A[-1,-2]=-4.0/(2.0*h)
        A[-1,-3]=1.0/(2.0*h)
        b[-1]=cfg.pinf/s[-1]
        p=spsolve(A.tocsr(),b)
        self.s=s; self.p=p
        self.cs=CubicSpline(s,p,bc_type='not-a-knot')
        self.c=self.cs.c.astype(np.float64)
        self.s_tf=tf.constant(self.s.astype(np.float32))
        self.c_tf=tf.constant(self.c.astype(np.float32))
        # Harmonic coefficient of the solved initial pressure. It is used only to anchor t=0 acceleration.
        self.A0=float(np.median(s*(p-cfg.pinf)))

    def tf(self,s):
        # The one-time finite-difference solve above identifies the harmonic coefficient A0.
        # In the spherical benchmark the exact Laplace solution is p0=p_inf+A0/s, so use
        # that equivalent representation inside AD rather than differentiating through a
        # spline/searchsorted graph at every collocation point.
        s=tf.cast(s,tf.float32)
        return tf.cast(self.cfg.pinf,s.dtype)+tf.cast(self.A0,s.dtype)/tf.maximum(s,tf.cast(1e-5,s.dtype))

p0=InitialLiquidPressure(cfg)
cfg.initial_harmonic_amplitude=p0.A0
cfg.initial_acceleration=p0.A0/(cfg.R0**2)

print('Coupled spherical reference:')
for tt in [0.0,0.25,0.60,min(0.90,cfg.tmax)]:
    R,V,A=ref.rva_np(tt)
    print(f't={tt:4.2f}  R={R:.6f}  Rdot={V:.6f}  V/V0={R**3:.6f}  Pv={ref.pv_np(R):+.6f}')
print('initial elliptic pressure:',
      f'a0={p0.a0:+.8f}',f'pGamma0={p0.p_gamma0:+.8f}',f'A0={p0.A0:+.8f}',
      f'max FD-vs-harmonic error={np.max(np.abs(p0.p-(cfg.pinf+p0.A0/p0.s))):.3e}')

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
# -----------------------------------------------------------------------------
# Neural conditioning -- smooth trunks where high derivatives are required
# -----------------------------------------------------------------------------
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
    """Smooth normalized physical coordinates. Used where 2nd/3rd AD derivatives matter."""
    def __init__(self,cfg,radial_even=False,name=None):
        super().__init__(name=name); self.cfg=cfg; self.radial_even=bool(radial_even)
    def call(self,X):
        X=tf.cast(X,tf.float32); r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]
        rn=(r-self.cfg.rmin)/(self.cfg.rmax-self.cfg.rmin)
        zn=(z-self.cfg.zmin)/(self.cfg.zmax-self.cfg.zmin)
        tn=(t-self.cfg.tmin)/(self.cfg.tmax-self.cfg.tmin)
        if self.radial_even: rn=tf.square(rn)
        return tf.concat([rn,zn,tn],axis=1)

class NormalizedTime(tf.keras.layers.Layer):
    def __init__(self,cfg,name=None): super().__init__(name=name); self.cfg=cfg
    def call(self,t):
        t=tf.cast(t,tf.float32)
        return (t-self.cfg.tmin)/(self.cfg.tmax-self.cfg.tmin)

class NormalizedFourierInput(tf.keras.layers.Layer):
    """Fourier features retained only for the primitive vapor field."""
    def __init__(self,cfg,radial_even=False,name=None):
        super().__init__(name=name); self.cfg=cfg; self.radial_even=bool(radial_even)
        self.ff=GaussianFourierBlock(cfg.ff_features,cfg.ff_scale)
    def call(self,X):
        X=tf.cast(X,tf.float32); r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]
        rn=(r-self.cfg.rmin)/(self.cfg.rmax-self.cfg.rmin)
        zn=(z-self.cfg.zmin)/(self.cfg.zmax-self.cfg.zmin)
        tn=(t-self.cfg.tmin)/(self.cfg.tmax-self.cfg.tmin)
        if self.radial_even: rn=tf.square(rn)
        return self.ff(tf.concat([rn,zn,tn],axis=1))

class VolumeScaledFourierInput(tf.keras.layers.Layer):
    """Vapor Fourier features in bubble-volume-scaled spatial coordinates.

    Inputs are (X, Lv), where Lv = R0*(V/V0)^(1/3).  For this spherical benchmark
    Lv=R(t).  The dependence on Lv(t) remains differentiable, so Eulerian time
    derivatives of the vapor fields include the correct chain rule through the
    changing characteristic length.
    """
    def __init__(self,cfg,radial_even=False,name=None):
        super().__init__(name=name); self.cfg=cfg; self.radial_even=bool(radial_even)
        self.ff=GaussianFourierBlock(cfg.ff_features,cfg.ff_scale)
    def call(self,inputs):
        X,Lv=inputs
        X=tf.cast(X,tf.float32); Lv=tf.cast(Lv,tf.float32)
        r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]
        Lsafe=tf.maximum(Lv,tf.cast(1e-5,Lv.dtype))
        rn=r/Lsafe
        zn=z/Lsafe
        tn=(t-self.cfg.tmin)/(self.cfg.tmax-self.cfg.tmin)
        if self.radial_even: rn=tf.square(rn)
        return self.ff(tf.concat([rn,zn,tn],axis=1))


class LiquidScalarNetwork(tf.keras.Model):
    def __init__(self,cfg,output_scale,layers=4,dim=64,bounded=True,name='scalar'):
        super().__init__(name=name); self.output_scale=float(output_scale); self.bounded=bool(bounded)
        # No Fourier features here: chi is differentiated to third order in liquid momentum.
        self.inp=NormalizedInput(cfg,radial_even=True,name=name+'_input')
        self.hidden=[tf.keras.layers.Dense(dim,activation='tanh') for _ in range(layers)]
        self.out=tf.keras.layers.Dense(1,kernel_initializer='zeros',bias_initializer='zeros')
    def call(self,X):
        y=self.inp(tf.cast(X,tf.float32))
        for layer in self.hidden: y=layer(y)
        y=self.out(y)
        if self.bounded: y=tf.tanh(y)
        return tf.cast(self.output_scale,tf.float32)*y

class LiquidPsiModel(tf.keras.Model):
    """Divergence-free learned liquid velocity correction only; pressure is analytical."""
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.chi=LiquidScalarNetwork(cfg,cfg.chi_scale,bounded=True,name='chi_l')
    def call(self,X):
        return self.chi(X)

# %% [notebook code cell 6]
def exact_values(X):
    X=tf.cast(X,tf.float32)
    return ref.fields_tf(X[:,0:1],X[:,1:2],X[:,2:3])

def relative_l2(pred,truth,eps=1e-12):
    return float(np.sqrt(np.mean((pred-truth)**2))/(np.sqrt(np.mean(truth**2))+eps))

# %% [notebook code cell 7]
class VaporNet(tf.keras.Model):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.ffe=VolumeScaledFourierInput(cfg,radial_even=False,name='vapor_ffe')
        self.common=[tf.keras.layers.Dense(60,activation='tanh') for _ in range(3)]
        self.heads=[]
        for name in ['urv','uzv','qv']:
            h=tf.keras.layers.Dense(60,activation='tanh',name=name+'_h')
            o=tf.keras.layers.Dense(1,kernel_initializer='zeros',bias_initializer='zeros',name=name+'_out')
            self.heads.append((name,h,o))
    def call(self,X,Lv):
        x=self.ffe((X,Lv))
        for h in self.common: x=h(x)
        out=[]
        for name,h,o in self.heads:
            y=o(h(x)); out.append(tf.tanh(y) if name!='qv' else y)
        return tf.concat(out,axis=1)

class RadiusNet(tf.keras.Model):
    """Positive radius with hard R(0), Rdot(0), and Poisson-derived Rddot(0)."""
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; self.inp=NormalizedTime(cfg,name='R_input')
        self.hidden=[tf.keras.layers.Dense(32,activation='tanh') for _ in range(2)]
        self.out=tf.keras.layers.Dense(1,kernel_initializer='zeros',bias_initializer='zeros')
    def normalized_raw(self,t):
        x=self.inp(t)
        for h in self.hidden: x=h(x)
        return tf.tanh(self.out(x))
    def call(self,t):
        t=tf.cast(t,tf.float32); tau=self.inp(t)
        a0=tf.cast(self.cfg.initial_acceleration,t.dtype)
        R0=tf.cast(self.cfg.R0,t.dtype)
        # The learned tau^3 term cannot alter R, Rdot or Rddot at t=0.
        f0=0.5*(a0/R0)*tf.square(t-self.cfg.tmin)
        flearn=tf.cast(self.cfg.radius_log_scale,t.dtype)*tf.pow(tau,3)*self.normalized_raw(t)
        return R0*tf.exp(f0+flearn)

class FullSphericalALEPINN(tf.keras.Model):
    def __init__(self,cfg,p0):
        super().__init__(); self.cfg=cfg; self.p0=p0
        self.liquid=LiquidPsiModel(cfg)
        self.vapor=VaporNet(cfg)
        self.rnet=RadiusNet(cfg)

    def radius(self,t): return self.rnet(t)

    def radius_rate(self,t):
        with tf.GradientTape() as tape:
            tape.watch(t); R=self.radius(t)
        return R,tape.gradient(R,t)

    def radius_acceleration(self,t):
        with tf.GradientTape() as tape:
            tape.watch(t); R,Rd=self.radius_rate(t)
        return R,Rd,tape.gradient(Rd,t)

    def sink_dynamics(self,t):
        with tf.GradientTape() as tape:
            tape.watch(t); R,Rd=self.radius_rate(t); C=R*R*Rd
        Cdot=tape.gradient(C,t)
        return R,Rd,C,Cdot

    def sink_scalars(self,t):
        R,Rd,C,_=self.sink_dynamics(t)
        return R,Rd,C

    def sink_velocity(self,r,z,t):
        R,Rd,C,_=self.sink_dynamics(t)
        s=tf.sqrt(r*r+z*z+tf.cast(1e-12,r.dtype))
        se=tf.maximum(s,0.5*tf.stop_gradient(R))
        return C*r/tf.pow(se,3),C*z/tf.pow(se,3),R,Rd,C

    def liquid_correction(self,X):
        X=tf.cast(X,tf.float32); r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([r,z]); inp=tf.concat([r,z,t],axis=1); chi=self.liquid(inp)
        chi_r=tape.gradient(chi,r); chi_z=tape.gradient(chi,z); del tape
        tau=(t-self.cfg.tmin)/(self.cfg.tmax-self.cfg.tmin)
        # O(t^2): the learned correction cannot steal the Poisson-derived initial acceleration.
        urc=-tau*tau*r*chi_z
        uzc=tau*tau*(2.0*chi+r*chi_r)
        return urc,uzc,chi

    def liquid_state(self,X):
        X=tf.cast(X,tf.float32); r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]
        urc,uzc,chi=self.liquid_correction(X)
        R,Rd,C,Cdot=self.sink_dynamics(t)
        s=tf.sqrt(r*r+z*z+tf.cast(1e-12,r.dtype))
        se=tf.maximum(s,0.5*tf.stop_gradient(R))
        urs=C*r/tf.pow(se,3); uzs=C*z/tf.pow(se,3)
        # FINAL PRESSURE SIMPLIFICATION: exact spherical pressure, no pressure NN.
        p=tf.cast(self.cfg.pinf,r.dtype)+Cdot/s-0.5*C*C/tf.pow(s,4)
        return urs+urc,uzs+uzc,p,chi

    def vapor_volume_ratio(self,t):
        """Current V_v(t)/V_v0 used for vapor feature scaling.

        Spherical benchmark: V/V0=(R/R0)^3.  In the nonspherical ALE model,
        replace only this geometry helper with the actual ALE bubble volume ratio.
        """
        R=self.radius(t)
        return tf.pow(R/self.cfg.R0,3.0)

    def vapor_length_scale(self,t):
        phi=tf.maximum(self.vapor_volume_ratio(t),tf.cast(1e-12,t.dtype))
        return tf.cast(self.cfg.R0,t.dtype)*tf.pow(phi,1.0/3.0)

    def vapor_state(self,t):
        R=self.radius(t)
        Pv=(self.cfg.Pi+self.cfg.pv0)*tf.pow(R/self.cfg.R0,-3.0*self.cfg.gamma_v)-self.cfg.Pi
        rho=self.cfg.rho_v_ratio*tf.pow(R/self.cfg.R0,-3.0)
        return Pv,rho

    def vapor_raw(self,X):
        X=tf.cast(X,tf.float32); t=X[:,2:3]
        Lv=self.vapor_length_scale(t)
        return self.vapor(X,Lv)

    def vapor_velocity(self,X):
        X=tf.cast(X,tf.float32); t=X[:,2:3]; uh_r,uh_z,qh=tf.split(self.vapor_raw(X),3,axis=1)
        tau=(t-self.cfg.tmin)/(self.cfg.tmax-self.cfg.tmin)
        return tau*self.cfg.u_scale*uh_r,tau*self.cfg.u_scale*uh_z,self.cfg.q_scale*qh

model=FullSphericalALEPINN(cfg,p0)
optimizer=tf.keras.optimizers.Adam(cfg.lr,beta_1=cfg.beta1)
print('liquid pressure network: NONE; p_l = p_inf + Cdot/s - C^2/(2 s^4)')
print('hard initial acceleration from Poisson solve:', cfg.initial_acceleration)

# %% [notebook code cell 8]
# -----------------------------------------------------------------------------
# Fixed Chebyshev time grid + UNIFORM high-resolution vapor spatial collocation.
#
# 33 Chebyshev--Lobatto time nodes are permanent.  t=0 is handled by the cheap
# exact initial vapor residual, leaving 32 nonzero full-PDE time slices.
#
# Every nonzero slice receives exactly 64*5 = 320 interior vapor points:
#
#     32 x 320 = 10,240 moving-vapor PDE collocation points.
#
# No point budget is stolen from early/intermediate times.  Spatial reference
# coordinates are refreshed every cfg.refresh_every epochs while the time nodes
# and tensor shapes remain fixed.  train_step still evaluates the complete set.
# -----------------------------------------------------------------------------


def chebyshev_lobatto_times(n,tmin,tmax):
    n=int(n)
    if n<2:
        return np.array([tmin],dtype=np.float32)
    j=np.arange(n,dtype=np.float64)
    tau=0.5*(1.0-np.cos(np.pi*j/(n-1)))
    return (tmin+(tmax-tmin)*tau).astype(np.float32)


Tnodes_np=chebyshev_lobatto_times(cfg.n_time_slices,cfg.tmin,cfg.tmax)
Tnodes=tf.constant(Tnodes_np,dtype=tf.float32)
Tvapor_np=Tnodes_np[1:]  # t=0 handled separately
N_vapor_times=len(Tvapor_np)
N_vapor_total=N_vapor_times*int(cfg.n_vapor_per_time)

# Liquid/far/symmetry sets remain fixed.
Xbox=tf.convert_to_tensor(sampler.box(cfg.n_liquid_fixed),tf.float32)
Xfr=tf.convert_to_tensor(sampler.far_r(cfg.n_boundary_1d),tf.float32)
Xft=tf.convert_to_tensor(sampler.far_top(cfg.n_boundary_1d),tf.float32)

Ue=sampler.sobol(cfg.n_boundary_1d,2)
Xeq=tf.convert_to_tensor(
    np.c_[cfg.rmax*Ue[:,0],np.zeros(cfg.n_boundary_1d),cfg.tmax*Ue[:,1]].astype(np.float32))
Ua=sampler.sobol(cfg.n_boundary_1d,2)
Xaxis=tf.convert_to_tensor(
    np.c_[np.zeros(cfg.n_boundary_1d),cfg.zmax*Ua[:,0],cfg.tmax*Ua[:,1]].astype(np.float32))


def vapor_t0_cloud(n,margin=0.02):
    n=int(n); out=[]; need=n
    while need>0:
        U=sampler.sobol(max(1024,2*need),2)
        r=cfg.axis_eps+(cfg.R0-cfg.axis_eps-margin)*U[:,0]
        z=(cfg.R0-margin)*U[:,1]
        keep=(r*r+z*z)<(cfg.R0-margin)**2
        X=np.c_[r[keep],z[keep],np.zeros(np.count_nonzero(keep))]
        take=min(need,len(X))
        if take:
            out.append(X[:take]); need-=take
    return np.vstack(out).astype(np.float32)


Xv0=tf.convert_to_tensor(vapor_t0_cloud(cfg.n_vapor_t0_fixed),tf.float32)


def unit_vapor_spatial(n,margin=0.02):
    """Sobol points in the reference quarter disk, returned as (yr,yz)."""
    n=int(n); out=[]; need=n
    while need>0:
        U=sampler.sobol(max(1024,2*need),2)
        yr=0.001+(1.0-margin-0.001)*U[:,0]
        yz=(1.0-margin)*U[:,1]
        keep=(yr*yr+yz*yz)<(1.0-margin)**2
        Y=np.c_[yr[keep],yz[keep]]
        take=min(need,len(Y))
        if take:
            out.append(Y[:take]); need-=take
    return np.vstack(out).astype(np.float32)


def make_vapor_time_slices():
    """320 reference-space vapor points at every nonzero fixed time node."""
    blocks=[]
    for ti in Tvapor_np:
        Ysp=unit_vapor_spatial(cfg.n_vapor_per_time)
        blocks.append(np.c_[Ysp,np.full(len(Ysp),ti,dtype=np.float32)])
    Y=np.vstack(blocks).astype(np.float32)
    if len(Y)!=N_vapor_total:
        raise RuntimeError('unexpected moving-vapor collocation size')
    return Y


def make_interface_time_slices():
    blocks=[]
    for ti in Tnodes_np:
        u=sampler.sobol(cfg.n_interface_per_time,1)[:,0]
        th=(0.5*np.pi*u).astype(np.float32)
        blocks.append(np.c_[th,np.full(len(th),ti,dtype=np.float32)])
    return np.vstack(blocks).astype(np.float32)


Yvap=tf.Variable(make_vapor_time_slices(),trainable=False,dtype=tf.float32,
                 name='Yvap_collocation')
Ginterface=tf.Variable(make_interface_time_slices(),trainable=False,dtype=tf.float32,
                       name='Ginterface_collocation')
Ygauge=tf.Variable(unit_vapor_spatial(cfg.n_gauge_spatial),trainable=False,dtype=tf.float32,
                   name='Ygauge_collocation')
Tgauge=Tnodes

U0=sampler.sobol(cfg.n_boundary_1d,1)[:,0]
G0=tf.convert_to_tensor(
    np.c_[0.5*np.pi*U0,np.zeros_like(U0)].astype(np.float32))


@tf.function
def map_vapor_params(Y):
    yr,yz,t=Y[:,0:1],Y[:,1:2],Y[:,2:3]
    # Collocation locations move with the learned domain, but their motion is not
    # differentiated through the outer parameter optimization.
    R=tf.stop_gradient(model.radius(t))
    return tf.concat([R*yr,R*yz,t],axis=1)


def refresh_spatial_collocation(verbose=True):
    """Refresh spatial Sobol coordinates only; fixed times and fixed shapes."""
    Yvap.assign(make_vapor_time_slices())
    Ginterface.assign(make_interface_time_slices())
    Ygauge.assign(unit_vapor_spatial(cfg.n_gauge_spatial))
    if verbose:
        print('spatial collocation refresh; Chebyshev times and 320 points/slice unchanged')


print('fixed Chebyshev time nodes:')
print(Tnodes_np)
print('deterministic training-set sizes:',
      'liquid',len(Xbox),
      'moving-vapor TOTAL',Yvap.shape[0],
      f'= {len(Tvapor_np)} nonzero times x {cfg.n_vapor_per_time}',
      'vap-t0',len(Xv0),
      'ALE-interface',Ginterface.shape[0],
      f'= {len(Tnodes_np)} times x {cfg.n_interface_per_time}',
      'gauge spatial',Ygauge.shape[0],
      f'x {len(Tnodes_np)} times',
      'far_r',len(Xfr),'far_top',len(Xft))
print('stochastic minibatching: NONE; spatial refresh every',cfg.refresh_every,'epochs')
print('vapor NN spatial feature scale: Lv = R0*(V/V0)^(1/3); spherical benchmark Lv=R(t)')
print('normal-traction weight lambda_n =',cfg.lambda_normal)

# %% [notebook code cell 9]
# -----------------------------------------------------------------------------
# Residual operators for fixed full-batch training.
#
# Expensive derivative work is grouped by residual type.  Higher derivative
# levels remain mathematically necessary for the streamfunction viscous term,
# but redundant constraints and redundant pressure differentiation are removed.
# -----------------------------------------------------------------------------
@tf.function
def liquid_terms(X):
    r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]

    # Three nested AD levels are required because the streamfunction correction
    # enters velocity through first derivatives and viscous momentum needs two
    # more spatial derivatives of velocity.  Everything needed at each level is
    # obtained from the same network evaluation and then reused.
    with tf.GradientTape(persistent=True) as tape3:
        tape3.watch([r,z])
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([r,z,t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([r,z])
                inp=tf.concat([r,z,t],axis=1)
                chi=model.liquid(inp)
                R,Rd,C,Cdot=model.sink_dynamics(t)
                s=tf.sqrt(r*r+z*z+tf.cast(1e-12,r.dtype))
                se=tf.maximum(s,0.5*tf.stop_gradient(R))
                urs=C*r/tf.pow(se,3)
                uzs=C*z/tf.pow(se,3)
                tau=(t-cfg.tmin)/(cfg.tmax-cfg.tmin)
            chi_r=tape1.gradient(chi,r)
            chi_z=tape1.gradient(chi,z)
            del tape1

            urc=-tau*tau*r*chi_z
            uzc=tau*tau*(2.0*chi+r*chi_r)
            url=urs+urc
            uzl=uzs+uzc

        url_r=tape2.gradient(url,r)
        url_z=tape2.gradient(url,z)
        url_t=tape2.gradient(url,t)
        uzl_r=tape2.gradient(uzl,r)
        uzl_z=tape2.gradient(uzl,z)
        uzl_t=tape2.gradient(uzl,t)
        del tape2

    url_rr=tape3.gradient(url_r,r)
    url_zz=tape3.gradient(url_z,z)
    uzl_rr=tape3.gradient(uzl_r,r)
    uzl_zz=tape3.gradient(uzl_z,z)
    del tape3

    # Exact spherical pressure gradient: avoid differentiating the pressure
    # expression with another tape.
    # p = p_inf + Cdot/s - C^2/(2 s^4)
    p=tf.cast(cfg.pinf,s.dtype)+Cdot/s-0.5*C*C/tf.pow(s,4)
    dpds=-Cdot/tf.pow(s,2)+2.0*C*C/tf.pow(s,5)
    p_r=dpds*r/s
    p_z=dpds*z/s

    rs=tf.maximum(r,tf.cast(cfg.axis_eps,r.dtype))
    gap=tf.cast(cfg.pde_interface_halfwidth,s.dtype)
    liq=tf.stop_gradient(tf.cast(s>R+gap,tf.float32))

    alr=url_t+url*url_r+uzl*url_z
    alz=uzl_t+url*uzl_r+uzl*uzl_z
    llr=url_rr+url_r/rs-url/(rs*rs)+url_zz
    llz=uzl_rr+uzl_r/rs+uzl_zz
    mlr=alr+p_r-llr/cfg.Re
    mlz=alz+p_z-llz/cfg.Re

    return dict(liq=liq,mlr=mlr,mlz=mlz,url=url,uzl=uzl,p=p,R=R,C=C,
                alr=alr,alz=alz,pr=p_r,pz=p_z)


@tf.function
def vapor_terms(X):
    r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]

    # One vapor forward evaluation, with first/second derivatives reused for
    # continuity and both momentum components.
    with tf.GradientTape(persistent=True) as tape3:
        tape3.watch([r,z])
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([r,z,t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([r,z,t])
                inp=tf.concat([r,z,t],axis=1)
                urv,uzv,qv=model.vapor_velocity(inp)
                Pv,rho=model.vapor_state(t)
            q_r=tape1.gradient(qv,r)
            q_z=tape1.gradient(qv,z)
            Pv_t=tape1.gradient(Pv,t)
            del tape1

        urv_r=tape2.gradient(urv,r)
        urv_z=tape2.gradient(urv,z)
        urv_t=tape2.gradient(urv,t)
        uzv_r=tape2.gradient(uzv,r)
        uzv_z=tape2.gradient(uzv,z)
        uzv_t=tape2.gradient(uzv,t)
        del tape2

    urv_rr=tape3.gradient(urv_r,r)
    urv_zz=tape3.gradient(urv_z,z)
    uzv_rr=tape3.gradient(uzv_r,r)
    uzv_zz=tape3.gradient(uzv_z,z)
    del tape3

    rs=tf.maximum(r,tf.cast(cfg.axis_eps,r.dtype))
    divv=urv_r+urv/rs+uzv_z
    contv=divv+Pv_t/(cfg.gamma_v*(cfg.Pi+Pv))

    avr=urv_t+urv*urv_r+uzv*urv_z
    avz=uzv_t+urv*uzv_r+uzv*uzv_z
    lvr=urv_rr+urv_r/rs-urv/(rs*rs)+urv_zz
    lvz=uzv_rr+uzv_r/rs+uzv_zz
    fac=(cfg.mu_v_ratio/cfg.Re)/rho

    mvr=avr+q_r-fac*lvr
    mvz=avz+q_z-fac*lvz
    return dict(contv=contv,mvr=mvr,mvz=mvz,urv=urv,uzv=uzv,qv=qv,Pv=Pv,rho=rho)


@tf.function
def vapor_t0_momentum_terms(X):
    """Cheap exact t=0 vapor momentum residual.

    Because u_v=tau*u_scale*u_hat and tau=0 at t=0,
        u_v = grad(u_v) = Laplacian(u_v) = 0,
    so convection and viscosity vanish exactly.  Also Rdot(0)=0 makes the
    t=0 vapor continuity equation an identity.  The only nontrivial initial
    vapor PDE is
        u_{v,t} + grad(q_v) = 0.

    At t=0, u_{v,t}=(u_scale/tmax)*u_hat, so no time derivative of the velocity
    network is needed.
    """
    r,z,t=X[:,0:1],X[:,1:2],X[:,2:3]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([r,z])
        inp=tf.concat([r,z,t],axis=1)
        raw=model.vapor_raw(inp)
        uh_r,uh_z,qh=tf.split(raw,3,axis=1)
        q=tf.cast(cfg.q_scale,qh.dtype)*qh
    q_r=tape.gradient(q,r)
    q_z=tape.gradient(q,z)
    del tape

    tau_t=tf.cast(1.0/(cfg.tmax-cfg.tmin),uh_r.dtype)
    ur_t=tf.cast(cfg.u_scale,uh_r.dtype)*tau_t*uh_r
    uz_t=tf.cast(cfg.u_scale,uh_z.dtype)*tau_t*uh_z
    return ur_t+q_r,uz_t+q_z


@tf.function
def interface_residuals(G):
    th,t=G[:,0:1],G[:,1:2]
    R,Rd=model.radius_rate(t)
    nr=tf.sin(th); nz=tf.cos(th)
    tr=tf.cos(th); tz=-tf.sin(th)
    r=R*nr; z=R*nz

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([r,z])
        X=tf.concat([r,z,t],axis=1)
        url,uzl,p,_=model.liquid_state(X)
        urv,uzv,q=model.vapor_velocity(X)
        Pv,rho=model.vapor_state(t)

    url_r=tape.gradient(url,r); url_z=tape.gradient(url,z)
    uzl_r=tape.gradient(uzl,r); uzl_z=tape.gradient(uzl,z)
    urv_r=tape.gradient(urv,r); urv_z=tape.gradient(urv,z)
    uzv_r=tape.gradient(uzv,r); uzv_z=tape.gradient(uzv,z)
    del tape

    un_l=url*nr+uzl*nz
    dnn_l=nr*(url_r*nr+url_z*nz)+nz*(uzl_r*nr+uzl_z*nz)
    dnn_v=nr*(urv_r*nr+urv_z*nz)+nz*(uzv_r*nr+uzv_z*nz)
    shear_l=tr*(2.0*url_r*nr+(url_z+uzl_r)*nz)+tz*((uzl_r+url_z)*nr+2.0*uzl_z*nz)
    shear_v=tr*(2.0*urv_r*nr+(urv_z+uzv_r)*nz)+tz*((uzv_r+urv_z)*nr+2.0*uzv_z*nz)

    pvtot=Pv+rho*q
    normal=(p-pvtot)-(2.0/cfg.Re)*dnn_l+(2.0*cfg.mu_v_ratio/cfg.Re)*dnn_v
    tangential=(1.0/cfg.Re)*shear_l-(cfg.mu_v_ratio/cfg.Re)*shear_v

    return dict(du_r=url-urv,du_z=uzl-uzv,kin=Rd-un_l,
                normal=normal,tangential=tangential,
                p=p,pvtot=pvtot,q=q,R=R,Rd=Rd)


@tf.function
def interface_components(G):
    I=interface_residuals(G)
    Jdu_r=mse(I['du_r'])
    Jdu_z=mse(I['du_z'])
    Jkin=mse(I['kin'])
    Jnormal=mse(I['normal'])
    Jtang=mse(I['tangential'])
    return Jdu_r,Jdu_z,Jkin,Jnormal,Jtang


@tf.function
def weighted_interface_loss(G):
    Jdu_r,Jdu_z,Jkin,Jnormal,Jtang=interface_components(G)
    Ji=Jdu_r+Jdu_z+Jkin+tf.cast(cfg.lambda_normal,Jnormal.dtype)*Jnormal+Jtang
    return Ji,Jdu_r,Jdu_z,Jkin,Jnormal,Jtang


@tf.function
def initial_interface_traction_loss(G):
    I=interface_residuals(G)
    return tf.cast(cfg.lambda_normal,I['normal'].dtype)*mse(I['normal'])+mse(I['tangential'])


@tf.function
def far_velocity_decay_loss(X):
    """Only the learned F3 velocity correction is trained at the far boundary.

    With p_l=p_inf+Cdot/s-C^2/(2s^4), the asymptotic pressure Robin relation is
    an analytical identity and does not need to be differentiated every epoch.
    """
    urc,uzc,_=model.liquid_correction(tf.cast(X,tf.float32))
    return mse(urc)+mse(uzc)


@tf.function
def far_exact_diagnostic(X):
    ur,uz,p,_=model.liquid_state(X)
    ue,ve,pe=exact_values(X)
    return mse(ur-ue)+mse(uz-ve)+mse(p-pe)


@tf.function
def q_gauge_loss(Y):
    """Zero-volume-mean q_v gauge at every fixed time node in one vectorized call."""
    yr,yz=Y[:,0:1],Y[:,1:2]
    nt=tf.shape(Tgauge)[0]
    ng=tf.shape(Y)[0]

    yr_all=tf.tile(yr,[nt,1])
    yz_all=tf.tile(yz,[nt,1])
    t_all=tf.reshape(tf.repeat(Tgauge,repeats=ng),[-1,1])

    R=tf.stop_gradient(model.radius(t_all))
    X=tf.concat([R*yr_all,R*yz_all,t_all],axis=1)
    _,_,q=model.vapor_velocity(X)

    q=tf.reshape(q,[nt,ng])
    w=tf.reshape(yr,[-1])
    qbar=tf.reduce_sum(q*w[None,:],axis=1)/(tf.reduce_sum(w)+1e-12)
    return tf.reduce_mean(tf.square(qbar))


@tf.function
def symmetry_loss(Xe,Xa):
    r,z,t=Xe[:,0:1],Xe[:,1:2],Xe[:,2:3]
    R=model.radius(t); s=tf.sqrt(r*r+z*z+1e-12)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(z)
        inp=tf.concat([r,z,t],axis=1)
        url,uzl,p,_=model.liquid_state(inp)
        urv,uzv,q=model.vapor_velocity(inp)
    url_z=tape.gradient(url,z)
    p_z=tape.gradient(p,z)
    urv_z=tape.gradient(urv,z)
    q_z=tape.gradient(q,z)
    del tape
    liq=tf.stop_gradient(tf.cast(s>R,tf.float32))
    vap=tf.stop_gradient(tf.cast(s<R,tf.float32))
    Je=(masked_mse(uzl,liq)+masked_mse(url_z,liq)+masked_mse(p_z,liq)
        +masked_mse(uzv,vap)+masked_mse(urv_z,vap)+masked_mse(q_z,vap))

    r,z,t=Xa[:,0:1],Xa[:,1:2],Xa[:,2:3]
    R=model.radius(t)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(r)
        inp=tf.concat([r,z,t],axis=1)
        urv,uzv,q=model.vapor_velocity(inp)
    uzv_r=tape.gradient(uzv,r)
    q_r=tape.gradient(q,r)
    del tape
    vap=tf.stop_gradient(tf.cast(z<R,tf.float32))
    Ja=masked_mse(urv,vap)+masked_mse(uzv_r,vap)+masked_mse(q_r,vap)
    return Je+Ja


@tf.function
def train_step():
    # Full deterministic sets: no random sampling/shuffling in the optimizer step.
    Xvb=map_vapor_params(Yvap)

    with tf.GradientTape() as tape:
        Tl=liquid_terms(Xbox)
        Tv=vapor_terms(Xvb)
        mvr0,mvz0=vapor_t0_momentum_terms(Xv0)

        Jml=masked_mse(Tl['mlr'],Tl['liq'])+masked_mse(Tl['mlz'],Tl['liq'])
        Jcv=mse(Tv['contv'])
        Jmv=mse(Tv['mvr'])+mse(Tv['mvz'])
        Jmv0=mse(mvr0)+mse(mvz0)

        Ji,Jdu_r,Jdu_z,Jkin,Jnormal,Jtang=weighted_interface_loss(Ginterface)
        Ji0=initial_interface_traction_loss(G0)
        Jfar=0.5*(far_velocity_decay_loss(Xfr)+far_velocity_decay_loss(Xft))
        Jq=q_gauge_loss(Ygauge)
        Jsym=symmetry_loss(Xeq,Xaxis)

        loss=Jml+Jcv+Jmv+Jmv0+Ji+Ji0+Jfar+Jq+Jsym

    g=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients([(a,b) for a,b in zip(g,model.trainable_variables) if a is not None])

    return (loss,Jml,Jcv,Jmv,Jmv0,Ji,Ji0,Jfar,Jq,Jsym,
            Jdu_r,Jdu_z,Jkin,Jnormal,Jtang)


def fit(epochs=22000,refresh_every=None):
    names=['loss','Jml','Jcv','Jmv','Jmv0','Ji(w)','Ji0','Jfar','Jq','Jsym',
           'Jdu_r','Jdu_z','Jkin','Jnormal','Jtang']
    print(' '.join(names))
    refresh_every=cfg.refresh_every if refresh_every is None else int(refresh_every)
    t_start=time.time(); t_mark=t_start; prev_it=-1
    for it in range(epochs):
        if refresh_every>0 and it>0 and it%refresh_every==0:
            refresh_spatial_collocation(verbose=False)
            print(f'--- spatial collocation refresh at iteration {it}; '
                  f'Chebyshev times fixed; vapor points/slice={cfg.n_vapor_per_time} ---')
        vals=train_step()
        if it%250==0 or it==epochs-1:
            now=time.time(); dt=now-t_start
            nblock=max(1,it-prev_it); step_s=(now-t_mark)/nblock
            print(it,*[f'{float(v.numpy()):.3e}' for v in vals],
                  f'step={step_s:.4f}s',f'elapsed={dt:.1f}s')
            t_mark=now; prev_it=it

fit()

# %% [notebook code cell 10]
# Collocation / volume-scaled feature / gauge sanity diagnostics
print('Chebyshev-Lobatto time nodes (fixed for entire run):')
print(Tnodes_np)
print('moving-vapor allocation: total =',N_vapor_total,
      'uniform points per nonzero time =',cfg.n_vapor_per_time)

print('volume-scaled vapor feature diagnostics:')
for target in [0.25,0.60,cfg.tmax]:
    tt=tf.constant([[target]],tf.float32)
    phi=float(model.vapor_volume_ratio(tt).numpy()[0,0])
    Lv=float(model.vapor_length_scale(tt).numpy()[0,0])
    print(f'  t={target:.2f}: V/V0={phi:.6f}, Lv={Lv:.6f}, physical bubble/Lv = O(1)')

@tf.function
def gauge_means_all_times(Y):
    yr,yz=Y[:,0:1],Y[:,1:2]
    nt=tf.shape(Tgauge)[0]; ng=tf.shape(Y)[0]
    yr_all=tf.tile(yr,[nt,1]); yz_all=tf.tile(yz,[nt,1])
    t_all=tf.reshape(tf.repeat(Tgauge,repeats=ng),[-1,1])
    R=tf.stop_gradient(model.radius(t_all))
    X=tf.concat([R*yr_all,R*yz_all,t_all],axis=1)
    _,_,q=model.vapor_velocity(X)
    q=tf.reshape(q,[nt,ng]); w=tf.reshape(yr,[-1])
    return tf.reduce_sum(q*w[None,:],axis=1)/(tf.reduce_sum(w)+1e-12)

qbars=gauge_means_all_times(Ygauge).numpy()
print('gauge mean max abs over all fixed time nodes =',float(np.max(np.abs(qbars))))
print('gauge means at t=0, near 0.25, near 0.60, tmax:')
for target in [0.0,0.25,0.60,cfg.tmax]:
    j=int(np.argmin(np.abs(Tnodes_np-target)))
    print(f'  target={target:.3f} node={Tnodes_np[j]:.6f} qbar={qbars[j]:+.6e}')

# %% [notebook code cell 11]
# Diagnostics
print('\ntrajectory / thermodynamics / exact-pressure + learned-liquid-velocity dynamics')
for tt in [0.0,0.25,0.60,0.90]:
    if tt<=cfg.tmax:
        Rref,Rdref,Accref=ref.rva_np(tt); ttf=tf.constant([[tt]],tf.float32)
        R,Rd,C,Cdot=model.sink_dynamics(ttf)
        Pv,rho=model.vapor_state(ttf)
        Cref=float(Rref*Rref*Rdref)
        Cdot_ref=float(2*Rref*Rdref*Rdref+Rref*Rref*Accref)
        mass_ratio=float((rho/cfg.rho_v_ratio*tf.pow(R/cfg.R0,3)).numpy()[0,0])
        print(f't={tt:.2f}: R={float(R.numpy()[0,0]):.6f}/{Rref:.6f}, '
              f'Rdot={float(Rd.numpy()[0,0]):+.6f}/{Rdref:+.6f}, C={float(C.numpy()[0,0]):+.6f}/{Cref:+.6f}, '
              f'Pv={float(Pv.numpy()[0,0]):+.6f}/{ref.pv_np(Rref):+.6f}, '
              f'Cdot={float(Cdot.numpy()[0,0]):+.6f}/{Cdot_ref:+.6f}, mass={mass_ratio:.8f}')

# Verify the Poisson-derived hard initial pressure.
Xs=tf.constant([[1.0,0.0,0.0],[2.0,0.0,0.0],[4.0,0.0,0.0]],tf.float32)
_,_,p_init,_=model.liquid_state(Xs)
p_target=p0.tf(Xs[:,0:1])
print('hard p_l(t=0) at s=1,2,4:',p_init.numpy().reshape(-1))
print('target p0 values:',p_target.numpy().reshape(-1))
print('initial pressure max abs error =',float(np.max(np.abs(p_init.numpy().reshape(-1)-p_target.numpy().reshape(-1)))))

I0=interface_residuals(G0)
print('initial interface normal-traction RMS =',np_rms(I0['normal'].numpy()))
print('initial q_Gamma mean =',float(np.mean(I0['q'].numpy())))
print('far velocity-decay loss =',float((0.5*(far_velocity_decay_loss(Xfr)+far_velocity_decay_loss(Xft))).numpy()))
print('far exact diagnostic (not optimized) =',float((0.5*(far_exact_diagnostic(Xfr)+far_exact_diagnostic(Xft))).numpy()))

Td=liquid_terms(Xbox); mask=Td['liq'].numpy().reshape(-1)>0.5
print('ALE liquid momentum RMS r/z =',np_rms(Td['mlr'].numpy().reshape(-1)[mask]),np_rms(Td['mlz'].numpy().reshape(-1)[mask]))

# Size of the learned liquid velocity correction: should remain small in an exact spherical benchmark.
Xcd=Xbox
urc,uzc,_=model.liquid_correction(Xcd)
print('learned liquid correction RMS ur/uz =',np_rms(urc.numpy()),np_rms(uzc.numpy()))


# Weighted-interface diagnostics on the exact fixed training interface set.
Ji,Jdu_r,Jdu_z,Jkin,Jnormal,Jtang=weighted_interface_loss(Ginterface)
print('interface MSE components (unweighted):',
      'du_r=',float(Jdu_r.numpy()),
      'du_z=',float(Jdu_z.numpy()),
      'kin=',float(Jkin.numpy()),
      'normal=',float(Jnormal.numpy()),
      'tangential=',float(Jtang.numpy()))
print('weighted interface loss =',float(Ji.numpy()),'with lambda_normal =',cfg.lambda_normal)

# %% [notebook code cell 12]
# ============================================================
# FRESH-POINT VALIDATION -- independent of all training points
# ============================================================
# This cell is intentionally part of the notebook so every completed run is
# checked for spatial generalization, especially in the compressed late-time vapor.

VAL_TIMES=[0.25,0.60,0.90]
N_VAP_VAL=5000
N_THETA_VAL=2000
VAL_MARGIN=0.02

rng_val=np.random.default_rng(20260823)


def rms_np(x):
    x=np.asarray(x).reshape(-1)
    return float(np.sqrt(np.mean(x*x)))


def fresh_vapor_points_at_time(t_value,n=N_VAP_VAL,margin=VAL_MARGIN):
    tt=tf.constant([[t_value]],dtype=tf.float32)
    R=float(model.radius(tt).numpy()[0,0])
    Rref=1.0-margin
    out=[]; need=int(n)
    while need>0:
        m=max(2048,2*need)
        yr=rng_val.uniform(0.001,Rref,size=m)
        yz=rng_val.uniform(0.0,Rref,size=m)
        keep=(yr*yr+yz*yz)<Rref**2
        yr=yr[keep]; yz=yz[keep]
        take=min(need,len(yr))
        if take:
            out.append(np.column_stack([
                R*yr[:take],R*yz[:take],np.full(take,t_value)
            ]))
            need-=take
    return tf.convert_to_tensor(np.vstack(out).astype(np.float32),tf.float32)


def validate_vapor_at_time(t_value,n=N_VAP_VAL):
    X=fresh_vapor_points_at_time(t_value,n=n)
    V=vapor_terms(X)
    cont=V['contv'].numpy(); mvr=V['mvr'].numpy(); mvz=V['mvz'].numpy()
    rho=V['rho'].numpy(); pmvr=rho*mvr; pmvz=rho*mvz

    print(f'\n========== FRESH VAPOR VALIDATION t={t_value:.2f} ==========')
    print('conditioned PDE residuals:')
    print(f'  RMS continuity = {rms_np(cont):.6e}')
    print(f'  RMS mom-r      = {rms_np(mvr):.6e}')
    print(f'  RMS mom-z      = {rms_np(mvz):.6e}')
    print('MSE equivalents:')
    print(f'  Jcv_val        = {np.mean(cont**2):.6e}')
    print(f'  Jmvr_val       = {np.mean(mvr**2):.6e}')
    print(f'  Jmvz_val       = {np.mean(mvz**2):.6e}')
    print(f'  Jmv_val        = {np.mean(mvr**2)+np.mean(mvz**2):.6e}')
    print('physical vapor momentum residuals (rho * residual):')
    print(f'  RMS rho*mom-r  = {rms_np(pmvr):.6e}')
    print(f'  RMS rho*mom-z  = {rms_np(pmvz):.6e}')
    return dict(X=X,RMS_cont=rms_np(cont),RMS_mvr=rms_np(mvr),RMS_mvz=rms_np(mvz),
                Jcv=float(np.mean(cont**2)),Jmv=float(np.mean(mvr**2)+np.mean(mvz**2)))


def validate_interface_at_time(t_value,ntheta=N_THETA_VAL):
    theta=np.linspace(0.0,0.5*np.pi,ntheta,dtype=np.float32)
    G=tf.convert_to_tensor(np.column_stack([
        theta,np.full(ntheta,t_value,dtype=np.float32)
    ]),tf.float32)
    I=interface_residuals(G)
    keys=['du_r','du_z','kin','normal','tangential']

    print(f'\n========== DENSE INTERFACE VALIDATION t={t_value:.2f} ==========')
    vals={}
    for key in keys:
        x=I[key].numpy().reshape(-1)
        vals[key]=dict(mean=float(np.mean(x)),rms=rms_np(x),
                       maxabs=float(np.max(np.abs(x))),mse=float(np.mean(x*x)))
        print(f'{key:12s}: mean={vals[key]["mean"]:+.6e}  '
              f'RMS={vals[key]["rms"]:.6e}  maxabs={vals[key]["maxabs"]:.6e}')

    Ji_unweighted=sum(vals[k]['mse'] for k in keys)
    Ji_weighted=(vals['du_r']['mse']+vals['du_z']['mse']+vals['kin']['mse']
                 +cfg.lambda_normal*vals['normal']['mse']+vals['tangential']['mse'])
    print(f'\n  Ji unweighted = {Ji_unweighted:.6e}')
    print(f'  Ji weighted   = {Ji_weighted:.6e}')
    print(f'  mean p_l      = {float(tf.reduce_mean(I["p"]).numpy()):+.6f}')
    print(f'  mean p_v,total= {float(tf.reduce_mean(I["pvtot"]).numpy()):+.6f}')
    print(f'  mean q_Gamma  = {float(tf.reduce_mean(I["q"]).numpy()):+.6f}')
    return vals


vapor_validation={}
interface_validation={}
for tt in VAL_TIMES:
    vapor_validation[tt]=validate_vapor_at_time(tt)
    interface_validation[tt]=validate_interface_at_time(tt)

# %% [notebook code cell 13]
# ============================================================
# VISUALIZATION — enjoy the collapse
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# ------------------------------------------------------------
# 1. Dense trajectory data
# ------------------------------------------------------------
tt = np.linspace(cfg.tmin, cfg.tmax, 500).astype(np.float32)
tt_tf = tf.constant(tt[:, None], dtype=tf.float32)

# radius_acceleration returns (R, Rdot, Rddot)
R_tf, Rd_tf, Rdd_tf = model.radius_acceleration(tt_tf)

R_nn   = R_tf.numpy().ravel()
Rd_nn  = Rd_tf.numpy().ravel()
Rdd_nn = Rdd_tf.numpy().ravel()

R_ref, Rd_ref, Rdd_ref = ref.rva_np(tt)
Pv_nn  = pv_from_R_np(R_nn)
Pv_ref = pv_from_R_np(R_ref)

C_nn   = R_nn**2 * Rd_nn
C_ref  = R_ref**2 * Rd_ref

Cd_nn  = 2*R_nn*Rd_nn**2 + R_nn**2*Rdd_nn
Cd_ref = 2*R_ref*Rd_ref**2 + R_ref**2*Rdd_ref


# ------------------------------------------------------------
# 2. Main trajectory figure
# ------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

# Radius
ax[0,0].plot(tt, R_ref, '--', lw=2.5, label='reference')
ax[0,0].plot(tt, R_nn,  lw=2.5, label='PINN')
ax[0,0].set_xlabel(r'$t$')
ax[0,0].set_ylabel(r'$R(t)$')
ax[0,0].set_title('Bubble radius')
ax[0,0].grid(alpha=0.25)
ax[0,0].legend(frameon=False)

# Wall velocity
ax[0,1].plot(tt, Rd_ref, '--', lw=2.5, label='reference')
ax[0,1].plot(tt, Rd_nn,  lw=2.5, label='PINN')
ax[0,1].set_xlabel(r'$t$')
ax[0,1].set_ylabel(r'$\dot R(t)$')
ax[0,1].set_title('Collapse velocity')
ax[0,1].grid(alpha=0.25)

# Vapor thermodynamic pressure
ax[1,0].plot(tt, Pv_ref, '--', lw=2.5, label='reference')
ax[1,0].plot(tt, Pv_nn,  lw=2.5, label='PINN')
ax[1,0].axhline(0.0, lw=1.0, alpha=0.4)
ax[1,0].set_xlabel(r'$t$')
ax[1,0].set_ylabel(r'$P_v(t)$')
ax[1,0].set_title('Vapor thermodynamic pressure')
ax[1,0].grid(alpha=0.25)

# Relative errors
Rerr  = np.abs(R_nn-R_ref) / np.maximum(np.abs(R_ref), 1e-12)
Verr  = np.abs(Rd_nn-Rd_ref) / np.maximum(np.abs(Rd_ref), 1e-3)
Cderr = np.abs(Cd_nn-Cd_ref) / np.maximum(np.abs(Cd_ref), 1e-3)

ax[1,1].semilogy(tt, np.maximum(Rerr,  1e-7), lw=2.2, label=r'$R$')
ax[1,1].semilogy(tt, np.maximum(Verr,  1e-7), lw=2.2, label=r'$\dot R$')
ax[1,1].semilogy(tt, np.maximum(Cderr, 1e-7), lw=2.2, label=r'$\dot C$')
ax[1,1].set_xlabel(r'$t$')
ax[1,1].set_ylabel('relative error')
ax[1,1].set_title('Trajectory error')
ax[1,1].grid(alpha=0.25, which='both')
ax[1,1].legend(frameon=False)

fig.suptitle(
    'Two-phase PINN bubble collapse — learned vs coupled spherical reference',
    fontsize=16
)

plt.show()


# ------------------------------------------------------------
# 3. Spatial vapor snapshots
#
# Color = total vapor pressure p_v,total = Pv + rho*q
# Arrows = learned vapor velocity
#
# Solid circle  = learned interface
# Dashed circle = reference interface
# ------------------------------------------------------------

snapshot_times = [0.25, 0.60, 0.90]

fig, axes = plt.subplots(
    1, len(snapshot_times),
    figsize=(15, 5),
    constrained_layout=True
)

# First pass: compute fields so all panels share one color scale
snapshots = []

for ts in snapshot_times:

    t1 = tf.constant([[ts]], dtype=tf.float32)

    R = float(model.radius(t1).numpy()[0,0])
    Pv, rho = model.vapor_state(t1)
    Pv = float(Pv.numpy()[0,0])
    rho = float(rho.numpy()[0,0])

    Rr, Rdr, _ = ref.rva_np(ts)

    # Regular grid in the learned bubble
    n = 100
    rr = np.linspace(0.0, R, n)
    zz = np.linspace(-R, R, 2*n)

    RR, ZZ = np.meshgrid(rr, zz)

    inside = RR**2 + ZZ**2 <= R**2

    # Network was trained on z >= 0. Reflect by symmetry.
    X = np.column_stack([
        RR[inside],
        np.abs(ZZ[inside]),
        np.full(np.count_nonzero(inside), ts)
    ]).astype(np.float32)

    ur, uz_abs, q = model.vapor_velocity(
        tf.constant(X, dtype=tf.float32)
    )

    ur = ur.numpy().ravel()
    uz = uz_abs.numpy().ravel()

    # u_z is odd across the equatorial plane
    sign_z = np.sign(ZZ[inside])
    uz = uz * sign_z

    q = q.numpy().ravel()
    pvtot = Pv + rho*q

    P = np.full_like(RR, np.nan, dtype=float)
    UR = np.full_like(RR, np.nan, dtype=float)
    UZ = np.full_like(RR, np.nan, dtype=float)

    P[inside]  = pvtot
    UR[inside] = ur
    UZ[inside] = uz

    snapshots.append((ts, R, Rr, RR, ZZ, P, UR, UZ))


# Shared pressure scale
all_p = np.concatenate([
    s[5][np.isfinite(s[5])] for s in snapshots
])

vmin = np.percentile(all_p, 2)
vmax = np.percentile(all_p, 98)


for axx, snap in zip(axes, snapshots):

    ts, R, Rr, RR, ZZ, P, UR, UZ = snap

    # pressure field
    cf = axx.contourf(
        RR, ZZ, P,
        levels=32,
        vmin=vmin,
        vmax=vmax,
        cmap='coolwarm'
    )

    # learned bubble outline
    theta = np.linspace(-np.pi/2, np.pi/2, 400)

    axx.plot(
        R*np.cos(theta),
        R*np.sin(theta),
        lw=3.0
    )

    # reference outline
    axx.plot(
        Rr*np.cos(theta),
        Rr*np.sin(theta),
        '--',
        lw=2.0
    )

    # velocity vectors
    stride = 12

    RRs = RR[::stride, ::stride]
    ZZs = ZZ[::stride, ::stride]
    URs = UR[::stride, ::stride]
    UZs = UZ[::stride, ::stride]

    good = np.isfinite(URs) & np.isfinite(UZs)

    axx.quiver(
        RRs[good], ZZs[good],
        URs[good], UZs[good],
        angles='xy',
        scale_units='xy',
        scale=8.0,
        width=0.004,
        alpha=0.8
    )

    axx.axvline(0.0, lw=0.8, alpha=0.35)
    axx.axhline(0.0, lw=0.8, alpha=0.35)

    axx.set_aspect('equal')

    lim = 1.07*max(R, Rr)

    axx.set_xlim(0, lim)
    axx.set_ylim(-lim, lim)

    axx.set_xlabel(r'$r$')

    if axx is axes[0]:
        axx.set_ylabel(r'$z$')

    _, Rd_tf = model.radius_rate(
        tf.constant([[ts]], tf.float32)
    )
    Rd = float(Rd_tf.numpy()[0,0])
    axx.set_title(
        rf'$t={ts:.2f}$' '\n'
        rf'$R={R:.4f},\quad \dot R={Rd:.3f}$'
    )


# shared colorbar
cbar = fig.colorbar(
    cf,
    ax=axes,
    shrink=0.88,
    pad=0.02
)

cbar.set_label(r'total vapor pressure $p_v^{tot}$')

legend = [
    Line2D([0], [0], lw=3, label='PINN interface'),
    Line2D([0], [0], lw=2, ls='--', label='reference interface')
]

axes[0].legend(
    handles=legend,
    frameon=False,
    loc='lower left'
)

fig.suptitle(
    'The collapse — vapor pressure and learned velocity field',
    fontsize=16
)

plt.show()


# ------------------------------------------------------------
# 4. One final satisfying number
# ------------------------------------------------------------

R90_tf, Rd90_tf = model.radius_rate(
    tf.constant([[0.9]], tf.float32)
)

R90_nn = float(R90_tf.numpy()[0,0])
Rd90_nn = float(Rd90_tf.numpy()[0,0])
R90_ref, Rd90_ref, _ = ref.rva_np(0.9)

print()
print('================ FINAL t = 0.90 =================')
print(f'R      : {R90_nn:.8f}   reference {R90_ref:.8f}'
      f'   error = {100*abs(R90_nn-R90_ref)/abs(R90_ref):.3f}%')

print(f'Rdot   : {Rd90_nn:.8f}  reference {Rd90_ref:.8f}'
      f'   error = {100*abs(Rd90_nn-Rd90_ref)/abs(Rd90_ref):.3f}%')

print('==================================================')

# %% [notebook code cell 14]

