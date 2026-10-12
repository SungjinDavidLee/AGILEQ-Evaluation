"""Small, working reference planner. Separate from the original seven-method demo."""
from dataclasses import dataclass
from collections import Counter
import time
import numpy as np
from .contracts import Plan, ContractError
from .route import ReferenceLine
from .validation import validate_path


def quintic(p0,v0,a0,p1,v1,a1,T):
    if T<=0:raise ContractError("Polynomial duration must be positive")
    c=np.array([p0,v0,a0/2,0.,0.,0.])
    matrix=np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]])
    rhs=np.array([p1-p0-v0*T-a0*T*T/2,v1-v0-a0*T,a1-a0])
    c[3:]=np.linalg.solve(matrix,rhs);return c


def peval(coeff,t,derivative=0):
    c=np.asarray(coeff)
    for _ in range(derivative):c=c[1:]*np.arange(1,len(c))
    return np.polynomial.polynomial.polyval(t,c)


@dataclass
class FrenetConfig:
    horizons_s: tuple = (3.,4.,5.)
    dt_s: float = .1
    max_speed_mps: float = 12.
    max_accel_mps2: float = 3.5
    max_curvature: float = .35
    max_lateral_accel: float = 3.5
    clearance_m: float = .2
    max_initial_projection_s_m: float = 10.  # local route must start within 5 m behind ego


class FrenetPlanner:
    def __init__(self,config=None):self.config=config or FrenetConfig()

    def plan(self,obs):
        start=time.perf_counter();obs.validate();cfg=self.config
        ref=ReferenceLine(obs.route_xy);s0,d0=ref.project([0.,0.],s_max=cfg.max_initial_projection_s_m)
        _,yaw0,k0=ref.at(s0);tangent=np.array([np.cos(yaw0),np.sin(yaw0)]);normal=np.array([-tangent[1],tangent[0]])
        denom=1-k0*d0
        if denom<.2:raise ContractError("Frenet projection is near singular")
        v0=float(obs.ego.velocity_flu@tangent/denom);dv0=float(obs.ego.velocity_flu@normal)
        # Initial acceleration projection is a planar approximation; validate against lab dynamics.
        a0=float(obs.ego.acceleration_flu@tangent);da0=float(obs.ego.acceleration_flu@normal)
        if v0<-.1:raise ContractError("Reference implementation does not plan reverse driving")
        stop_s=None
        if obs.traffic.stop_required:
            stop_s,_=ref.project(obs.traffic.stop_line_xy.mean(axis=0),s_min=s0)
            stop_s-=obs.ego.length_m/2+obs.ego.bbox_center_flu[0]+cfg.clearance_m+.2
        best=None;rejects=Counter();tested=0
        for T in cfg.horizons_s:
            times=np.linspace(0,T,int(np.ceil(T/cfg.dt_s))+1)
            vt_values=[0.] if stop_s is not None else sorted(set([0.,obs.target_speed_mps*.6,obs.target_speed_mps]))
            for vt in vt_values:
                end_s=s0+max(0.,.5*(max(v0,0.)+vt)*T)
                if stop_s is not None:end_s=min(end_s,stop_s)
                if end_s>ref.s[-1]-.1:
                    if vt>0:continue
                    end_s=ref.s[-1]-.1
                if end_s<s0-1e-6:continue
                sc=quintic(s0,v0,a0,end_s,vt,0,T);ss=peval(sc,times);sd=peval(sc,times,1)
                if np.any(sd<-.01) or ss.min()<0 or ss.max()>ref.s[-1]:continue
                center,heading,k=ref.at(ss)
                normals=np.stack([-np.sin(heading),np.cos(heading)],axis=1)
                tangents=np.stack([np.cos(heading),np.sin(heading)],axis=1)
                for target_d in obs.lateral_targets_m:
                    tested+=1
                    dc=quintic(d0,dv0,da0,float(target_d),0,0,T)
                    dd=peval(dc,times);ddot=peval(dc,times,1);scale=1-k*dd
                    if np.any(scale<.2):rejects['frenet_singularity']+=1;continue
                    xy=center+normals*dd[:,None]
                    velocity=tangents*(sd*scale)[:,None]+normals*ddot[:,None]
                    speed=np.linalg.norm(velocity,axis=1)
                    yaw=heading+np.arctan2(ddot,np.maximum(sd*scale,1e-9))
                    # At standstill heading remains the reference tangent.
                    yaw=np.where(speed<.05,heading,yaw)
                    yaw=np.unwrap(yaw)
                    acceleration=np.gradient(speed,times,edge_order=2)
                    curvature=np.divide(np.gradient(yaw,times,edge_order=2),speed,out=np.zeros_like(speed),where=speed>.1)
                    status='STOP' if vt==0 else 'FEASIBLE'
                    plan=Plan(obs.ego.stamp,times,xy,yaw,speed,acceleration,curvature,status,'candidate')
                    valid,why=validate_path(plan,obs,margin_m=cfg.clearance_m,max_speed=cfg.max_speed_mps,
                        max_accel=cfg.max_accel_mps2,max_curvature=cfg.max_curvature,max_lateral_accel=cfg.max_lateral_accel)
                    if not valid:rejects[why]+=1;continue
                    j=np.gradient(acceleration,times)
                    jl=np.gradient(speed*speed*curvature,times)
                    plan.cost=float(.15*np.mean(j*j+jl*jl)+.3*np.mean(dd*dd)+2*(vt-obs.target_speed_mps)**2-.15*(end_s-s0))
                    plan.reason=why
                    if best is None or plan.cost<best.cost:best=plan
        if best is None:best=Plan.emergency(obs.ego.stamp,'no_feasible_candidate')
        best.diagnostics={'tested_candidates':tested,'rejections':dict(rejects),'planner_ms':(time.perf_counter()-start)*1000,
                          'implementation':'python_frenet_reference_v0.1','continuous_safety_proof':False}
        return best
