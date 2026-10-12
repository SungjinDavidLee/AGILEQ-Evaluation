import numpy as np
from .contracts import finite_array, ContractError
from .frames import world_to_flu


class ReferenceLine:
    """Dense polyline reference. Use a smooth lane connector for intersections."""
    def __init__(self,xy):
        a=finite_array(xy)
        if a.ndim!=2 or a.shape[1]!=2 or len(a)<3:raise ContractError("Invalid reference polyline")
        keep=np.r_[True,np.linalg.norm(np.diff(a,axis=0),axis=1)>1e-6]
        self.xy=a[keep]
        if len(self.xy)<3:raise ContractError("Degenerate reference")
        self.s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.xy,axis=0),axis=1))]
        g=np.gradient(self.xy,self.s,axis=0)
        self.yaw=np.unwrap(np.arctan2(g[:,1],g[:,0]))
        self.k=np.gradient(self.yaw,self.s)

    def at(self,s):
        s=np.asarray(s,dtype=float)
        if np.any(s<self.s[0]-1e-6) or np.any(s>self.s[-1]+1e-6):
            raise ContractError("Reference horizon exhausted; do not extrapolate")
        xy=np.stack([np.interp(s,self.s,self.xy[:,i]) for i in range(2)],axis=-1)
        return xy,np.interp(s,self.s,self.yaw),np.interp(s,self.s,self.k)

    def project(self,p,*,s_min=0.,s_max=None):
        p=finite_array(p,(2,))
        s_max=self.s[-1] if s_max is None else min(s_max,self.s[-1])
        if s_min<0 or s_max<s_min:raise ContractError('Invalid reference projection window')
        lo=np.maximum(self.s[:-1],s_min);hi=np.minimum(self.s[1:],s_max)
        allowed=hi>=lo
        if not allowed.any():raise ContractError('No segment in projection window')
        def tangent_residual(s):
            q,yaw,_=self.at(s)
            return float(np.dot(p-q,[np.cos(yaw),np.sin(yaw)]))
        # Project using the same normal that at(s) uses for reconstruction.
        # Segment-tangent projection combined with an interpolated normal can shift ego.
        roots=[]
        for a,b in zip(lo[allowed],hi[allowed]):
            fa,fb=tangent_residual(a),tangent_residual(b)
            if abs(fa)<1e-9:roots.append(a)
            if abs(fb)<1e-9:roots.append(b)
            if fa*fb<0:
                for _ in range(35):
                    mid=(a+b)/2;fm=tangent_residual(mid)
                    if fa*fm<=0:b=mid
                    else:a=mid;fa=fm
                roots.append((a+b)/2)
        candidates=roots or [s_min,s_max]
        s=min(candidates,key=lambda z:np.linalg.norm(self.at(z)[0]-p))
        q,yaw,_=self.at(s);lateral=np.dot(p-q,[-np.sin(yaw),np.cos(yaw)])
        return float(s),float(lateral)


class RouteCursor:
    """Monotonic local-window progress; reset for every new episode/route."""
    def __init__(self,world_xyz):
        self.xyz=finite_array(world_xyz)
        if self.xyz.ndim!=2 or self.xyz.shape[1]!=3 or len(self.xyz)<3:
            raise ContractError("Route needs world [N,3] coordinates")
        if np.any(np.linalg.norm(np.diff(self.xyz[:,:2],axis=0),axis=1)<1e-6):
            raise ContractError("Deduplicate the route before RouteCursor")
        self.ref=ReferenceLine(self.xyz[:,:2]);self.progress=0.

    def update(self,world_xyz,world_forward_xy,max_advance_m):
        if max_advance_m<=0:raise ContractError("Invalid route progress window")
        p=finite_array(world_xyz,(3,))[:2];heading=finite_array(world_forward_xy,(2,))
        norm=np.linalg.norm(heading)
        if norm<1e-6:raise ContractError("Missing heading")
        heading=heading/norm
        candidates=[]
        for i in range(len(self.xyz)-1):
            lo=max(self.ref.s[i],self.progress)
            hi=min(self.ref.s[i+1],self.progress+max_advance_m)
            if hi<lo:continue
            a=self.xyz[i,:2];d=self.xyz[i+1,:2]-a;length=np.linalg.norm(d)
            if np.dot(d/length,heading)<.3:continue
            u=np.clip(np.dot(p-a,d)/(length*length),(lo-self.ref.s[i])/length,(hi-self.ref.s[i])/length)
            candidates.append((np.linalg.norm(p-(a+u*d)),self.ref.s[i]+u*length))
        if not candidates:raise ContractError("No forward route match; reset/relocalize explicitly")
        dist,s=min(candidates)
        if dist>8:raise ContractError("Ego is too far from active route")
        self.progress=max(self.progress,float(s));return self.progress

    def local(self,world_from_ego,behind_m=5.,ahead_m=60.,spacing_m=.5):
        if spacing_m<=0:raise ContractError("Invalid route sampling interval")
        a=max(0,self.progress-behind_m);b=min(self.ref.s[-1],self.progress+ahead_m)
        s=np.linspace(a,b,max(3,int(np.ceil((b-a)/spacing_m))+1))
        xyz=np.stack([np.interp(s,self.ref.s,self.xyz[:,i]) for i in range(3)],axis=1)
        return world_to_flu(xyz,world_from_ego)[:,:2]


def carla_global_route(carla_map,start_location,destination_location,resolution_m=1.0):
    """CARLA 0.9.15-style agent API; matching PythonAPI/carla must be importable."""
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    planner=GlobalRoutePlanner(carla_map,resolution_m)
    trace=planner.trace_route(start_location,destination_location)
    xyz=[];options=[]
    for wp,option in trace:
        p=wp.transform.location;v=[p.x,p.y,p.z]
        if not xyz or np.linalg.norm(np.asarray(v)[:2]-np.asarray(xyz[-1])[:2])>1e-6:
            xyz.append(v);options.append(str(option))
    if len(xyz)<3:raise ContractError("Global route creation failed")
    return np.asarray(xyz),options
