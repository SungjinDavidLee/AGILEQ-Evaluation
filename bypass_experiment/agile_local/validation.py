"""Sampled swept footprint checks; not a continuous-time safety proof."""
import numpy as np
from .frames import grid_indices


def corners(xy,yaw,length,width,offset=(0.,0.),margin=0.):
    local=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*np.array([length/2+margin,width/2+margin])+offset
    c,s=np.cos(yaw),np.sin(yaw);rot=np.array([[c,-s],[s,c]])
    return local@rot.T+xy


def overlap(a,b):
    for polygon in (a,b):
        edges=np.roll(polygon,-1,axis=0)-polygon
        for edge in edges:
            axis=np.array([-edge[1],edge[0]])
            pa=a@axis;pb=b@axis
            if pa.max()<pb.min() or pb.max()<pa.min():return False
    return True


def validate_path(plan,obs,*,margin_m=.15,max_speed=12.,max_accel=3.5,max_curvature=.35,max_lateral_accel=3.5):
    n=len(plan.t)
    if n<2:return False,"empty_or_short"
    if np.asarray(plan.xy).shape!=(n,2) or any(np.asarray(a).shape!=(n,) for a in [plan.t,plan.yaw,plan.speed,plan.acceleration,plan.curvature]):
        return False,"shape"
    if any(not np.isfinite(a).all() for a in [plan.t,plan.xy,plan.yaw,plan.speed,plan.acceleration,plan.curvature]):
        return False,"nonfinite"
    s=obs.ego.stamp
    if plan.stamp.episode!=s.episode or plan.stamp.frame!=s.frame or abs(plan.stamp.time_s-s.time_s)>1e-4:
        return False,"plan_stamp"
    if abs(plan.t[0])>1e-6 or np.any(np.diff(plan.t)<=0):return False,"time_order"
    if np.linalg.norm(plan.xy[0])>.05:return False,"initial_position"
    yaw_error=np.arctan2(np.sin(plan.yaw[0]),np.cos(plan.yaw[0]))
    if abs(yaw_error)>.08:return False,"initial_heading"
    if abs(plan.speed[0]-np.linalg.norm(obs.ego.velocity_flu))>.1:return False,"initial_speed"
    if plan.status not in {'FEASIBLE','STOP'}:return False,"status"
    if plan.status=='STOP' and plan.speed[-1]>.1:return False,"stop_terminal_speed"
    if plan.speed.min()<-.01 or plan.speed.max()>max_speed:return False,"speed"
    if np.max(np.abs(plan.acceleration))>max_accel:return False,"longitudinal_acceleration"
    if np.max(np.abs(plan.curvature))>max_curvature:return False,"curvature"
    if np.max(plan.speed**2*np.abs(plan.curvature))>max_lateral_accel:return False,"lateral_acceleration"
    if any(p.times_s[-1]<plan.t[-1]-1e-6 for p in obs.predictions):return False,"prediction_horizon"
    bev=obs.bev;ego=obs.ego
    step=min(bev.dx,bev.dy)/2
    # Fill interior, not only corners, to catch thin obstacles and solid markings.
    lx=np.linspace(-ego.length_m/2-margin_m,ego.length_m/2+margin_m,int(np.ceil((ego.length_m+2*margin_m)/step))+1)
    ly=np.linspace(-ego.width_m/2-margin_m,ego.width_m/2+margin_m,int(np.ceil((ego.width_m+2*margin_m)/step))+1)
    footprint=np.stack(np.meshgrid(lx,ly,indexing="ij"),axis=-1).reshape(-1,2)+np.array(ego.bbox_center_flu)
    times=[plan.t[0]]
    radius=np.hypot(ego.length_m/2,ego.width_m/2)+np.linalg.norm(ego.bbox_center_flu)+margin_m
    yaw_unwrapped=np.unwrap(plan.yaw)
    for i in range(1,len(plan.t)):
        travel=np.linalg.norm(plan.xy[i]-plan.xy[i-1])+radius*abs(yaw_unwrapped[i]-yaw_unwrapped[i-1])
        n=max(1,int(np.ceil(travel/step)),int(np.ceil((plan.t[i]-plan.t[i-1])/.05)))
        times.extend(np.linspace(plan.t[i-1],plan.t[i],n+1)[1:])
    for t in times:
        xy=np.array([np.interp(t,plan.t,plan.xy[:,i]) for i in range(2)])
        yaw=np.interp(t,plan.t,yaw_unwrapped);c,s=np.cos(yaw),np.sin(yaw)
        cloud=footprint@np.array([[c,s],[-s,c]])+xy
        rc,inside=grid_indices(cloud,bev)
        if not inside.all():return False,"outside_bev"
        r,cc=rc[:,0],rc[:,1]
        if not np.all(bev.free[r,cc]) or np.any(bev.occupied[r,cc]):return False,"occupancy"
        if np.any(bev.forbidden[r,cc]):return False,"forbidden_boundary"
        body=corners(xy,yaw,ego.length_m,ego.width_m,ego.bbox_center_flu,margin_m)
        if obs.traffic.stop_required:
            distance=(body-obs.traffic.stop_line_xy[0])@obs.traffic.approach_unit
            if distance.max()>-.05:return False,"stop_line"
        for obj in obs.predictions:
            oxy=np.array([np.interp(t,obj.times_s,obj.xy[:,j]) for j in range(2)])
            oyaw=np.interp(t,obj.times_s,np.unwrap(obj.yaw))
            if overlap(body,corners(oxy,oyaw,obj.length_m,obj.width_m,margin=obj.margin_m)):
                return False,"predicted_collision"
    return True,"sampled_constraints_passed"
