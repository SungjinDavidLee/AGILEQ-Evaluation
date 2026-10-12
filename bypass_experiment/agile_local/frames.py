"""CARLA world (LH) <-> ego FLU (RH). No implicit nuScenes conversion."""
import numpy as np
from .contracts import finite_array, ContractError

FLIP = np.diag([1.,-1.,1.,1.])


def check_transform(matrix):
    m=finite_array(matrix,(4,4))
    if not np.allclose(m[3],[0,0,0,1]) or not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-5):
        raise ContractError("Expected a rigid 4x4 transform")
    return m


def transform_points(points, matrix):
    p=finite_array(points)
    if p.ndim!=2 or p.shape[1] not in (2,3):
        raise ContractError("Expected [N,2] or [N,3] points")
    if p.shape[1]==2:p=np.c_[p,np.zeros(len(p))]
    return (np.c_[p,np.ones(len(p))]@check_transform(matrix).T)[:,:3]


def world_to_flu(points_world, world_from_carla_ego):
    return transform_points(points_world,FLIP@np.linalg.inv(check_transform(world_from_carla_ego)))


def flu_to_world(points_flu, world_from_carla_ego):
    return transform_points(points_flu,check_transform(world_from_carla_ego)@FLIP)


def world_vector_to_flu(v, world_from_carla_ego):
    """Velocity/direction: rotate only, never add translation."""
    return (FLIP[:3,:3]@check_transform(world_from_carla_ego)[:3,:3].T@finite_array(v,(3,)))


def reanchor_points(points, old_pose, new_pose):
    """Static points only. Dynamic predictions also require time advancement."""
    return world_to_flu(flu_to_world(points,old_pose),new_pose)


def grid_indices(xy, bev):
    p=finite_array(xy)
    rc=np.floor((p-np.array([bev.x_min,bev.y_min]))/np.array([bev.dx,bev.dy])).astype(int)
    inside=(rc[:,0]>=0)&(rc[:,1]>=0)&(rc[:,0]<bev.free.shape[0])&(rc[:,1]<bev.free.shape[1])
    return rc,inside


def grid_centers(bev):
    r,c=np.indices(bev.free.shape)
    return np.stack([bev.x_min+(r+.5)*bev.dx,bev.y_min+(c+.5)*bev.dy],axis=-1)


def carla_bgra_to_rgb(image):
    return np.frombuffer(image.raw_data,dtype=np.uint8).reshape(image.height,image.width,4)[:,:,:3][:,:,::-1].copy()


def world_to_camera_cv(points_world, world_from_camera_carla):
    """CARLA camera [forward,right,up] -> OpenCV [right,down,forward]."""
    local=transform_points(points_world,np.linalg.inv(check_transform(world_from_camera_carla)))
    return np.stack([local[:,1],-local[:,2],local[:,0]],axis=1)
