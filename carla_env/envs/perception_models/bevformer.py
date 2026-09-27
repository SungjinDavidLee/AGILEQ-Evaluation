import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision
from efficientnet_pytorch import EfficientNet
import math
from .ops.modules import MSDeformAttn, MSDeformAttn3D
from .models import build_model


scene_centroid_x = 0.0
scene_centroid_y = 1.0
scene_centroid_z = 0.0

scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()
XMIN, XMAX = -20, 20
ZMIN, ZMAX = -20, 20
YMIN, YMAX = -5, 5
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
EPS = 1e-6

Z, Y, X = 200, 8, 200
def pack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    B_, S = shapelist[:2]
    assert(B==B_)
    otherdims = shapelist[2:]
    tensor = torch.reshape(tensor, [B*S]+otherdims)
    return tensor
def eye_4x4(B, device='cuda'):
    rt = torch.eye(4, device=torch.device(device)).view(1,4,4).repeat([B, 1, 1])
    return rt
def unpack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    assert(BS%B==0)
    otherdims = shapelist[1:]
    S = int(BS/B)
    tensor = torch.reshape(tensor, [B,S]+otherdims)
    return tensor
def split_intrinsics(K):

    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0
def merge_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    K = torch.zeros(B, 4, 4, dtype=torch.float32, device=fx.device)
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = x0
    K[:,1,2] = y0
    K[:,2,2] = 1.0
    K[:,3,3] = 1.0
    return K
def merge_rt(r, t):


    B, C, D = list(r.shape)
    B2, D2 = list(t.shape)
    assert(C==3)
    assert(D==3)
    assert(B==B2)
    assert(D2==3)
    t = t.view(B, 3)
    rt = eye_4x4(B, device=t.device)
    rt[:,:3,:3] = r
    rt[:,:3,3] = t
    return rt
def merge_rtlist(rlist, tlist):
    B, N, D, E = list(rlist.shape)
    assert(D==3)
    assert(E==3)
    B, N, F = list(tlist.shape)
    assert(F==3)

    __p = lambda x: pack_seqdim(x, B)
    __u = lambda x: unpack_seqdim(x, B)
    rlist_, tlist_ = __p(rlist), __p(tlist)
    rtlist_ = merge_rt(rlist_, tlist_)
    rtlist = __u(rtlist_)
    return rtlist
def get_camM_T_camXs(origin_T_camXs, ind=0):
    B, S = list(origin_T_camXs.shape)[0:2]
    camM_T_camXs = torch.zeros_like(origin_T_camXs)
    for b in list(range(B)):
        camM_T_origin = safe_inverse_single(origin_T_camXs[b,ind])
        for s in list(range(S)):
            camM_T_camXs[b,s] = torch.matmul(camM_T_origin, origin_T_camXs[b,s])
    return camM_T_camXs
def safe_inverse(a):
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2)

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv
def split_rt_single(rt):
    r = rt[:3, :3]
    t = rt[:3, 3].view(3)
    return r, t
def safe_inverse_single(a):
    r, t = split_rt_single(a)
    t = t.view(3,1)
    r_transpose = r.t()
    inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
    bottom_row = a[3:4, :]
    inv = torch.cat([inv, bottom_row], 0)
    return inv
def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)

    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:,:,:3]
    return xyz2
def matmul2(mat1, mat2):
    return torch.matmul(mat1, mat2)
def normalize_grid3d(grid_z, grid_y, grid_x, Z, Y, X, clamp_extreme=True):

    grid_z = 2.0*(grid_z / float(Z-1)) - 1.0
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0

    if clamp_extreme:
        grid_z = torch.clamp(grid_z, min=-2.0, max=2.0)
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)

    return grid_z, grid_y, grid_x
def meshgrid3d(B, Z, Y, X, stack=False, norm=False, device='cuda'):


    grid_z = torch.linspace(0.0, Z-1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y-1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)

    if norm:
        grid_z, grid_y, grid_x = normalize_grid3d(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:


        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x
def gridcloud3d(B, Z, Y, X, norm=False, device='cuda'):

    grid_z, grid_y, grid_x = meshgrid3d(B, Z, Y, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])

    xyz = torch.stack([x, y, z], dim=2)

    return xyz
def normalize_grid2d(grid_y, grid_x, Y, X, clamp_extreme=True):

    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0

    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)

    return grid_y, grid_x
def meshgrid2d(B, Y, X, stack=False, norm=False, device='cuda'):


    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2d(
            grid_y, grid_x, Y, X)

    if stack:


        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x
def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx*sx
    fy = fy*sy
    x0 = x0*sx
    y0 = y0*sy
    K = merge_intrinsics(fx, fy, x0, y0)
    return K

class Vox_util(object):
    def __init__(self, Z, Y, X, scene_centroid, bounds, pad=None, assert_cube=False):
        self.XMIN, self.XMAX, self.YMIN, self.YMAX, self.ZMIN, self.ZMAX = bounds
        B, D = list(scene_centroid.shape)
        self.Z, self.Y, self.X = Z, Y, X

        scene_centroid = scene_centroid.detach().cpu().numpy()
        x_centroid, y_centroid, z_centroid = scene_centroid[0]
        self.XMIN += x_centroid
        self.XMAX += x_centroid
        self.YMIN += y_centroid
        self.YMAX += y_centroid
        self.ZMIN += z_centroid
        self.ZMAX += z_centroid

        self.default_vox_size_X = (self.XMAX-self.XMIN)/float(X)
        self.default_vox_size_Y = (self.YMAX-self.YMIN)/float(Y)
        self.default_vox_size_Z = (self.ZMAX-self.ZMIN)/float(Z)

        if pad:
            Z_pad, Y_pad, X_pad = pad
            self.ZMIN -= self.default_vox_size_Z * Z_pad
            self.ZMAX += self.default_vox_size_Z * Z_pad
            self.YMIN -= self.default_vox_size_Y * Y_pad
            self.YMAX += self.default_vox_size_Y * Y_pad
            self.XMIN -= self.default_vox_size_X * X_pad
            self.XMAX += self.default_vox_size_X * X_pad

        if assert_cube:

            if (not np.isclose(self.default_vox_size_X, self.default_vox_size_Y)) or (not np.isclose(self.default_vox_size_X, self.default_vox_size_Z)):
                print('Z, Y, X', Z, Y, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX),
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX),
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX),
                )
                print('self.default_vox_size_X', self.default_vox_size_X)
                print('self.default_vox_size_Y', self.default_vox_size_Y)
                print('self.default_vox_size_Z', self.default_vox_size_Z)
            assert(np.isclose(self.default_vox_size_X, self.default_vox_size_Y))
            assert(np.isclose(self.default_vox_size_X, self.default_vox_size_Z))

    def Ref2Mem(self, xyz, Z, Y, X, assert_cube=False):

        B, N, C = list(xyz.shape)
        device = xyz.device
        assert(C==3)
        mem_T_ref = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=device)
        xyz = apply_4x4(mem_T_ref, xyz)
        return xyz

    def Mem2Ref(self, xyz_mem, Z, Y, X, assert_cube=False):

        B, N, C = list(xyz_mem.shape)
        ref_T_mem = self.get_ref_T_mem(B, Z, Y, X, assert_cube=assert_cube, device=xyz_mem.device)
        xyz_ref = apply_4x4(ref_T_mem, xyz_mem)
        return xyz_ref

    def get_mem_T_ref(self, B, Z, Y, X, assert_cube=False, device='cuda'):
        vox_size_X = (self.XMAX-self.XMIN)/float(X)
        vox_size_Y = (self.YMAX-self.YMIN)/float(Y)
        vox_size_Z = (self.ZMAX-self.ZMIN)/float(Z)

        if assert_cube:
            if (not np.isclose(vox_size_X, vox_size_Y)) or (not np.isclose(vox_size_X, vox_size_Z)):
                print('Z, Y, X', Z, Y, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX),
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX),
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX),
                )
                print('vox_size_X', vox_size_X)
                print('vox_size_Y', vox_size_Y)
                print('vox_size_Z', vox_size_Z)
            assert(np.isclose(vox_size_X, vox_size_Y))
            assert(np.isclose(vox_size_X, vox_size_Z))
        center_T_ref = eye_4x4(B, device=device)
        center_T_ref[:,0,3] = -self.XMIN-vox_size_X/2.0
        center_T_ref[:,1,3] = -self.YMIN-vox_size_Y/2.0
        center_T_ref[:,2,3] = -self.ZMIN-vox_size_Z/2.0

        mem_T_center = eye_4x4(B, device=device)
        mem_T_center[:,0,0] = 1./vox_size_X
        mem_T_center[:,1,1] = 1./vox_size_Y
        mem_T_center[:,2,2] = 1./vox_size_Z
        mem_T_ref = matmul2(mem_T_center, center_T_ref)

        return mem_T_ref

    def get_ref_T_mem(self, B, Z, Y, X, assert_cube=False, device='cuda'):
        mem_T_ref = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=device)
        ref_T_mem = mem_T_ref.inverse()
        return ref_T_mem

    def get_inbounds(self, xyz, Z, Y, X, already_mem=False, padding=0.0, assert_cube=False):

        if not already_mem:
            xyz = self.Ref2Mem(xyz, Z, Y, X, assert_cube=assert_cube)

        x = xyz[:,:,0]
        y = xyz[:,:,1]
        z = xyz[:,:,2]

        x_valid = ((x-padding)>-0.5).byte() & ((x+padding)<float(X-0.5)).byte()
        y_valid = ((y-padding)>-0.5).byte() & ((y+padding)<float(Y-0.5)).byte()
        z_valid = ((z-padding)>-0.5).byte() & ((z+padding)<float(Z-0.5)).byte()
        nonzero = (~(z==0.0)).byte()

        inbounds = x_valid & y_valid & z_valid & nonzero
        return inbounds.bool()

    def voxelize_xyz(self, xyz_ref, Z, Y, X, already_mem=False, assert_cube=False, clean_eps=0):
        B, N, D = list(xyz_ref.shape)
        assert(D==3)
        if already_mem:
            xyz_mem = xyz_ref
        else:
            xyz_mem = self.Ref2Mem(xyz_ref, Z, Y, X, assert_cube=assert_cube)
            xyz_zero = self.Ref2Mem(xyz_ref[:,0:1]*0, Z, Y, X, assert_cube=assert_cube)
        vox = self.get_occupancy(xyz_mem, Z, Y, X, clean_eps=clean_eps, xyz_zero=xyz_zero)
        return vox

    def voxelize_xyz_and_feats(self, xyz_ref, feats, Z, Y, X, already_mem=False, assert_cube=False, clean_eps=0):
        B, N, D = list(xyz_ref.shape)
        B2, N2, D2 = list(feats.shape)
        assert(D==3)
        assert(B==B2)
        assert(N==N2)
        if already_mem:
            xyz_mem = xyz_ref
        else:
            xyz_mem = self.Ref2Mem(xyz_ref, Z, Y, X, assert_cube=assert_cube)
            xyz_zero = self.Ref2Mem(xyz_ref[:,0:1]*0, Z, Y, X, assert_cube=assert_cube)
        feats = self.get_feat_occupancy(xyz_mem, feats, Z, Y, X, clean_eps=clean_eps, xyz_zero=xyz_zero)
        return feats

    def get_occupancy(self, xyz, Z, Y, X, clean_eps=0, xyz_zero=None):

        B, N, C = list(xyz.shape)
        assert(C==3)

        inbounds = self.get_inbounds(xyz, Z, Y, X, already_mem=True)
        x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]
        mask = torch.zeros_like(x)
        mask[inbounds] = 1.0

        if xyz_zero is not None:
            dist = torch.norm(xyz_zero-xyz, dim=2)
            mask[dist < 0.1] = 0

        if clean_eps > 0:
            xyz_round = torch.round(xyz)
            dist = torch.norm(xyz_round - xyz, dim=2)
            mask[dist > clean_eps] = 0

        x = x*mask
        y = y*mask
        z = z*mask

        x = torch.round(x)
        y = torch.round(y)
        z = torch.round(z)
        x = torch.clamp(x, 0, X-1).int()
        y = torch.clamp(y, 0, Y-1).int()
        z = torch.clamp(z, 0, Z-1).int()

        x = x.view(B*N)
        y = y.view(B*N)
        z = z.view(B*N)

        dim3 = X
        dim2 = X * Y
        dim1 = X * Y * Z

        base = torch.arange(0, B, dtype=torch.int32, device=xyz.device)*dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B*N)

        vox_inds = base + z * dim2 + y * dim3 + x
        voxels = torch.zeros(B*Z*Y*X, device=xyz.device).float()
        voxels[vox_inds.long()] = 1.0
        voxels[base.long()] = 0.0
        voxels = voxels.reshape(B, 1, Z, Y, X)
        return voxels

    def get_feat_occupancy(self, xyz, feat, Z, Y, X, clean_eps=0, xyz_zero=None):
        B, N, C = list(xyz.shape)
        B2, N2, D2 = list(feat.shape)
        assert(C==3)
        assert(B==B2)
        assert(N==N2)
        inbounds = self.get_inbounds(xyz, Z, Y, X, already_mem=True)
        x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]
        mask = torch.zeros_like(x)
        mask[inbounds] = 1.0

        if xyz_zero is not None:
            dist = torch.norm(xyz_zero-xyz, dim=2)
            mask[dist < 0.1] = 0

        if clean_eps > 0:
            xyz_round = torch.round(xyz)
            dist = torch.norm(xyz_round - xyz, dim=2)
            mask[dist > clean_eps] = 0

        x = x*mask
        y = y*mask
        z = z*mask
        feat = feat*mask.unsqueeze(-1)

        x = torch.round(x)
        y = torch.round(y)
        z = torch.round(z)
        x = torch.clamp(x, 0, X-1).int()
        y = torch.clamp(y, 0, Y-1).int()
        z = torch.clamp(z, 0, Z-1).int()


        perm = torch.randperm(N)
        x = x[:, perm]
        y = y[:, perm]
        z = z[:, perm]
        feat = feat[:, perm]

        x = x.view(B*N)
        y = y.view(B*N)
        z = z.view(B*N)
        feat = feat.view(B*N, -1)

        dim3 = X
        dim2 = X * Y
        dim1 = X * Y * Z

        base = torch.arange(0, B, dtype=torch.int32, device=xyz.device)*dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B*N)

        vox_inds = base + z * dim2 + y * dim3 + x
        feat_voxels = torch.zeros((B*Z*Y*X, D2), device=xyz.device).float()
        feat_voxels[vox_inds.long()] = feat

        feat_voxels[base.long()] = 0.0
        feat_voxels = feat_voxels.reshape(B, Z, Y, X, D2).permute(0, 4, 1, 2, 3)

        return feat_voxels

    def unproject_image_to_mem(self, rgb_camB, pixB_T_camA, camB_T_camA, Z, Y, X, assert_cube=False, xyz_camA=None):
        B, C, H, W = list(rgb_camB.shape)

        if xyz_camA is None:
            xyz_memA = gridcloud3d(B, Z, Y, X, norm=False, device=pixB_T_camA.device)
            xyz_camA = self.Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)

        xyz_camB = apply_4x4(camB_T_camA, xyz_camA)
        z = xyz_camB[:,:,2]

        xyz_pixB = apply_4x4(pixB_T_camA, xyz_camA)
        normalizer = torch.unsqueeze(xyz_pixB[:,:,2], 2)
        EPS=1e-6

        xy_pixB = xyz_pixB[:,:,:2]/torch.clamp(normalizer, min=EPS)

        x, y = xy_pixB[:,:,0], xy_pixB[:,:,1]

        x_valid = (x>-0.5).bool() & (x<float(W-0.5)).bool()
        y_valid = (y>-0.5).bool() & (y<float(H-0.5)).bool()
        z_valid = (z>0.0).bool()
        valid_mem = (x_valid & y_valid & z_valid).reshape(B, 1, Z, Y, X).float()


        y_pixB, x_pixB = normalize_grid2d(y, x, H, W)

        z_pixB = torch.zeros_like(x)
        xyz_pixB = torch.stack([x_pixB, y_pixB, z_pixB], axis=2)
        rgb_camB = rgb_camB.unsqueeze(2)
        xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
        values = F.grid_sample(rgb_camB, xyz_pixB, align_corners=False)

        values = torch.reshape(values, (B, C, Z, Y, X))
        values = values * valid_mem
        return values

    def warp_tiled_to_mem(self, rgb_tileB, pixB_T_camA, camB_T_camA, Z, Y, X, DMIN, DMAX, assert_cube=False):

        B, C, D, H, W = list(rgb_tileB.shape)

        xyz_memA = gridcloud3d(B, Z, Y, X, norm=False, device=pixB_T_camA.device)

        xyz_camA = self.Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)

        xyz_camB = apply_4x4(camB_T_camA, xyz_camA)
        z_camB = xyz_camB[:,:,2]

        z_tileB = (D-1.0) * (z_camB-float(DMIN)) / float(DMAX-DMIN)

        xyz_pixB = apply_4x4(pixB_T_camA, xyz_camA)
        normalizer = torch.unsqueeze(xyz_pixB[:,:,2], 2)
        EPS=1e-6

        xy_pixB = xyz_pixB[:,:,:2]/torch.clamp(normalizer, min=EPS)

        x, y = xy_pixB[:,:,0], xy_pixB[:,:,1]


        x_valid = (x>-0.5).bool() & (x<float(W-0.5)).bool()
        y_valid = (y>-0.5).bool() & (y<float(H-0.5)).bool()
        z_valid = (z_camB>0.0).bool()
        valid_mem = (x_valid & y_valid & z_valid).reshape(B, 1, Z, Y, X).float()

        z_tileB, y_pixB, x_pixB = normalize_grid3d(z_tileB, y, x, D, H, W)
        xyz_pixB = torch.stack([x_pixB, y_pixB, z_tileB], axis=2)
        xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
        values = F.grid_sample(rgb_tileB, xyz_pixB, align_corners=False)

        values = torch.reshape(values, (B, C, Z, Y, X))
        values = values * valid_mem
        return values


    def apply_mem_T_ref_to_lrtlist(self, lrtlist_cam, Z, Y, X, assert_cube=False):

        B, N, C = list(lrtlist_cam.shape)
        assert(C==19)
        mem_T_cam = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=lrtlist_cam.device)

    def xyz2circles(self, xyz, radius, Z, Y, X, soft=True, already_mem=True, also_offset=False, grid=None):

        B, N, D = list(xyz.shape)
        assert(D==3)
        if not already_mem:
            xyz = self.Ref2Mem(xyz, Z, Y, X)

        if grid is None:
            grid_z, grid_y, grid_x = meshgrid3d(B, Z, Y, X, stack=False, norm=False, device=xyz.device)

            grid = torch.stack([grid_x, grid_y, grid_z], dim=1)


        xyz = xyz.reshape(B, N, 3, 1, 1, 1)
        grid = grid.reshape(B, 1, 3, Z, Y, X)

        xyz = xyz.round()

        if torch.is_tensor(radius):
            radius = radius.clamp(min=0.01)

        if soft:
            off = grid - xyz

            dist_grid = torch.sum(off**2, dim=2, keepdim=False)

            if torch.is_tensor(radius):
                radius = radius.reshape(B, N, 1, 1, 1)
            mask = torch.exp(-dist_grid/(2*radius*radius))

            mask[mask < 0.001] = 0.0

            if also_offset:
                return mask, off
            else:
                return mask
        else:
            assert(False)

            dist_grid = torch.norm(grid - xyz, dim=2, keepdim=False)


            radius = radius.reshape(B, N, 1, 1, 1)

            within_radius_mask = (dist_grid < radius).float()
            within_radius_mask = torch.sum(within_radius_mask, dim=1, keepdim=True).clamp(0, 1)
            return within_radius_mask

    def xyz2circles_bev(self, xyz, radius, Z, Y, X, already_mem=True, also_offset=False):


        B, N, D = list(xyz.shape)
        assert(D==3)
        if not already_mem:
            xyz = self.Ref2Mem(xyz, Z, Y, X)

        xz = torch.stack([xyz[:,:,0], xyz[:,:,2]], dim=2)

        grid_z, grid_x = meshgrid2d(B, Z, X, stack=False, norm=False, device=xyz.device)

        grid = torch.stack([grid_x, grid_z], dim=1)


        xz = xz.reshape(B, N, 2, 1, 1)
        grid = grid.reshape(B, 1, 2, Z, X)


        xz = xz.round()

        if torch.is_tensor(radius):
            radius = radius.clamp(min=0.01)

        off = grid - xz

        dist_grid = torch.sum(off**2, dim=2, keepdim=False)

        if torch.is_tensor(radius):
            radius = radius.reshape(B, N, 1, 1, 1)
        mask = torch.exp(-dist_grid/(2*radius*radius))

        mask[mask < 0.001] = 0.0


        mask = mask.unsqueeze(-2)
        off = off.unsqueeze(-2)


        if also_offset:
            return mask, off
        else:
            return mask
def camera2pixels(xyz, pix_T_cam):


    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    x, y, z = torch.unbind(xyz, dim=-1)
    B = list(z.shape)[0]

    fx = torch.reshape(fx, [B,1])
    fy = torch.reshape(fy, [B,1])
    x0 = torch.reshape(x0, [B,1])
    y0 = torch.reshape(y0, [B,1])
    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])

    EPS = 1e-4
    z = torch.clamp(z, min=EPS)
    x = (x*fx)/z + x0
    y = (y*fy)/z + y0
    xy = torch.stack([x, y], dim=-1)
    return xy
def reduce_masked_mean(x, mask, dim=None, keepdim=False):


    for (a,b) in zip(x.size(), mask.size()):
        assert(a==b)
    prod = x*mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS+torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS+torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer/denom
    return mean
def GN(C, max_groups=32):

    for g in range(min(max_groups, C), 0, -1):
        if C % g == 0:
            return nn.GroupNorm(g, C)
    return nn.GroupNorm(1, C)
def make_head_from100(in_ch: int = 64, mid_ch: int = 64, align_corners: bool = False):

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners),
        nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
        GN(mid_ch),
        nn.ReLU(inplace=True),

        nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
        GN(mid_ch),
        nn.ReLU(inplace=True),

        nn.Conv2d(mid_ch, 1, kernel_size=1, padding=0),
    )
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            GN(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            GN(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)

        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class BevEncode_effi_B0(nn.Module):
    def __init__(self, inC):
        super(BevEncode_effi_B0, self).__init__()

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.trunk._conv_stem = nn.Conv2d(inC, 32, kernel_size=3, padding=1, bias=False)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(112, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),)

        self.up2 = nn.Sequential(
            nn.Upsample(size=(50, 50), mode='bilinear',
                        align_corners=True),
            nn.Conv2d(320, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),)


        self.up1_BEV = Up(64 + 64, 64)
        self.up2_BEV = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.up1_lane = Up(64 + 64, 64)
        self.up2_lane = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.up1_lane2 = Up(64 + 64, 64)
        self.up2_lane2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.up1_ped = Up(64 + 64, 64)
        self.up2_ped = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        self.up1_veh = Up(64 + 64, 64)
        self.up2_veh = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        self.up1_stop = Up(64 + 64, 64)
        self.up2_stop = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )


    def get_eff_depth(self, x):
        endpoints = []

        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x


        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints.append(prev_x)
            prev_x = x


        endpoints.append(x)
        del endpoints[0]
        return endpoints

    def forward(self, x):
        pyramid = self.get_eff_depth(x)

        ups=self.up1(pyramid[2])
        ups2=self.up2(pyramid[3])

        road = self.up1_BEV(ups2, ups)
        road = self.up2_BEV(road)

        lane_broken = self.up1_lane(ups2, ups)
        lane_broken = self.up2_lane(lane_broken)

        lane_solid = self.up1_lane2(ups2, ups)
        lane_solid = self.up2_lane2(lane_solid)

        veh = self.up1_veh(ups2, ups)
        veh = self.up2_veh(veh)

        ped = self.up1_ped(ups2, ups)
        ped = self.up2_ped(ped)

        stop = self.up1_stop(ups2, ups)
        stop = self.up2_stop(stop)


        return road, lane_broken,lane_solid, veh, ped, stop

class FPN(nn.Module):
    def __init__(self, in_channels=[128, 256, 512], out_channels=256):
        super().__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels
        ])

    def forward(self, x_list):

        laterals = [conv(x) for conv, x in zip(self.lateral_convs, x_list)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode='nearest')
            laterals[i-1] = laterals[i-1] + upsampled

        outs = [conv(x) for conv, x in zip(self.fpn_convs, laterals)]

        return outs
class Encoder_inter(nn.Module):
    def __init__(self, data_conf,C):
        super().__init__()
        self.C=C
        config=data_conf['config']

        self.camencode = build_model(config)

        self.fpn = FPN(in_channels=[128, 256, 512], out_channels=self.C)

    def forward(self,x):
        x = self.camencode(x)
        x=self.fpn(x[1:])

        return x


class VanillaSelfAttention(nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super(VanillaSelfAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn(d_model=dim, n_levels=1, n_heads=4, n_points=8)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, query_pos=None):

        inp_residual = query.clone()

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        device = query.device
        Z, X = 200, 200
        ref_z, ref_x = torch.meshgrid(
            torch.linspace(0.5, Z-0.5, Z, dtype=torch.float, device=query.device),
            torch.linspace(0.5, X-0.5, X, dtype=torch.float, device=query.device)
        )
        ref_z = ref_z.reshape(-1)[None] / Z
        ref_x = ref_x.reshape(-1)[None] / X
        reference_points = torch.stack((ref_z, ref_x), -1)
        reference_points = reference_points.repeat(B, 1, 1).unsqueeze(2)

        B, N, C = query.shape
        input_spatial_shapes = query.new_zeros([1,2]).long()
        input_spatial_shapes[:] = 200
        input_level_start_index = query.new_zeros([1,]).long()
        queries = self.deformable_attention(query, reference_points, query.clone(),
            input_spatial_shapes, input_level_start_index)

        queries = self.output_proj(queries)

        return self.dropout(queries) + inp_residual

class SpatialCrossAttention(nn.Module):


    def __init__(self, dim=128, dropout=0.1):
        super(SpatialCrossAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn3D(embed_dims=dim, num_heads=4, num_levels=3, num_points=8)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value,level_start_index=None, query_pos=None, reference_points_cam=None, spatial_shapes=None, bev_mask=None):

        inp_residual = query
        slots = torch.zeros_like(query)

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        S, M, _, _ = key.shape

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        queries_rebatch = query.new_zeros(
            [B, S, max_len, self.dim])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [B, S, max_len, D, 2])

        for j in range(B):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        key = key.permute(2, 0, 1, 3).reshape(
            B * S, M, C)
        value = value.permute(2, 0, 1, 3).reshape(
            B * S, M, C)

        queries = self.deformable_attention(query=queries_rebatch.view(B*S, max_len, self.dim),
            key=key, value=value,
            reference_points=reference_points_rebatch.view(B*S, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index).view(B, S, max_len, self.dim)

        for j in range(B):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


class bevformer(nn.Module):
    def __init__(self, data_conf,):
        super(bevformer, self).__init__()
        self.data_conf = data_conf

        self.feat2d_dim = feat2d_dim = latent_dim = 256

        self.encoder = Encoder_inter(data_conf,feat2d_dim)

        self.decoder = BevEncode_effi_B0(inC=feat2d_dim)
        self.vox_util = Vox_util(
            Z, Y, X,
            scene_centroid=scene_centroid.cuda(),
            bounds=bounds,
            assert_cube=False)


        self.bev_queries = nn.Parameter(0.1*torch.randn(latent_dim, Z, X))
        self.bev_queries_pos = nn.Parameter(0.1*torch.randn(latent_dim, Z, X))
        num_layers = 6
        self.num_layers = num_layers

        self.self_attn_layers = nn.ModuleList([
            VanillaSelfAttention(dim=latent_dim) for _ in range(num_layers)
        ])

        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])


        self.cross_attn_layers = nn.ModuleList([
            SpatialCrossAttention(dim=latent_dim) for _ in range(num_layers)
        ])

        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])
        ffn_dim = 1028
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, latent_dim)) for _ in range(num_layers)
        ])
        self.norm3_layers = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])

    def forward(self, bev_imgs,cam_pam):
        B, N, C, H, W = bev_imgs.shape
        __p = lambda x: pack_seqdim(x, B)
        __u = lambda x: unpack_seqdim(x, B)


        intrins_ = __p(cam_pam['intrins'])
        pix_T_cams_ = merge_intrinsics(*split_intrinsics(intrins_)).cuda()
        pix_T_cams = __u(pix_T_cams_)

        ego_T_camXs = merge_rtlist(
            cam_pam['rots'],
            cam_pam['trans']
        ).cuda()

        bev_imgs = __p(bev_imgs)

        pix_T_cams_ = __p(pix_T_cams)


        ego_T_camXs_ = __p(ego_T_camXs)
        camXs_T_ego_ = safe_inverse(ego_T_camXs_)
        feat_flatten = []
        spatial_shapes = []

        mlvl_feats = self.encoder(bev_imgs)
        for lvl, feat in enumerate(mlvl_feats):
            BN, C, Hf, Wf = feat.shape

            feat = feat.view(B, N, C, Hf, Wf)
            spatial_shapes.append([Hf, Wf])
            feat = feat.reshape(B, N, C, Hf*Wf).permute(1, 3, 0, 2)
            feat_flatten.append(feat)

        bev_keys = torch.cat(feat_flatten, 1)

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_keys.device)

        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ))


        bev_queries = self.bev_queries.clone().unsqueeze(0).repeat(B,1,1,1).reshape(B, self.feat2d_dim, -1).permute(0,2,1)


        bev_queries_pos = self.bev_queries_pos.clone().unsqueeze(0).repeat(B,1,1,1).reshape(B, self.feat2d_dim, -1).permute(0,2,1)

        xyz_mem_ = gridcloud3d(B * N, Z, Y, X, norm=False, device=bev_imgs.device)


        xyz_ego_ = self.vox_util.Mem2Ref(
            xyz_mem_,
            Z,
            Y,
            X,
            assert_cube=False
        )


        xyz_camXs_ = apply_4x4(camXs_T_ego_, xyz_ego_)


        xy_camXs_ = camera2pixels(xyz_camXs_, pix_T_cams_)
        reference_points_cam = xy_camXs_.reshape(B, N, Z, Y, X, 2).permute(1, 0, 2, 4, 3, 5).reshape(N, B, Z*X, Y, 2)
        reference_points_cam[..., 0:1] = reference_points_cam[..., 0:1] / float(W)
        reference_points_cam[..., 1:2] = reference_points_cam[..., 1:2] / float(H)


        bev_mask = ((reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0)).squeeze(-1)


        for i in range(self.num_layers):
            bev_queries = self.self_attn_layers[i](bev_queries, bev_queries_pos)
            bev_queries = self.norm1_layers[i](bev_queries)

            bev_queries = self.cross_attn_layers[i](bev_queries, bev_keys, bev_keys,
                level_start_index=level_start_index,
                query_pos=bev_queries_pos,
                reference_points_cam = reference_points_cam,
                spatial_shapes = spatial_shapes,
                bev_mask = bev_mask)
            bev_queries = self.norm2_layers[i](bev_queries)
            bev_queries = bev_queries + self.ffn_layers[i](bev_queries)
            bev_queries = self.norm3_layers[i](bev_queries)

        feat_bev = bev_queries.permute(0, 2, 1).reshape(B, self.feat2d_dim, Z, X).contiguous()


        road, lane_broken,lane_solid, veh, ped, stop=self.decoder(feat_bev)

        return road, lane_broken,lane_solid, veh, ped, stop
