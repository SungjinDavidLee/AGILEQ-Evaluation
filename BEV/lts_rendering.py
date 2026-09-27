import torch

import numpy as np
import torch.nn.functional as F

from PIL import Image
import cv2


PIXELS_PER_METER = 5


class Renderer():
    def __init__(self, map_offset, map_dims, data_generation=True):
        self.args = {'device': 'cuda'}
        if data_generation:
            self.PIXELS_AHEAD_VEHICLE = 0
            self.local_view_dims = (500, 500)
            self.crop_dims = (500, 500)
        else:
            self.PIXELS_AHEAD_VEHICLE = 100 + 10
            self.local_view_dims = (320, 320)
            self.crop_dims = (192, 192)

        self.map_offset = map_offset
        self.map_dims = map_dims
        self.local_view_scale = (
            self.local_view_dims[1] / self.map_dims[1],
            self.local_view_dims[0] / self.map_dims[0]
        )
        self.crop_scale = (
            self.crop_dims[1] / self.map_dims[1],
            self.crop_dims[0] / self.map_dims[0]
        )

    def world_to_pix(self, pos):
        pos_px = (pos-self.map_offset) * PIXELS_PER_METER

        return pos_px

    def world_to_pix_crop_batched(self, query_pos, crop_pos, crop_yaw, offset=(0, 0)):


        crop_yaw = crop_yaw + np.pi / 2
        batch_size = crop_pos.shape[0]


        rotation = torch.stack(
            [torch.cos(crop_yaw), -torch.sin(crop_yaw),
            torch.sin(crop_yaw),  torch.cos(crop_yaw)],
            dim=-1,
        ).view(batch_size, 2, 2)

        crop_pos_px = self.world_to_pix(crop_pos)


        shift = torch.tensor(
            [0., - self.PIXELS_AHEAD_VEHICLE],
            device=self.args['device'],
        )

        query_pos_px_map = self.world_to_pix(query_pos)

        query_pos_px = torch.transpose(rotation, -2, -1).unsqueeze(1) @ \
            (query_pos_px_map - crop_pos_px).unsqueeze(-1)
        query_pos_px = query_pos_px.squeeze(-1) - shift


        pos_px_crop = query_pos_px + torch.tensor([self.crop_dims[1] / 2, self.crop_dims[0] / 2], device=self.args['device'])

        return pos_px_crop

    def world_to_pix_crop(self, query_pos, crop_pos, crop_yaw, offset=(0, 0)):


        crop_yaw = crop_yaw + np.pi / 2


        rotation = torch.tensor(
            [[torch.cos(crop_yaw), -torch.sin(crop_yaw)],
            [torch.sin(crop_yaw),  torch.cos(crop_yaw)]],
            device=self.args['device'],
        )

        crop_pos_px = self.world_to_pix(crop_pos)


        shift = torch.tensor(
            [0., - self.PIXELS_AHEAD_VEHICLE],
            device=self.args['device'],
        )

        query_pos_px_map = self.world_to_pix(query_pos)

        query_pos_px = rotation.T @ (query_pos_px_map - crop_pos_px) - shift


        pos_px_crop = query_pos_px + torch.tensor([self.crop_dims[1] / 2, self.crop_dims[0] / 2], device=self.args['device'])

        return pos_px_crop

    def world_to_rel(self, pos):
        pos_px = self.world_to_pix(pos)
        pos_rel = pos_px / torch.tensor([self.map_dims[1],self.map_dims[0]], device=self.args['device'])

        pos_rel = pos_rel * 2 - 1

        return pos_rel

    def render_agent(self, grid, vehicle, position, orientation):

        orientation = orientation - np.pi/2
        scale_h = torch.tensor([grid.size(2) / vehicle.size(2)], device=self.args['device'])
        scale_w = torch.tensor([grid.size(3) / vehicle.size(3)], device=self.args['device'])


        position = self.world_to_rel(position) * -1


        scale_transform = torch.tensor(
            [[scale_w, 0, 0],
            [0, scale_h, 0],
            [0, 0, 1]],
            device=self.args['device'],
        ).view(1, 3, 3)

        rotation_transform = torch.tensor(
            [[torch.cos(orientation), torch.sin(orientation), 0],
            [-torch.sin(orientation), torch.cos(orientation), 0],
            [0, 0, 1]],
            device=self.args['device'],
        ).view(1, 3, 3)

        translation_transform = torch.tensor(
            [[1, 0, position[0]],
            [0, 1, position[1]],
            [0, 0, 1]],
            device=self.args['device'],
        ).view(1, 3, 3)


        affine_transform = scale_transform @ rotation_transform @ translation_transform

        affine_grid = F.affine_grid(
            affine_transform[:, 0:2, :],
            (1, 1, grid.shape[2], grid.shape[3]),
            align_corners=True,
        )

        vehicle_rendering = F.grid_sample(
            vehicle,
            affine_grid,
            align_corners=True,
        )

        grid[:, 5, ...] += vehicle_rendering.squeeze()

        return grid

    def render_agent_bv(
            self,
            grid,
            grid_pos,
            grid_orientation,
            vehicle,
            position,
            orientation,
            channel=5,
            state=None,
        ):


        orientation = orientation + np.pi / 2


        pos_pix_bv = self.world_to_pix_crop(position, grid_pos, grid_orientation)


        h, w = (grid.size(-2), grid.size(-1))
        pos_rel_bv = pos_pix_bv / torch.tensor([h, w], device=self.args['device'])
        pos_rel_bv = pos_rel_bv * 2 -1
        pos_rel_bv = pos_rel_bv * -1

        scale_h = torch.tensor([grid.size(2) / vehicle.size(2)], device=self.args['device'])
        scale_w = torch.tensor([grid.size(3) / vehicle.size(3)], device=self.args['device'])


        scale_transform = torch.tensor(
            [[scale_w, 0, 0],
            [0, scale_h, 0],
            [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)


        grid_orientation = grid_orientation + np.pi / 2
        rotation_transform = torch.tensor(
            [[torch.cos(orientation - grid_orientation), torch.sin(orientation - grid_orientation), 0],
            [- torch.sin(orientation - grid_orientation), torch.cos(orientation - grid_orientation), 0],
            [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)

        translation_transform = torch.tensor(
            [[1, 0, pos_rel_bv[0]],
            [0, 1, pos_rel_bv[1]],
            [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)


        affine_transform = scale_transform @ rotation_transform @ translation_transform

        affine_grid = F.affine_grid(
            affine_transform[:, 0:2, :],
            (1, 1, grid.shape[2], grid.shape[3]),
            align_corners=True,
        )

        vehicle_rendering = F.grid_sample(
            vehicle,
            affine_grid,
            align_corners=True,
        )

        if state == 'Green':
            channel = 4
        elif state == 'Yellow':
            channel = 3
        elif state == 'Red':
            channel = 2

        grid[:, channel, ...] += vehicle_rendering.squeeze()


    def render_agent_bv_batched(
            self,
            grid,
            grid_pos,
            grid_orientation,
            vehicle,
            position,
            orientation,
            channel=5,
        ):


        orientation = orientation + np.pi / 2
        batch_size = position.shape[0]

        pos_pix_bv = self.world_to_pix_crop_batched(position, grid_pos, grid_orientation)


        h, w = (grid.size(-2), grid.size(-1))
        pos_rel_bv = pos_pix_bv / torch.tensor([h, w], device=self.args['device'])
        pos_rel_bv = pos_rel_bv * 2 -1
        pos_rel_bv = pos_rel_bv * -1

        scale_h = torch.tensor([grid.size(2) / vehicle.size(2)], device=self.args['device'])
        scale_w = torch.tensor([grid.size(3) / vehicle.size(3)], device=self.args['device'])


        scale_transform = torch.tensor(
            [[scale_w, 0, 0],
            [0, scale_h, 0],
            [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3).expand(batch_size, -1, -1)


        grid_orientation = grid_orientation + np.pi / 2
        angle_delta = orientation - grid_orientation
        zeros = torch.zeros_like(angle_delta)
        ones = torch.ones_like(angle_delta)
        rotation_transform = torch.stack(
            [ torch.cos(angle_delta), torch.sin(angle_delta), zeros,
            -torch.sin(angle_delta), torch.cos(angle_delta), zeros,
            zeros,                   zeros,                  ones],
            dim=-1
        ).view(batch_size, 3, 3)

        translation_transform = torch.stack(
            [ones,  zeros, pos_rel_bv[..., 0:1],
             zeros, ones,  pos_rel_bv[..., 1:2],
             zeros, zeros, ones],
            dim=-1,
        ).view(batch_size, 3, 3)


        affine_transform = scale_transform @ rotation_transform @ translation_transform

        affine_grid = F.affine_grid(
            affine_transform[:, 0:2, :],
            (batch_size, 1, grid.shape[2], grid.shape[3]),
            align_corners=True,
        )

        vehicle_rendering = F.grid_sample(
            vehicle,
            affine_grid,
            align_corners=True,
        )

        for i in range(batch_size):
            grid[:, int(channel[i].item()), ...] += vehicle_rendering[i].squeeze()


    def render_agent_stopline_bv_batched(
            self,
            grid,
            grid_pos,
            grid_orientation,
            position,

            channel=5,
        ):


        batch_size = position.shape[0]
        start_pos = position[:,:, :2]
        end_pos = position[:,:, 2:]


        start_pos_pix = self.world_to_pix_crop_batched(start_pos, grid_pos, grid_orientation)
        end_pos_pix = self.world_to_pix_crop_batched(end_pos, grid_pos, grid_orientation)

        for i in range(batch_size):
            ch = int(channel[i].item()) if isinstance(channel, torch.Tensor) else channel
            img = grid[0, ch].detach().cpu().numpy()

            pt1 = tuple(start_pos_pix[i, 0].round().int().tolist())
            pt2 = tuple(end_pos_pix[i, 0].round().int().tolist())

            img = cv2.line(img, pt1, pt2, color=1, thickness=4)

            grid[0, ch] = torch.from_numpy(img).to(grid.device)

    def visualize_light_birdview(self, grid):

        colors = {
            2: (255, 0, 0),
            3: (255, 255, 0),
            4: (0, 255, 0),
        }

        grid = grid.detach().cpu()

        grid_img = np.zeros((grid.shape[2], grid.shape[3], 3), dtype=np.uint8)

        for channel_idx, color in colors.items():
            mask = grid[0, channel_idx, ...] > 0
            grid_img[mask] = color

        pil_img = Image.fromarray(grid_img)

        return pil_img

    def visualize_walker_view(self, grid):

        grid = grid.detach().cpu()

        grid_img = np.zeros((grid.shape[2], grid.shape[3]), dtype=np.uint8)

        mask = grid[0, 6, ...] > 0
        grid_img[mask] = 255

        pil_img = Image.fromarray(grid_img)

        return pil_img

    def visualize_das_view(self, grid):

        grid = grid.detach().cpu()

        grid_img = np.zeros((grid.shape[2], grid.shape[3]), dtype=np.uint8)

        mask = grid[0, 0, ...] > 0
        grid_img[mask] = 255

        pil_img = Image.fromarray(grid_img)

        return pil_img
    def visualize_lane_view(self, grid):

        grid = grid.detach().cpu()

        grid_img = np.zeros((grid.shape[2], grid.shape[3]), dtype=np.uint8)

        mask = grid[0, 1, ...] > 0
        grid_img[mask] = 255

        pil_img = Image.fromarray(grid_img)

        return pil_img

    def visualize_stopline_view(self, grid):

        grid = grid.detach().cpu()

        grid_img = np.zeros((grid.shape[2], grid.shape[3]), dtype=np.uint8)

        mask = grid[0, 7, ...] > 0
        grid_img[mask] = 255

        pil_img = Image.fromarray(grid_img)

        return pil_img

    def visualize_vehicle_view(self, grid):
        grid = grid.detach().cpu()

        grid_img = np.zeros((grid.shape[2], grid.shape[3]), dtype=np.uint8)

        mask = grid[0, 5, ...] > 0
        grid_img[mask] = 255

        pil_img = Image.fromarray(grid_img)

        return pil_img


    def get_local_birdview(self, grid, position, orientation):


        position = self.world_to_rel(position)

        orientation = orientation + np.pi/2

        scale_transform = torch.tensor(
            [[self.crop_scale[1], 0, 0],
            [0, self.crop_scale[0], 0],
            [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)


        rotation_transform = torch.tensor(
            [[torch.cos(orientation), -torch.sin(orientation), 0],
            [torch.sin(orientation), torch.cos(orientation), 0],
            [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)


        shift = torch.tensor([0., - 2 * self.PIXELS_AHEAD_VEHICLE / self.map_dims[0]], device=self.args['device'])
        position = position + rotation_transform[0, 0:2, 0:2] @ shift

        translation_transform = torch.tensor(
            [[1, 0, position[0] / self.crop_scale[0]],
            [0, 1, position[1] / self.crop_scale[1]],
            [0, 0, 1]],
            device=self.args['device']
        ).view(1, 3, 3)


        local_view_transform = scale_transform @ translation_transform @ rotation_transform

        affine_grid = F.affine_grid(
            local_view_transform[:, 0:2, :],
            (1, 1, self.crop_dims[0], self.crop_dims[0]),
            align_corners=True,
        )

        local_view = F.grid_sample(
            grid,
            affine_grid,
            align_corners=True,
        )

        return local_view

    def step(self, actions):


        print(self.ego.state, actions)
        self.ego.set_state(self.ego.motion_model(self.ego.state, actions=actions))
        self.adv.set_state(self.adv.motion_model(self.adv.state))


        self.timestep +=1

    def visualize_grid(self, grid, type='LTS_Reduced'):

        if type=='LTS_Reduced':
            colors = [
                (102, 102, 102),
                (253, 253, 17),


                (0, 0, 142),
                (220, 20, 60),
            ]

        elif type=='Trajectory_planner':
            colors = [
                (102, 102, 102),
                (253, 253, 17),


            ]

        elif type=='LTS_Full':
            colors = [
                (102, 102, 102),
                (253, 253, 17),
                (204, 6, 5),
                (250, 210, 1),
                (39, 232, 51),
                (0, 0, 142),
                (220, 20, 60),
            ]
        elif type=='LTS_FullFuture':
            colors = [
                (102, 102, 102),
                (253, 253, 17),
                (204, 6, 5),
                (250, 210, 1),
                (39, 232, 51),
                (0, 0, 142),
                (220, 20, 60),
                *[(0, 0, 142+(11*i)) for i in range(grid.shape[1]-7)],
            ]
        elif type=='LTS_ReducedFuture':
            colors = [
                (102, 102, 102),
                (253, 253, 17),


                (0, 0, 142),
                (220, 20, 60),
                *[(0, 0, 142+(11*i)) for i in range(grid.shape[1]-7)],
            ]

        grid = grid.detach().cpu()

        grid_img = np.zeros((grid.shape[2:4] + (3,)), dtype=np.uint8)
        grid_img[...] = [0, 0, 0]

        for i in range(len(colors)):
            grid_img[grid[0, i, ...] > 0] = colors[i]

        pil_img = Image.fromarray(grid_img)

        return pil_img

    def bev_to_gray_img(self, grid):

        colors = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ]

        grid = grid.detach().cpu()

        grid_img = np.zeros((grid.shape[2:4]), dtype=np.uint8)

        for i in range(len(colors)):
            grid_img[grid[0, i, ...] > 0] = colors[i]

        pil_img = Image.fromarray(grid_img)

        return pil_img
