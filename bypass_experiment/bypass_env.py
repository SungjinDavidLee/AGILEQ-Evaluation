"""
Bypass reproduction env (G/F/E switching) used for the demo video.

Differences to carla_env/bypass/plan_bypass.py (the mask-only version in the main code):
  * BEV input: every channel is rendered from map / simulator ground truth with synthetic noise (perception_noise.py).
  * Trigger: _is_vehicle_hazard() (simulator) instead of the Algorithm-1 block.
  * F safety checks also use the ground-truth box of the target NPC.
  * Single stationary NPC scenario, shorter episode limits, per-step logging and an optional video overlay.
Settings come from configs/*.json (see README.md).
"""
import os
import sys
import json
import math
import types

import numpy as np
import cv2
import torch
import carla

from agile_local.frenet import quintic, peval
from agile_local.route import ReferenceLine
from agile_local.validation import validate_path, corners, overlap
from agile_local.contracts import Stamp, Ego, BEV, Traffic, Plan, Prediction, ContractError
from agile_local.frames import world_to_flu, flu_to_world, world_vector_to_flu

from carla_env.envs.carla_FixedRoute_env import CarlaRouteEnv
from carla_env.envs.carla_FixedRoute_env import make_vehicle_bev_ego_aligned
from carla_env.wrappers import build_projection_matrix, get_image_point
from carla_env.envs import carla_FixedRoute_env as _fr

OT_CFG = None      # set by run_bypass_demo.py before eval.py constructs the env
LOG_DIR = None


class DemoFinished(Exception):
    """Raised from reset() after n_episodes so eval.py's run_eval() saves CSV/video and returns."""


class BypassRouteEnv(CarlaRouteEnv):
    PURGE_PASSED_WAYPOINTS = False    # keep the waypoint handling of the original demo run
    HAZARD_MIN_LOOKAHEAD_M = 20.0     # keep the hazard range of the original demo run

    def __init__(self, *args, **kwargs):
        cfg = OT_CFG or {}
        self.ot_cfg = cfg
        self.enable_bypass = bool(cfg.get('enable_bypass', False))
        self.enable_single_npc_scenario = bool(cfg.get('enable_single_npc_scenario', False))
        if 'start_carla' in cfg:
            kwargs['start_carla'] = cfg['start_carla']
        tr, lg, sf, fr = cfg.get('trigger', {}), cfg.get('legality', {}), cfg.get('safety', {}), cfg.get('frenet', {})
        self.SPEED_MAX_KMH = tr.get('speed_max_kmh', 5.0)
        self.N_TRIGGER = tr.get('n_trigger_steps', 45)
        self.LEGAL_LOOKAHEAD_M = lg.get('lookahead_m', 40.0)
        self.SAFETY_PROB_THRESHOLD = sf.get('safety_prob_threshold', 0.2)
        self.CLEARANCE_M = sf.get('clearance_m', 0.2)
        self.EGO_CLEAR_M = sf.get('ego_mask_clear_margin_m', 0.3)
        self.L_CHANGE = fr.get('lane_change_length_m', 10.0)
        self.HOLD_FRAC = fr.get('hold_offset_frac', 1.0)     # F path lateral target = HOLD_FRAC * lane width
        self.RETURN_LEN = fr.get('return_length_m', None)    # length of the return leg (default: same as L_eff)
        # quintic shift of W over L peaks at kappa ~ 5.77 W / L^2; a_lat = v^2 kappa <= 3.5 -> L >= ~2.4 v (W = 3.5 m)
        self.LAT_T = fr.get('lane_change_time_s', 2.6)
        self.RETURN_MARGIN_M = fr.get('return_margin_m', 7.0)
        self.TAIL_M = fr.get('tail_m', 10.0)
        self.F_SPEED = fr.get('target_speed_kmh', 15.0) / 3.6
        self.MAX_POINTS_AW = fr.get('max_points_aw', 8)
        self.SAMPLE_M = fr.get('sample_spacing_m', 1.0)
        sc = cfg.get('scenario', {})
        self.NPC_AHEAD_M = sc.get('npc_ahead_m', 45.0)
        self.NPC_BP = sc.get('npc_blueprint', 'vehicle.lincoln.mkz_2020')
        self.NPC_SPEED_KMH = sc.get('npc_speed_kmh', 0.0)
        self.N_EPISODES = cfg.get('n_episodes', None)
        cm, em = cfg.get('commit', {}), cfg.get('emergency', {})
        self.N_ABORT = cm.get('abort_consecutive_steps', 5)
        self.MIN_HOLD_S = cm.get('min_hold_s', 1.5)
        self.NEAR_M = cm.get('near_horizon_m', 10.0)
        self.COR_T = cm.get('ego_corridor_time_s', 1.5)
        self.COR_MIN_M = cm.get('ego_corridor_min_m', 2.0)
        self.OFF_PATH_M = cm.get('off_path_m', 2.5)
        self.F_STALL_S = cm.get('f_stall_s', 5.0)
        self.F_SPEED_CAP_KMH = cm.get('f_speed_cap_kmh', None)   # F only: brake above this, steer untouched
        self.F_CAP_BRAKE = cm.get('f_speed_cap_brake', 0.3)
        # F path sweep: hits within this arc length ahead of the ego (the car being passed right now) are not hazards
        self.D_IGNORE_M = cm.get('path_ignore_m', 2.0)
        self.E_STOP_KMH = em.get('stop_speed_kmh', 0.5)
        self.E_EXIT_S = em.get('exit_hold_s', 1.0)
        self.E_CLEAR_M = em.get('clear_m', 6.0)
        dl = cfg.get('demo_limits', {})
        self.DEMO_STOP_S = dl.get('stop_timeout_s', None)      # replaces the stock 80 s safe timeout (shorter)
        self.DEMO_MAX_S = dl.get('max_episode_s', None)        # replaces the stock 500 s scenario timeout

        self._ot_init_done = False
        self._ot_resets = 0
        self._ot_reset_episode_state()
        self._ot_rows, self._ot_episodes = [], []
        self._ot_frame = {}
        super().__init__(*args, **kwargs)
        self._install_gt_veh_head()
        self._ot_init_done = True

    # ------------------------------------------------------------------ GT vehicle channel
    def _gt_vehicles(self):
        tr = self.vehicle.get_transform()
        data_rad = {'x': float(tr.location.x), 'y': float(tr.location.y), 'yaw': math.radians(tr.rotation.yaw)}
        ego_loc = tr.location
        vehicles, ids = [], []
        for vid in self.npc_vehicle_list:
            v = self.world.get_actor(vid)
            if v and v.is_alive and v.id != self.vehicle.id and v.get_location().distance(ego_loc) < self.detection_radius:
                bb = v.bounding_box
                c = v.get_transform().transform(bb.location)
                vehicles.append([c.x, c.y, math.radians(v.get_transform().rotation.yaw), 2.0 * bb.extent.x, 2.0 * bb.extent.y])
                ids.append(vid)
        return data_rad, vehicles, ids

    def _gt_vehicle_mask(self):
        """200x200 bool, env BEV layout (row0 = +20 m forward, col = right), same calls as carla_env_train.get_bev()."""
        data_rad, vehicles, _ = self._gt_vehicles()
        veh500 = make_vehicle_bev_ego_aligned(data_rad, vehicles, out_size=500)
        return veh500[150:350, 150:350] > 0

    def _gt_static_masks(self):
        """GT road (DAS) / lane / stop-line masks, same calls as carla_env_train.get_bev(): global map images cropped
        with crop_yaw_nn at 0.2 m/px around the ego (500x500, centre 150:350 = the 200x200 model window)."""
        tr = self.vehicle.get_transform()
        x, y, z, yaw = tr.location.x, tr.location.y, tr.location.z, tr.rotation.yaw
        cx, cy = _fr.world_to_pixel(x, y, self.OFFSET_X, self.OFFSET_Y, 5.0)
        town = self.world.get_map().name.split('/')[-1].lower()
        high = z > self.limit
        das = self.das_global if not high else (self.das_04 if town.startswith('town04') else self.das_05 if town.startswith('town05') else self.das_global)
        lane = self.lane_global if not high else (self.lane_04 if town.startswith('town04') else self.lane_05 if town.startswith('town05') else self.lane_global)
        out = {}
        for k, img in (('road', das), ('lane', lane), ('stop', self.stop_global)):
            c = _fr.crop_yaw_nn(img, (cx, cy), yaw, fill=0) if img is not None else None
            out[k] = (c[150:350, 150:350] > 0) if c is not None else np.zeros((200, 200), bool)
        return out

    def _gt_static_logits(self):
        from perception_noise import StaticChannelNoise, LaneLikeNoise
        cfgc = (self.ot_cfg or {}).get('gt_all_channels', {})
        if getattr(self, '_static_n', None) is None or self._static_ep != self._ot_resets:
            rng = np.random.default_rng(int(cfgc.get('seed', 0)) + self._ot_resets)
            self._static_n = {}
            for k in ('road', 'lane', 'stop'):
                c = dict(cfgc.get(k, {}))
                kind = c.pop('type', 'static')
                self._static_n[k] = LaneLikeNoise(rng, **c) if kind == 'lane_like' else StaticChannelNoise(rng, **c)
            self._static_ep = self._ot_resets
        m = self._gt_static_masks()
        self.last_gt_road = m['road']
        tr = self.vehicle.get_transform()
        ego = (tr.location.x, tr.location.y, math.radians(tr.rotation.yaw))
        return {k: (n.step(m[k], ego) if isinstance(n, LaneLikeNoise) else n.step(m[k])) for k, n in self._static_n.items()}

    def _gt_vehicle_logits(self):
        """+-20 logits of the crisp GT rectangle, or the perception-like rendering when gt_veh_shape_noise is set."""
        sn = (self.ot_cfg or {}).get('gt_veh_shape_noise')
        if not sn:
            return np.where(self._gt_vehicle_mask(), 20.0, -20.0).astype(np.float32)
        from perception_noise import PerceptionLikeRenderer
        if getattr(self, '_shape_r', None) is None or self._shape_ep != self._ot_resets:
            self._shape_r = PerceptionLikeRenderer(sn.get('preset', 'D'), rng=np.random.default_rng(int(sn.get('seed', 0)) + self._ot_resets))
            self._shape_ep = self._ot_resets
        data_rad, vehicles, ids = self._gt_vehicles()
        return self._shape_r.render_logits(data_rad, vehicles, ids=ids)

    def _install_gt_veh_head(self):
        env, model = self, self.bev_model

        class _GTVehHead:
            """Delegates to BEVFormer and swaps only the veh head for GT (+-20 logits)."""
            def __call__(self, *a, **k):
                road, lane_broken, lane_solid, veh, ped, stop = model(*a, **k)
                env.last_pred_veh_logits = veh
                to_t = lambda a, like: torch.from_numpy(a).to(like.device, like.dtype).view(
                    *([1] * (like.dim() - 2)), *a.shape).expand_as(like)
                lg = to_t(env._gt_vehicle_logits(), veh)
                env.last_gt_veh = lg > 0
                if (env.ot_cfg or {}).get('gt_all_channels', {}).get('enabled'):
                    env.last_pred_road_logits = road
                    st = env._gt_static_logits()
                    # lane_broken carries all GT markings (the env feeds max(lane_broken, lane_solid) to the policy)
                    return (to_t(st['road'], road), to_t(st['lane'], lane_broken), torch.full_like(lane_solid, -20.0),
                            lg, torch.full_like(ped, -20.0), to_t(st['stop'], stop))
                return road, lane_broken, lane_solid, lg, ped, stop

            def __getattr__(self, name):
                return getattr(model, name)

        self.bev_model = _GTVehHead()

    # ------------------------------------------------------------------ episode bookkeeping
    def _ot_reset_episode_state(self):
        self.path_mode = 'G'
        self.trigger_count = 0
        self.f_path_world = None     # [N,2] world xy of the current F path (0.5 m spacing, from ego)
        self.f_target_npc = None     # actor id of the NPC being bypassed
        self.f_landing = None        # (x, y, yaw) world pose of the route point where F rejoins the route
        self.f_enter_step = None
        self.f_hazard_count = 0      # consecutive hazardous F steps (hysteresis)
        self.f_stall_count = 0
        self.e_ok_count = 0          # consecutive E steps satisfying the E -> G exit condition
        self._ot_twp = None          # T_wp actually built for the state this step (raw m + normalised)
        self.demo_stop_steps = 0
        self._ot_ep = {'trigger_events': 0, 'f_entries': 0, 'f_returns': 0, 'f_aborts': [],
                       'e_entries': 0, 'e_to_g': 0, 'e_to_f': 0, 'f_steps': 0, 'e_steps': 0,
                       'gate_block_steps': {'legality': 0, 'safety': 0, 'both': 0},
                       'gate_block_reasons': {}, 'min_dist_to_npc': float('inf'), 'max_lateral_offset_m': 0.0,
                       'max_left_offset_m': 0.0, 'reached_left_lane': False,
                       'd_ignore_steps': 0, 'd_ignore_only_steps': 0, 'demo_timeout': ''}

    def reset(self, is_training=False):
        if self._ot_init_done:
            self._ot_close_episode()
            self._ot_resets += 1
            if self.N_EPISODES is not None and self._ot_resets > self.N_EPISODES:
                raise DemoFinished(f'{self.N_EPISODES} demo episodes finished')
        self._ot_reset_episode_state()
        return super().reset(is_training)

    def _ot_close_episode(self):
        if self._ot_resets == 0:
            return  # env-construction reset, not an eval episode
        ep = dict(self._ot_ep, episode=self._ot_resets, steps=int(self.step_count),
                  collision_vehicle=bool(self.collision_true), collision_static=bool(self.collision_static),
                  success=bool(self.success_state), safe_timeout=bool(self.safe_timeout),
                  route_completion=float(self.distance_traveled_on_route / max(self.route_length, 1e-6)))
        self._ot_episodes.append(ep)
        self._ot_flush()

    def _ot_flush(self):
        if not LOG_DIR:
            return
        import pandas as pd
        os.makedirs(LOG_DIR, exist_ok=True)
        pd.DataFrame(self._ot_rows).to_csv(os.path.join(LOG_DIR, 'steps.csv'), index=False)
        with open(os.path.join(LOG_DIR, 'episodes.json'), 'w') as f:
            json.dump(self._ot_episodes, f, indent=2, default=float)

    def close(self):
        try:
            if self._ot_init_done and self._ot_resets <= (self.N_EPISODES or 10 ** 9):
                self._ot_close_episode()
        finally:
            super().close()

    # ------------------------------------------------------------------ scenario (single stationary NPC)
    def walkers_position_save(self):
        if not self.enable_single_npc_scenario:
            return super().walkers_position_save()

    def _spawn_walkers_v2(self):
        if not self.enable_single_npc_scenario:
            return super()._spawn_walkers_v2()

    def _spawn_vehicles(self):
        if not self.enable_single_npc_scenario:
            return super()._spawn_vehicles()
        # NPC in the ego lane, NPC_AHEAD_M along the route from the start
        acc, k = 0.0, 0
        for k in range(1, len(self.route_waypoints)):
            acc += self.route_waypoints[k - 1][0].transform.location.distance(self.route_waypoints[k][0].transform.location)
            if acc >= self.NPC_AHEAD_M:
                break
        wp = self.route_waypoints[k][0]
        bp = self.world.get_blueprint_library().find(self.NPC_BP)
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name', 'scenario_npc')
        npc = None
        for dz in (0.3, 0.8, 1.5):
            t = carla.Transform(wp.transform.location + carla.Location(z=dz), wp.transform.rotation)
            npc = self.world.try_spawn_actor(bp, t)
            if npc is not None:
                break
        if npc is None:
            print('[bypass_demo] WARNING: scenario NPC could not be spawned')
            return
        if self.NPC_SPEED_KMH > 0:
            npc.set_autopilot(True, self.traffic_manager.get_port())
            self.traffic_manager.auto_lane_change(npc, False)
            self.traffic_manager.set_desired_speed(npc, self.NPC_SPEED_KMH)
        else:
            npc.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        self.npc_vehicle_list.append(npc.id)
        self._tick_safe()
        print(f'[bypass_demo] scenario NPC {npc.type_id} id={npc.id} at {acc:.1f} m ahead '
              f'({wp.transform.location.x:.1f}, {wp.transform.location.y:.1f}), speed {self.NPC_SPEED_KMH} km/h')

    def _scenario_npc(self):
        for vid in self.npc_vehicle_list:
            a = self.world.get_actor(vid)
            if a is not None and a.is_alive:
                return a
        return None

    # ------------------------------------------------------------------ danger mask copies (parallel)
    def _danger_mask_with_threshold(self, veh_threshold):
        """Verbatim copy of the danger-mask block in CarlaRouteEnv.step(), with the veh threshold as a parameter
        (0.5 reproduces the stock mask). Uses the raw heads logged by step() for this very frame."""
        out = self.last_bev_outputs
        veh, ped, stop = out['veh'], out['ped'], out['stop']
        light_idx = self.last_tlc_outputs['light_idx'] if getattr(self, 'last_tlc_outputs', None) else 0

        pred_veh_mask = (torch.sigmoid(veh) > veh_threshold).float().squeeze().cpu().numpy()
        pred_ped_mask = (torch.sigmoid(ped) > 0.5).float().squeeze().cpu().numpy()

        kernel_veh = np.ones((3, 3), np.uint8)
        veh_inflated = cv2.dilate(pred_veh_mask, kernel_veh, iterations=1)
        kernel_ped = np.ones((5, 5), np.uint8)
        ped_inflated = cv2.dilate(pred_ped_mask, kernel_ped, iterations=1)

        danger_mask = np.zeros((200, 200), dtype=np.uint8)
        danger_mask[veh_inflated > 0] = 255
        danger_mask[ped_inflated > 0] = 255

        if light_idx != 0:
            stop_np = (torch.sigmoid(stop) >= 0.5).float().squeeze().cpu().numpy().astype(np.uint8)
            kernel_stop = np.ones((3, 3), np.uint8)
            stop_thick = cv2.dilate(stop_np, kernel_stop, iterations=2)
            danger_mask[stop_thick > 0] = 255

        kernel = np.ones((3, 3), np.uint8)
        danger_mask = cv2.dilate(danger_mask, kernel, iterations=1)
        return danger_mask

    # ------------------------------------------------------------------ hooks called by step()
    def get_way_v2(self):
        g_ways = super().get_way_v2()
        if not self.enable_bypass:
            return g_ways
        try:
            self._bypass_update(g_ways)
        except Exception as e:  # never let the demo logic crash the stock loop
            self._ot_frame['error'] = f'{type(e).__name__}: {e}'
            self._set_mode_g(f'error:{type(e).__name__}')
        if self.path_mode == 'F' and self.f_path_world is not None:
            return self._f_ways_renderer()
        return g_ways

    def get_truncated_waypoints_state(self, max_points=8, ignore_obstacles=True):
        f_mode = self.enable_bypass and self.path_mode == 'F' and self.f_path_world is not None
        if ignore_obstacles:
            current_max_len = max_points
        else:
            current_max_len = min(max_points, self.valid_waypoint_len) if hasattr(self, 'valid_waypoint_len') else max_points
        ego_transform = self.vehicle.get_transform()
        ego_loc, ego_fwd, ego_right = ego_transform.location, ego_transform.get_forward_vector(), ego_transform.get_right_vector()
        if f_mode:
            pts = [(p[0], p[1], ego_loc.z) for p in self._f_samples(max_points)]
        else:
            pts = [(w[0].transform.location.x, w[0].transform.location.y, w[0].transform.location.z)
                   for w in list(self.local_planner._waypoints_queue)[:max_points]]
        # Same local transform / normalisation as the parent (raw metres kept for logging)
        MAX_WAYPOINT_DIST = 30.0
        raw, local_waypoints = [], []
        for i in range(min(current_max_len, len(pts))):
            dx, dy, dz = pts[i][0] - ego_loc.x, pts[i][1] - ego_loc.y, pts[i][2] - ego_loc.z
            local_x = dx * ego_fwd.x + dy * ego_fwd.y + (0.0 if f_mode else dz * ego_fwd.z)
            local_y = dx * ego_right.x + dy * ego_right.y + (0.0 if f_mode else dz * ego_right.z)
            raw.extend([local_x, local_y])
            local_waypoints.extend([np.clip(local_x / MAX_WAYPOINT_DIST, -1.0, 1.0),
                                    np.clip(local_y / MAX_WAYPOINT_DIST, -1.0, 1.0)])
        target_len = max_points * 2
        if len(local_waypoints) < target_len:
            local_waypoints.extend([0.0] * (target_len - len(local_waypoints)))
        if not f_mode:
            parent = super().get_truncated_waypoints_state(max_points, ignore_obstacles)
            if self.enable_bypass and not np.allclose(parent, local_waypoints, atol=1e-6):
                self._ot_frame['error'] = 'twp_copy_mismatch'
            local_waypoints = parent
        self._ot_twp = {'src': 'F' if f_mode else 'G', 'n_valid': len(raw) // 2, 'raw': raw, 'norm': list(local_waypoints)}
        return local_waypoints

    # ------------------------------------------------------------------ video-only overlay (never part of the state)
    def _vis_global_route_overlay(self):
        """Draw the global route waypoints as a dashed line on COPIES of the two images the stock step() composes into
        the recorded frame (front_cam_pv viewer image and bev_image_vis). The policy state (bev_image, T_wp) is untouched."""
        vc = (self.ot_cfg or {}).get('vis_global_route')
        if not vc or getattr(self, 'route_waypoints', None) is None:
            return
        color = tuple(vc.get('color_rgb', [255, 0, 0]))             # frames are stored RGB-ordered in the video
        step_wp = int(vc.get('dot_every_wp', 2))                    # waypoints are 1 m apart: one dot every 2 m
        i0 = max(0, self.current_waypoint_index - 5)
        wps = [w.transform.location for w, _ in self.route_waypoints[i0:i0 + int(vc.get('ahead_wp', 45))]]
        if len(wps) < 2:
            return
        # front viewer camera
        cam = getattr(self, 'front_cam_pv_rgb', None)
        if cam is not None and cam.img is not None:
            img = np.ascontiguousarray(cam.img.copy())
            w2c = np.array(cam.sensor.get_transform().get_inverse_matrix())
            K = build_projection_matrix(int(cam.sensor.attributes['image_size_x']), int(cam.sensor.attributes['image_size_y']),
                                        float(cam.sensor.attributes['fov']))
            ego = self.vehicle.get_location()
            r_near, r_far = float(vc.get('cam_radius_near', 12)), float(vc.get('cam_radius_far', 4))
            for k, loc in enumerate(wps):
                if k % step_wp:
                    continue
                dist = loc.distance(ego)
                if not 2.5 < dist < 45:
                    continue
                p_ = get_image_point(carla.Location(loc.x, loc.y, loc.z + 0.05), K, w2c)
                if p_ is None or not (0 <= p_[0] < img.shape[1] and 0 <= p_[1] < img.shape[0]):
                    continue
                r = int(round(np.clip(r_near * 6.0 / dist, r_far, r_near)))   # larger when close (perspective)
                cv2.circle(img, (int(p_[0]), int(p_[1])), r, color, -1, cv2.LINE_AA)
            cam.img = img
        # BEV visualisation (200 x 200, same renderer transform as the stock route channel)
        if getattr(self, 'bev_image_vis', None) is not None:
            tr = self.vehicle.get_transform()
            ways = torch.tensor([[l.x, l.y] for l in wps], device='cuda', dtype=torch.float32).unsqueeze(1)
            pos = torch.tensor([[tr.location.x, tr.location.y]] * len(wps), device='cuda', dtype=torch.float32).unsqueeze(1)
            yaw = torch.tensor([[tr.rotation.yaw / 180 * np.pi]] * len(wps), device='cuda', dtype=torch.float32).unsqueeze(1)
            px = (self.renderer.world_to_pix_crop_batched(ways, pos, yaw).squeeze(1).cpu().numpy() - 150.0)
            bev = np.ascontiguousarray(self.bev_image_vis.copy())
            rb = int(vc.get('bev_radius', 2))
            for k in range(0, len(px), step_wp):
                if 0 <= px[k][0] < bev.shape[1] and 0 <= px[k][1] < bev.shape[0]:
                    cv2.circle(bev, (int(px[k][0]), int(px[k][1])), rb, color, -1)
            self.bev_image_vis = bev

    def step(self, action):
        if self.enable_bypass or self.bypass.ENABLED:
            try:
                self._vis_global_route_overlay()
            except Exception as ex:                                 # visualisation must never break the run
                print(f'[vis] overlay skipped: {type(ex).__name__}: {ex}', flush=True)
        self._ot_frame = {}
        self._ot_twp = None
        core_before = self.bypass.mode
        if self.enable_bypass and self.path_mode == 'E' and action is not None:
            self._ot_frame.update(act_in_steer=float(action[0]), act_in_dw=float(action[1]), act_override=1)
            action = np.array([0.0, -1.0], dtype=np.float32)          # EMERGENCY: steer 0, full brake
        elif action is not None:
            self._ot_frame.update(act_in_steer=float(action[0]), act_in_dw=float(action[1]), act_override=0)
            if (self.enable_bypass and self.path_mode == 'F' and self.F_SPEED_CAP_KMH is not None
                    and self.speed > self.F_SPEED_CAP_KMH):
                # F speed cap: light brake (dw = -F_CAP_BRAKE -> throttle 0, brake F_CAP_BRAKE), steer untouched
                action = np.array([float(action[0]), -self.F_CAP_BRAKE], dtype=np.float32)
                self._ot_frame['act_override'] = 2
        mode_before = self.path_mode
        # [steer fix] stock order: smooth_steering(policy) clips to past_steering +-0.1, then local_planner.run_step()
        # overwrites past_steering with the route-PID steer. In F/E only, re-anchor the rate limit on the steer that was
        # actually applied last step, so the route PID no longer pulls the F/E steer back. G keeps the stock behaviour.
        vc = self.local_planner._vehicle_controller
        self._ot_frame['steer_anchor_pid'] = round(float(vc.past_steering), 4)
        if self.enable_bypass and self.path_mode in ('F', 'E') and action is not None and getattr(self, 'control', None) is not None:
            vc.past_steering = float(self.control.steer)
            self._ot_frame['steer_anchor_restored'] = 1
        else:
            self._ot_frame['steer_anchor_restored'] = 0
        self._ot_frame['steer_anchor_used'] = round(float(vc.past_steering), 4)
        out = super().step(action)
        out = self._demo_limits(out, action)
        if self.enable_bypass:
            self._ot_frame['mode_at_action'] = mode_before
            self._ot_log_step(action)
        elif self.bypass.ENABLED:
            self._log_main_module(action, core_before)
        return out

    def _log_main_module(self, action, before):
        """Step log when the main module (carla_env/bypass) drives the bypass instead of this env."""
        o, after = self.bypass, self.bypass.mode
        event = f'{before}->{after}' if after != before else np.nan
        key = {('G', 'F'): 'f_entries', ('E', 'F'): 'e_to_f', ('F', 'G'): 'f_to_g', ('F', 'E'): 'e_entries', ('E', 'G'): 'e_to_g'}
        if (before, after) in key:
            self._ot_ep[key[(before, after)]] = self._ot_ep.get(key[(before, after)], 0) + 1
        if after == 'F':
            self._ot_ep['f_steps'] += 1
        loc = self.vehicle.get_location()
        self._ot_rows.append({'episode': self._ot_resets, 'step': int(self.step_count), 'action_none': int(action is None),
                              'path_mode': after, 'mode_event': event, 'ego_speed': round(float(self.speed), 3),
                              'trigger_count': o.trigger_count, 'trigger_active': int(o.trigger_count >= o.TRIGGER_STEPS),
                              'g_blocked': int(o.g_blocked), 'd_route': o.d_route, 'ego_x': round(loc.x, 3), 'ego_y': round(loc.y, 3),
                              'collision': int(self.collision_true), 'success': int(self.success_state)})
        if self.vis_info is not None:
            self.vis_info['MODE'] = f"{after} trig {min(o.trigger_count, o.TRIGGER_STEPS)}/{o.TRIGGER_STEPS}"

    def _demo_limits(self, out, action):
        """Shorter test-time episode limits (the stock 80 s stopped / 500 s scenario limits stay as a fallback)."""
        if action is None or out[2]:
            return out
        self.demo_stop_steps = self.demo_stop_steps + 1 if self.speed < 1.0 else 0
        why = None
        if self.DEMO_STOP_S and self.demo_stop_steps >= self.DEMO_STOP_S * self.fps:
            why, self.safe_timeout = f'demo_stop_timeout_{self.DEMO_STOP_S:g}s', True
        elif self.DEMO_MAX_S and self.step_count >= self.DEMO_MAX_S * self.fps:
            why = f'demo_max_episode_{self.DEMO_MAX_S:g}s'
        if why is None:
            return out
        self.terminal_state = True
        self._ot_ep['demo_timeout'] = why
        self._ot_frame['mode_event'] = f'END:{why}'
        info = out[3]
        if isinstance(info, dict):
            info = dict(info, demo_timeout=why)
        return out[0], out[1], True, info

    # ------------------------------------------------------------------ F path helpers
    def _f_project(self):
        """Ego projection on the frozen F path -> (arc length s, cross-track distance, arc table)."""
        p = self.f_path_world
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        loc = self.vehicle.get_location()
        e = np.array([loc.x, loc.y])
        a, b = p[:-1], p[1:]
        ab = b - a
        t = np.clip(((e - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0.0, 1.0)
        q = a + ab * t[:, None]
        d = np.linalg.norm(q - e, axis=1)
        i = int(np.argmin(d))
        return float(s[i] + t[i] * np.linalg.norm(ab[i])), float(d[i]), s

    def _f_samples(self, n):
        """Frozen F path points every SAMPLE_M metres ahead of the ego's projection on it (world xy)."""
        p = self.f_path_world
        s_proj, _, s = self._f_project()
        q = s_proj + np.arange(1, n + 1) * self.SAMPLE_M
        q = q[q <= s[-1]]
        if len(q) == 0:
            return np.zeros((0, 2))
        return np.stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])], axis=1)

    def _f_ways_renderer(self):
        pts = self._f_samples(self.MAX_POINTS_AW)
        if len(pts) == 0:
            return torch.tensor([[[250.0, 250.0]]], device='cuda', dtype=torch.float32)
        tr = self.vehicle.get_transform()
        ego_pos = [tr.location.x, tr.location.y]
        ego_yaw = [tr.rotation.yaw / 180 * np.pi]
        ways_torch = torch.tensor(pts, device='cuda', dtype=torch.float32).unsqueeze(1)
        pos_b = torch.tensor([ego_pos] * len(pts), device='cuda', dtype=torch.float32).unsqueeze(1)
        yaw_b = torch.tensor([ego_yaw] * len(pts), device='cuda', dtype=torch.float32).unsqueeze(1)
        ways_torch = self.renderer.world_to_pix_crop_batched(ways_torch, pos_b, yaw_b)
        ways_torch = torch.cat([torch.tensor([[[250.0, 250.0]]], device=ways_torch.device), ways_torch], dim=0)
        self.ways_bev = ways_torch
        return ways_torch

    def _set_mode_g(self, reason=None):
        if self.path_mode == 'F':
            if reason == 'returned':
                self._ot_ep['f_returns'] += 1
            else:
                self._ot_ep['f_aborts'].append({'step': int(self.step_count), 'reason': reason, 'to': 'G'})
        elif self.path_mode == 'E':
            self._ot_ep['e_to_g'] += 1
        self._ot_frame['mode_event'] = f'{self.path_mode}->G:{reason}'
        self.path_mode = 'G'
        self.f_path_world = None
        self.f_target_npc = None
        self.f_landing = None

    def _set_mode_e(self, reason):
        """F abort -> EMERGENCY (brake to standstill). The target NPC id is kept for the E exit check."""
        self._ot_ep['f_aborts'].append({'step': int(self.step_count), 'reason': reason, 'to': 'E',
                                        'hold_s': round((self.step_count - self.f_enter_step) / self.fps, 2),
                                        'speed_kmh': round(float(self.speed), 2)})
        self._ot_ep['e_entries'] += 1
        self._ot_frame['mode_event'] = f'F->E:{reason}'
        self.path_mode = 'E'
        self.f_path_world = None
        self.f_landing = None
        self.trigger_count = 0       # a new F attempt needs a fresh n_trigger_steps window
        self.e_ok_count = 0

    # ------------------------------------------------------------------ gates
    def _legal_length(self):
        """Distance from the current route waypoint to the first waypoint where a left lane change stops being legal."""
        idx = self.current_waypoint_index
        acc = 0.0
        prev = None
        for i in range(idx, len(self.route_waypoints)):
            wp = self.route_waypoints[i][0]
            if prev is not None:
                acc += prev.transform.location.distance(wp.transform.location)
            if acc > self.LEGAL_LOOKAHEAD_M:
                return self.LEGAL_LOOKAHEAD_M, 'lookahead_reached'
            left = wp.get_left_lane()
            ok = (wp.left_lane_marking.lane_change in (carla.LaneChange.Left, carla.LaneChange.Both)
                  and left is not None and wp.lane_id * left.lane_id > 0
                  and left.lane_type == carla.LaneType.Driving)
            if not ok:
                return acc, f'illegal_at_{acc:.1f}m'
            prev = wp
        return acc, 'route_end'

    def _route_frame(self):
        """Route reference line in the current ego FLU frame -> (ref, M, s0, d0, W); d > 0 = left of the route."""
        tr = self.vehicle.get_transform()
        M = np.array(tr.get_matrix())
        i0 = max(0, self.current_waypoint_index - 5)
        route = [self.route_waypoints[i][0].transform.location for i in range(i0, min(len(self.route_waypoints), i0 + 90))]
        route_xyz = np.array([[p.x, p.y, p.z] for p in route])
        ref = ReferenceLine(world_to_flu(route_xyz, M)[:, :2])
        s0, d0 = ref.project([0.0, 0.0])
        W = float(self.route_waypoints[self.current_waypoint_index][0].lane_width)
        return ref, M, float(s0), float(d0), W

    @staticmethod
    def _npc_flu(npc, M):
        """NPC bbox centre / yaw / size in the ego FLU frame."""
        bb = npc.bounding_box
        c = npc.get_transform().transform(bb.location)
        yaw = math.radians(npc.get_transform().rotation.yaw)
        two = world_to_flu(np.array([[c.x, c.y, c.z], [c.x + math.cos(yaw), c.y + math.sin(yaw), c.z]]), M)[:, :2]
        return two[0], float(math.atan2(*(two[1] - two[0])[::-1])), 2 * bb.extent.x, 2 * bb.extent.y

    def _build_candidate(self, npc):
        """Frenet out-and-back path from the current ego state. Returns dict with FLU/world path and landmarks."""
        ref, M, s0, d0, W = self._route_frame()
        W = W * self.HOLD_FRAC                                             # lateral target of the F path
        _, yaw_r, k0 = ref.at(s0)

        bb = npc.bounding_box
        c = npc.get_transform().transform(bb.location)
        npc_flu = world_to_flu(np.array([[c.x, c.y, c.z]]), M)[0, :2]
        s_npc, d_npc = ref.project(npc_flu)
        npc_len = 2 * bb.extent.x
        ego_len, ego_wid = 2 * self.vehicle.bounding_box.extent.x, 2 * self.vehicle.bounding_box.extent.y

        dp0 = float(np.clip(math.tan(-yaw_r) * (1 - k0 * d0), -0.5, 0.5))   # ego heading relative to reference
        v_now = self.speed / 3.6
        L_eff = max(self.L_CHANGE, self.LAT_T * v_now)                       # speed-adaptive lane-change length
        s2 = s_npc + npc_len / 2 + ego_len / 2 + self.RETURN_MARGIN_M       # back to the route from here
        legs = []
        if s0 < s2:
            L1 = max(3.0, L_eff * min(1.0, abs(W - d0) / W))
            s1 = min(s0 + L1, s2)
            legs.append((s0, s1, quintic(d0, dp0, 0.0, W, 0.0, 0.0, s1 - s0)))
            legs.append((s1, s2, None))                                        # hold at d = W
            L_ret = max(L_eff, self.RETURN_LEN or 0.0)                      # gentler return once the obstacle is passed
            s3 = s2 + L_ret
            legs.append((s2, s3, quintic(W, 0.0, 0.0, 0.0, 0.0, 0.0, L_ret)))
        else:
            L2 = max(4.0, L_eff * min(1.0, abs(d0) / W))
            s3 = s0 + L2
            legs.append((s0, s3, quintic(d0, dp0, 0.0, 0.0, 0.0, 0.0, L2)))
        s_end = min(s3 + self.TAIL_M, ref.s[-1] - 0.1)
        if s3 > ref.s[-1] - 0.1:
            raise ContractError('reference too short for the manoeuvre')

        ss = np.arange(s0, s_end, 0.5)
        dd = np.zeros_like(ss)
        for a, b, coeff in legs:
            m = (ss >= a) & (ss <= b)
            dd[m] = W if coeff is None else peval(coeff, ss[m] - a)
        center, heading, _ = ref.at(ss)
        normal = np.stack([-np.sin(heading), np.cos(heading)], axis=1)
        xy_flu = center + normal * dd[:, None]
        xy_world = flu_to_world(xy_flu, M)[:, :2]
        yaw_npc = world_to_flu(np.array([[c.x, c.y, c.z],
                                          [c.x + math.cos(math.radians(npc.get_transform().rotation.yaw)),
                                           c.y + math.sin(math.radians(npc.get_transform().rotation.yaw)), c.z]]), M)[:, :2]
        npc_yaw_flu = float(math.atan2(*(yaw_npc[1] - yaw_npc[0])[::-1]))
        return {'npc_flu': npc_flu, 'npc_yaw_flu': npc_yaw_flu, 'npc_len': npc_len, 'npc_wid': 2 * bb.extent.y,
                'L_eff': L_eff, 'ref': ref, 'M': M, 's0': s0, 'd0': d0, 's_npc': s_npc, 'd_npc': d_npc, 's2': s2, 's3': s3, 'W': W,
                'xy_flu': xy_flu, 'xy_world': xy_world, 'ego_len': ego_len, 'ego_wid': ego_wid,
                'required_len': s3 - s0}

    def _occupancy_flu(self, danger_mask, ego_len, ego_wid):
        """env BEV (row0 = +20 m forward, col = right) -> agile_local ego_flu grid (row -> +x forward, col -> +y left)."""
        occ = danger_mask > 0
        # the model sometimes paints the ego itself as a vehicle; the ego footprint cannot be an obstacle
        hl = int(math.ceil((ego_len / 2 + self.EGO_CLEAR_M) / 0.2))
        hw = int(math.ceil((ego_wid / 2 + self.EGO_CLEAR_M) / 0.2))
        occ[100 - hl:100 + hl + 1, 100 - hw:100 + hw + 1] = False
        return occ[::-1, ::-1].copy()

    def _safety_check(self, cand, danger_low):
        """validate_path on the in-BEV part of the candidate + landing footprint check."""
        stamp = Stamp(f'ep{self.episode_idx}', int(self.step_count), float(self.step_count) / self.fps)
        M = cand['M']
        v = self.vehicle.get_velocity()
        vel_flu = world_vector_to_flu([v.x, v.y, v.z], M)[:2]
        bbl = self.vehicle.bounding_box.location
        ego = Ego(stamp, M, vel_flu, np.zeros(2), length_m=cand['ego_len'], width_m=cand['ego_wid'],
                  bbox_center_flu=(bbl.x, -bbl.y))
        occ = self._occupancy_flu(danger_low, cand['ego_len'], cand['ego_wid'])
        bev = BEV(stamp, np.ones_like(occ), occ, np.zeros_like(occ), -20.0, -20.0, 0.2, 0.2)
        traffic = Traffic(stamp, 'NONE', False, 1.0)
        obs = types.SimpleNamespace(ego=ego, bev=bev, traffic=traffic, predictions=[])  # target NPC added below

        # ---- landing footprint (route centre at s3) ----
        (lxy,), lyaw, _ = cand['ref'].at([cand['s3']])
        m = self.CLEARANCE_M
        lx = np.linspace(-cand['ego_len'] / 2 - m, cand['ego_len'] / 2 + m, 25)
        ly = np.linspace(-cand['ego_wid'] / 2 - m, cand['ego_wid'] / 2 + m, 13)
        fp = np.stack(np.meshgrid(lx, ly, indexing='ij'), -1).reshape(-1, 2)
        c, s = math.cos(lyaw[0]), math.sin(lyaw[0])
        cloud = fp @ np.array([[c, s], [-s, c]]) + lxy
        rc = np.floor((cloud - np.array([-20.0, -20.0])) / 0.2).astype(int)
        inside = (rc >= 0).all(1) & (rc < 200).all(1)
        if inside.all():
            landing = 'occupied' if occ[rc[:, 0], rc[:, 1]].any() else 'clear'
        else:
            landing = 'not_visible'
        if landing == 'occupied':
            return False, 'EMERGENCY_landing_occupied', landing

        # ---- swept footprint along the visible part of the path ----
        xy = cand['xy_flu']
        x_lim = 19.8 - (cand['ego_len'] / 2 + m) - abs(bbl.x)
        y_lim = 19.8 - (cand['ego_wid'] / 2 + m)
        vis = (xy[:, 0] <= x_lim) & (np.abs(xy[:, 1]) <= y_lim)
        k = int(np.argmin(vis)) if not vis.all() else len(xy)
        xy = xy[:k]
        if len(xy) < 3:
            return False, 'path_not_visible', landing
        sig = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        heading = np.unwrap(np.arctan2(np.gradient(xy[:, 1], sig), np.gradient(xy[:, 0], sig)))
        kappa = np.gradient(heading, sig)
        v0 = float(np.linalg.norm(vel_flu))
        vt = self.F_SPEED
        T = 2 * sig[-1] / max(v0 + vt, 0.1)
        sc = quintic(0.0, v0, 0.0, sig[-1], vt, 0.0, T)
        t = np.linspace(0, T, int(np.ceil(T / 0.1)) + 1)
        st, sd, sdd = peval(sc, t), peval(sc, t, 1), peval(sc, t, 2)
        st = np.clip(st, 0, sig[-1])
        pxy = np.stack([np.interp(st, sig, xy[:, 0]), np.interp(st, sig, xy[:, 1])], axis=1)
        pxy[0] = 0.0  # plan starts at the ego origin (projection residual is < 1 mm)
        plan = Plan(stamp, t, pxy, np.interp(st, sig, heading), np.maximum(sd, 0.0), sdd,
                    np.interp(st, sig, kappa), 'FEASIBLE', 'candidate')
        # the NPC being bypassed (the path is built around its position) as a static prediction, so the swept
        # footprint must clear it even where the AGP mask misses it -> validate_path 'predicted_collision'
        obs.predictions = [Prediction(str(self.f_target_npc), np.array([0.0, t[-1]]),
                                      np.array([cand['npc_flu'], cand['npc_flu']]),
                                      np.array([cand['npc_yaw_flu'], cand['npc_yaw_flu']]),
                                      cand['npc_len'], cand['npc_wid'], margin_m=0.25)]
        ok, why = validate_path(plan, obs, margin_m=self.CLEARANCE_M)
        self._ot_diag_first_hit(xy, heading, cand, occ)
        self._ot_debug_image(cand, occ, xy, ok, why)
        return ok, why, landing

    def _ot_diag_first_hit(self, xy, heading, cand, occ):
        """Diagnostic only (the gate decision is validate_path's): first path point whose footprint touches occ."""
        m = self.CLEARANCE_M
        lx = np.linspace(-cand['ego_len'] / 2 - m, cand['ego_len'] / 2 + m, 25)
        ly = np.linspace(-cand['ego_wid'] / 2 - m, cand['ego_wid'] / 2 + m, 13)
        fp = np.stack(np.meshgrid(lx, ly, indexing='ij'), -1).reshape(-1, 2)
        for i in range(len(xy)):
            c, s_ = math.cos(heading[i]), math.sin(heading[i])
            cloud = fp @ np.array([[c, s_], [-s_, c]]) + xy[i]
            rc = np.floor((cloud + 20.0) / 0.2).astype(int)
            ok = (rc >= 0).all(1) & (rc < 200).all(1)
            hit = occ[rc[ok, 0], rc[ok, 1]]
            if hit.any():
                h = rc[ok][hit]
                self._ot_frame.update(first_hit_path_fwd=round(float(xy[i, 0]), 2), first_hit_path_left=round(float(xy[i, 1]), 2),
                                      first_hit_occ_fwd=round(float(-20 + (h[:, 0].mean() + .5) * .2), 2),
                                      first_hit_occ_left=round(float(-20 + (h[:, 1].mean() + .5) * .2), 2))
                return

    def _ot_debug_image(self, cand, occ, xy, ok, why):
        every = int(self.ot_cfg.get('debug_image_every', 0) or 0)
        if not LOG_DIR or every <= 0 or self.step_count % every:
            return
        d = os.path.join(LOG_DIR, 'debug')
        os.makedirs(d, exist_ok=True)
        img = np.zeros((200, 200, 3), np.uint8)
        occ_env = occ[::-1, ::-1]                      # back to env layout (row0 = forward, col = right)
        img[occ_env] = (0, 200, 255)
        to_px = lambda p: (int(round(100 - p[1] / 0.2)), int(round(100 - p[0] / 0.2)))  # (col, row)
        full = cand['xy_flu']
        for a, b in zip(full[:-1], full[1:]):
            cv2.line(img, to_px(a), to_px(b), (120, 120, 120), 1)
        for a, b in zip(xy[:-1], xy[1:]):
            cv2.line(img, to_px(a), to_px(b), (0, 255, 0) if ok else (0, 0, 255), 1)
        cv2.rectangle(img, (94, 88), (106, 112), (255, 255, 255), 1)
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f'ep{self._ot_resets} s{self.step_count} {self.path_mode} safety:{why}', (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(d, f'ep{self._ot_resets}_s{self.step_count:05d}.png'), img)

    def _bypass_update(self, g_ways):
        f = self._ot_frame
        # ---------- parallel Algorithm-1 lengths on the G path (for logging only) ----------
        check = g_ways.clone() - 150.0
        danger05 = self._danger_mask_with_threshold(0.5)
        danger_low = self._danger_mask_with_threshold(self.SAFETY_PROB_THRESHOLD)
        f['valid_waypoint_len_G05'] = int(self._check_path_collision_and_cut(check, danger05))
        f['valid_waypoint_len_safety'] = int(self._check_path_collision_and_cut(check, danger_low))
        # lateral position w.r.t. the route, every step (V1 only had it on gated steps)
        try:
            _, _, _, d_route, W = self._route_frame()
            f.update(d_route=round(d_route, 3), lane_width=W)
            self._ot_ep['max_left_offset_m'] = max(self._ot_ep['max_left_offset_m'], d_route)
            if d_route > W / 2:
                self._ot_ep['reached_left_lane'] = True
        except ContractError:
            pass

        # ---------- 1. trigger ----------
        npc_vehicle, npc_distance, _ = self._is_vehicle_hazard()
        # TLCFormer prediction of this frame, same rule as step()'s final_vector:
        # dist_idx == 21 -> no light, light_idx == 0 -> Green, otherwise Red (Red/Yellow merged)
        tlc = getattr(self, 'last_tlc_outputs', None)
        if tlc is None or tlc['dist_idx'] == 21:
            light = 'NONE'
        else:
            light = 'GREEN' if tlc['light_idx'] == 0 else 'RED'
        walker, _ = self._get_closest_hazardous_walker_v2()
        cond = (npc_vehicle is not None and light != 'RED'
                and walker is None and self.speed < self.SPEED_MAX_KMH)
        prev_active = self.trigger_count >= self.N_TRIGGER
        self.trigger_count = self.trigger_count + 1 if cond else 0
        trigger_active = self.trigger_count >= self.N_TRIGGER
        if trigger_active and not prev_active:
            self._ot_ep['trigger_events'] += 1
        f.update(trigger_cond=int(cond), trigger_count=self.trigger_count, trigger_active=int(trigger_active),
                 hazard_npc=int(npc_vehicle is not None), pred_light=light, walker=int(walker is not None))

        if self.path_mode == 'F':
            self._ot_ep['f_steps'] += 1
            self._f_update(danger_low)
            return
        if self.path_mode == 'E':
            self._ot_ep['e_steps'] += 1
            if trigger_active and self._try_enter_f(npc_vehicle, danger_low):
                return
            self._e_update(danger_low, npc_vehicle)
            return
        if trigger_active:
            self._try_enter_f(npc_vehicle, danger_low)

    # ---------- G/E -> F: full gates on a fresh candidate, then freeze it ----------
    def _try_enter_f(self, npc, danger_low):
        f = self._ot_frame
        if npc is None:
            return False
        self.f_target_npc = npc.id          # used by the static-prediction check in _safety_check
        try:
            cand = self._build_candidate(npc)
        except ContractError as e:
            f.update(gate_safety_pass=0, safety_reason=f'plan:{e}')
            return False
        legal_len, legal_why = self._legal_length()
        leg_ok = legal_len >= min(cand['required_len'], self.LEGAL_LOOKAHEAD_M) and cand['required_len'] <= self.LEGAL_LOOKAHEAD_M
        saf_ok, saf_why, landing = self._safety_check(cand, danger_low)
        f.update(gate_legality_pass=int(leg_ok), legal_len_m=round(legal_len, 2), legal_reason=legal_why,
                 required_len_m=round(cand['required_len'], 2), gate_safety_pass=int(saf_ok), safety_reason=saf_why,
                 landing=landing, s0=round(cand['s0'], 2), d0=round(cand['d0'], 3), s_npc=round(cand['s_npc'], 2),
                 s2=round(cand['s2'], 2), s3=round(cand['s3'], 2), lane_width=cand['W'], L_eff=round(cand['L_eff'], 2))
        if not (leg_ok and saf_ok):
            key = 'both' if not (leg_ok or saf_ok) else ('legality' if not leg_ok else 'safety')
            self._ot_ep['gate_block_steps'][key] += 1
            r = saf_why if not saf_ok else legal_why
            self._ot_ep['gate_block_reasons'][r] = self._ot_ep['gate_block_reasons'].get(r, 0) + 1
            if self.path_mode == 'G':
                self.f_target_npc = None
            return False
        # ---------- switch to F with a frozen path ----------
        (lxy,), lyaw, _ = cand['ref'].at([cand['s3']])
        lw = flu_to_world(np.array([[lxy[0], lxy[1], 0.0], [lxy[0] + math.cos(lyaw[0]), lxy[1] + math.sin(lyaw[0]), 0.0]]),
                          cand['M'])[:, :2]
        self.f_landing = (lw[0, 0], lw[0, 1], math.atan2(lw[1, 1] - lw[0, 1], lw[1, 0] - lw[0, 0]))
        if self.path_mode == 'E':
            self._ot_ep['e_to_f'] += 1
        self._ot_ep['f_entries'] += 1
        f['mode_event'] = f'{self.path_mode}->F'
        self.path_mode = 'F'
        self.f_path_world = cand['xy_world']
        self.f_enter_step = int(self.step_count)
        self.f_hazard_count = 0
        self.f_stall_count = 0
        return True

    # ---------- per-step F: follow the frozen path, hysteresis abort ----------
    def _f_update(self, danger_low):
        f = self._ot_frame
        npc = self.world.get_actor(self.f_target_npc) if self.f_target_npc is not None else None
        if npc is None or not npc.is_alive:
            self._set_mode_g('target_npc_gone')
            return
        tr = self.vehicle.get_transform()
        M = np.array(tr.get_matrix())
        hold_s = (self.step_count - self.f_enter_step) / self.fps
        s_proj, cte, s_tab = self._f_project()
        remaining = s_tab[-1] - s_proj
        f.update(f_hold_s=round(hold_s, 3), f_proj_s=round(s_proj, 2), f_cte=round(cte, 3), f_remaining_m=round(remaining, 2))

        # ---- normal end: past the landing point and back on the route ----
        lx, ly, lyaw = self.f_landing
        along = (tr.location.x - lx) * math.cos(lyaw) + (tr.location.y - ly) * math.sin(lyaw)
        f['f_along_landing'] = round(along, 2)
        if along >= -1.0 and abs(f.get('d_route', 99.0)) < 0.5:
            self._set_mode_g('returned')
            return
        if remaining < 2.0:
            self._set_mode_g('path_end')
            return

        # ---- hazards on the remaining frozen path + short straight corridor ahead of the ego ----
        ego_len, ego_wid = 2 * self.vehicle.bounding_box.extent.x, 2 * self.vehicle.bounding_box.extent.y
        occ = self._occupancy_flu(danger_low, ego_len, ego_wid)
        npc_box = self._npc_flu(npc, M)
        p = self.f_path_world
        q = np.arange(s_proj, s_tab[-1], 0.25)
        pw = np.stack([np.interp(q, s_tab, p[:, 0]), np.interp(q, s_tab, p[:, 1]), np.full(len(q), tr.location.z)], 1)
        xy = world_to_flu(pw, M)[:, :2]
        xy = np.vstack([[0.0, 0.0], xy])                      # sweep starts at the ego itself
        path_why, path_at = self._sweep(xy, occ, npc_box, ego_len, ego_wid, ignore_m=self.D_IGNORE_M)
        ign = self._last_sweep_ignored
        v = self.speed / 3.6
        cor_len = max(self.COR_MIN_M, self.COR_T * v)
        cx = np.arange(0.25, cor_len + 1e-6, 0.25)
        cxy = np.stack([cx, np.zeros_like(cx)], 1)
        cor_why, cor_at = self._sweep(cxy, occ, npc_box, ego_len, ego_wid)

        reason = None
        if cor_why is not None:
            reason = f'corridor_{cor_why}'
        elif path_why is not None and (path_at <= self.NEAR_M or hold_s >= self.MIN_HOLD_S):
            reason = f'{path_why}' + ('_near' if path_at <= self.NEAR_M else '_far')
        elif cte > self.OFF_PATH_M:
            reason = 'off_path'
        self.f_hazard_count = self.f_hazard_count + 1 if reason else 0
        if ign is not None:
            self._ot_ep['d_ignore_steps'] += 1
            if reason is None:
                self._ot_ep['d_ignore_only_steps'] += 1     # this step would have counted as a hazard without D_ignore
            f.update(f_path_ignored=ign[0], f_path_ignored_at_m=round(ign[1], 2))
        f.update(f_path_hazard=path_why or '', f_path_hazard_at_m=round(path_at, 2) if path_why else np.nan,
                 f_corridor_hazard=cor_why or '', f_corridor_len_m=round(cor_len, 2),
                 f_hazard_reason=reason or '', f_hazard_count=self.f_hazard_count)
        if reason and self.f_hazard_count >= self.N_ABORT:
            self._set_mode_e(reason)
            return
        # ---- stall: stopped on the F path for too long ----
        self.f_stall_count = self.f_stall_count + 1 if (self.speed < self.E_STOP_KMH and hold_s >= self.MIN_HOLD_S) else 0
        if self.f_stall_count >= self.F_STALL_S * self.fps:
            self._set_mode_e('f_stall')
            return
        self._ot_debug_image({'xy_flu': xy}, occ, xy[:int(self.NEAR_M / 0.25) + 1], reason is None, reason or 'clear')

    def _sweep(self, xy, occ, npc_box, ego_len, ego_wid, ignore_m=0.0):
        """Ego footprint along an FLU polyline (heading from the polyline) -> (reason, arc m) of the first hit, or
        (None, 0). Stops silently where the footprint leaves the BEV (not visible = not checked).
        Hits at arc <= ignore_m are skipped (first one kept in self._last_sweep_ignored) and the sweep continues."""
        self._last_sweep_ignored = None
        if len(xy) < 2:
            return None, 0.0
        m = self.CLEARANCE_M
        bbl = self.vehicle.bounding_box.location
        off = np.array([bbl.x, -bbl.y])
        lx = np.linspace(-ego_len / 2 - m, ego_len / 2 + m, 25)
        ly = np.linspace(-ego_wid / 2 - m, ego_wid / 2 + m, 13)
        fp = np.stack(np.meshgrid(lx, ly, indexing='ij'), -1).reshape(-1, 2) + off
        sig = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        dxy = np.gradient(xy, axis=0)
        heading = np.arctan2(dxy[:, 1], dxy[:, 0])
        npc_c = corners(npc_box[0], npc_box[1], npc_box[2], npc_box[3], margin=0.25)
        for i in range(len(xy)):
            c, s_ = math.cos(heading[i]), math.sin(heading[i])
            cloud = fp @ np.array([[c, s_], [-s_, c]]) + xy[i]
            rc = np.floor((cloud + 20.0) / 0.2).astype(int)
            if not ((rc >= 0).all() and (rc < 200).all()):
                return None, 0.0
            hit = None
            if occ[rc[:, 0], rc[:, 1]].any():
                hit = 'occupancy'
            elif overlap(corners(xy[i], heading[i], ego_len, ego_wid, off, m), npc_c):
                hit = 'predicted_collision'
            if hit is None:
                continue
            if sig[i] <= ignore_m:
                if self._last_sweep_ignored is None:
                    self._last_sweep_ignored = (hit, float(sig[i]))
                continue
            return hit, float(sig[i])
        return None, 0.0

    # ---------- EMERGENCY: braking is applied in step(); here only the E -> G exit ----------
    def _e_update(self, danger_low, npc_vehicle):
        f = self._ot_frame
        stopped = self.speed < self.E_STOP_KMH
        near_clear = False
        try:
            ref, M, s0, d0, W = self._route_frame()
            ss = np.arange(s0 + 0.5, s0 + self.E_CLEAR_M + 1e-6, 0.25)
            center, _, _ = ref.at(ss)
            xy = np.vstack([[0.0, 0.0], center])               # from the ego back onto the route (the G path)
            ego_len, ego_wid = 2 * self.vehicle.bounding_box.extent.x, 2 * self.vehicle.bounding_box.extent.y
            occ = self._occupancy_flu(danger_low, ego_len, ego_wid)
            npc = self.world.get_actor(self.f_target_npc) if self.f_target_npc is not None else self._scenario_npc()
            npc_box = self._npc_flu(npc, M) if (npc is not None and npc.is_alive) else (np.array([1e3, 1e3]), 0.0, 0.1, 0.1)
            why, at = self._sweep(xy, occ, npc_box, ego_len, ego_wid)
            near_clear = why is None
            f.update(e_g_hazard=why or '', e_g_hazard_at_m=round(at, 2) if why else np.nan)
        except ContractError as e:
            f.update(e_g_hazard=f'plan:{e}')
        ok = stopped and near_clear and npc_vehicle is None
        self.e_ok_count = self.e_ok_count + 1 if ok else 0
        f.update(e_stopped=int(stopped), e_near_clear=int(near_clear), e_ok_count=self.e_ok_count)
        if self.e_ok_count >= self.E_EXIT_S * self.fps:
            self._set_mode_g('e_clear')

    # ------------------------------------------------------------------ logging
    def _ot_log_step(self, action):
        f = self._ot_frame
        npc = self._scenario_npc()
        loc = self.vehicle.get_location()
        dist = loc.distance(npc.get_location()) if npc is not None else float('nan')
        if np.isfinite(dist):
            self._ot_ep['min_dist_to_npc'] = min(self._ot_ep['min_dist_to_npc'], dist)
        if 'd0' in f:
            self._ot_ep['max_lateral_offset_m'] = max(self._ot_ep['max_lateral_offset_m'], abs(float(f['d0'])))
        row = {'episode': self._ot_resets, 'step': int(self.step_count), 'action_none': int(action is None),
               'path_mode': self.path_mode, 'ego_speed': round(float(self.speed), 3),
               'dist_to_npc': round(dist, 3) if np.isfinite(dist) else np.nan,
               'valid_waypoint_len': int(getattr(self, 'valid_waypoint_len', -1)),
               'ego_x': round(loc.x, 3), 'ego_y': round(loc.y, 3),
               'collision': int(self.collision_true), 'success': int(self.success_state)}
        pv, gv = getattr(self, 'last_pred_veh_logits', None), getattr(self, 'last_gt_veh', None)
        if pv is not None and gv is not None:
            pm = (torch.sigmoid(pv) > 0.5).squeeze().cpu().numpy()
            gm = gv.squeeze().cpu().numpy()
            inter, union = int((pm & gm).sum()), int((pm | gm).sum())
            row.update(gt_veh_px=int(gm.sum()), pred_veh_px=int(pm.sum()), gt_pred_iou=round(inter / union, 3) if union else np.nan)
        pr, gr = getattr(self, 'last_pred_road_logits', None), getattr(self, 'last_gt_road', None)
        if pr is not None and gr is not None:
            pm = (torch.sigmoid(pr) > 0.5).squeeze().cpu().numpy()
            u = int((pm | gr).sum())
            row['gt_pred_road_iou'] = round(int((pm & gr).sum()) / u, 3) if u else np.nan
        ctrl = getattr(self, 'control', None)
        if ctrl is not None and action is not None:
            row.update(ctrl_steer=round(float(ctrl.steer), 4), ctrl_throttle=round(float(ctrl.throttle), 4),
                       ctrl_brake=round(float(ctrl.brake), 4))
        # T_wp actually put into the CrossQ state this step: raw ego-local metres (x fwd, y right) + normalised (/30, clip)
        tw = self._ot_twp
        row['twp_src'] = tw['src'] if tw else ''
        row['twp_n_valid'] = tw['n_valid'] if tw else np.nan
        state_twp = getattr(self, 'predicted_waypoints_norm', None)
        for i in range(7):
            for j, ax in enumerate('xy'):
                row[f'twp{i + 1}_{ax}_m'] = round(float(tw['raw'][2 * i + j]), 3) if tw and 2 * i + j < len(tw['raw']) else np.nan
                row[f'twp{i + 1}_{ax}_norm'] = round(float(tw['norm'][2 * i + j]), 5) if tw else np.nan
        if tw and state_twp is not None and not np.allclose(state_twp, tw['norm']):
            f['error'] = 'twp_state_mismatch'
        for k in ['trigger_cond', 'trigger_count', 'trigger_active', 'gate_legality_pass', 'gate_safety_pass',
                  'valid_waypoint_len_G05', 'valid_waypoint_len_safety', 'hazard_npc', 'pred_light', 'walker',
                  'legal_len_m', 'legal_reason', 'required_len_m', 'safety_reason', 'landing', 's0', 'd0',
                  's_npc', 's2', 's3', 'lane_width', 'L_eff', 'first_hit_path_fwd', 'first_hit_path_left',
                  'first_hit_occ_fwd', 'first_hit_occ_left', 'd_route', 'mode_at_action', 'mode_event',
                  'act_in_steer', 'act_in_dw', 'act_override',
                  'f_hold_s', 'f_proj_s', 'f_cte', 'f_remaining_m', 'f_along_landing', 'f_path_hazard',
                  'f_path_hazard_at_m', 'f_corridor_hazard', 'f_corridor_len_m', 'f_hazard_reason', 'f_hazard_count',
                  'e_stopped', 'e_near_clear', 'e_ok_count', 'e_g_hazard', 'e_g_hazard_at_m',
                  'steer_anchor_pid', 'steer_anchor_used', 'steer_anchor_restored',
                  'f_path_ignored', 'f_path_ignored_at_m']:
            row[k] = f.get(k, np.nan)
        self._ot_rows.append(row)
        if len(self._ot_rows) % 200 == 0:
            self._ot_flush()
        # two extra lines on the recorded frame (keeps the text panel under the 720 px camera height)
        if self.vis_info is not None:
            gl = {1: 'ok', 0: 'NO'}.get(f.get('gate_legality_pass'), '-')
            gs = {1: 'ok', 0: 'NO'}.get(f.get('gate_safety_pass'), '-')
            self.vis_info['MODE'] = f"{self.path_mode} trig {min(self.trigger_count, self.N_TRIGGER)}/{self.N_TRIGGER}"
            self.vis_info['GATE'] = f"L:{gl} S:{gs} {str(f.get('safety_reason', ''))[:10]}"
