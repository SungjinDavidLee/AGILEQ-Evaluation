import math

import numpy as np
import torch
import carla

from .geometry import (PlanError, ReferenceLine, quintic, peval,
                       world_to_flu, flu_to_world, world_vector_to_flu)
from .planners import BypassContext, plan_bypass, rejoin_goal_from_occupancy, resample, headings

BEV_RES = 0.2


class PlanBypass:
    """
    Bypass mode of the High-Level Waypoint Guidance (Algorithm 1).
      G: route waypoints truncated by the obstacle mask (stock behaviour)
      F: frozen bypass path from plan_bypass(METHOD, ...) (shift to the neighbour lane, pass, return); its first
         waypoints are truncated by the same obstacle mask, so the policy stops if the bypass path becomes blocked
    Every decision uses the BEV masks and the TLC prediction only.
    OVERRIDE_ACTIONS (off by default) caps the speed in F, for policies that were not trained for bypassing.
    """

    ENABLED = False
    OVERRIDE_ACTIONS = False

    TRIGGER_STEPS = 45
    TRIGGER_SPEED_KMH = 5.0
    ROUTE_POINTS = 8
    TRIGGER_POINTS = None        # None: Psi_T as given to update() (Algorithm 1); an int: that many route queue points
    LEGAL_LOOKAHEAD_M = 50.0

    OBSTACLE_LEN_M = 7.0
    TAIL_M = 10.0
    SAMPLE_M = 1.0               # waypoint spacing on the bypass path (same as the global route)

    CLEARANCE_M = 0.2
    REAR_CHECK_M = 15.0
    SIDE = 'left'              # 'left' or 'right'
    METHOD = 'frenet'          # frenet, lattice, astar, rrtstar, qp, nmpc, mppi (see planners.py)
    PLAN_BUDGET_S = 2.0
    EGO_CLEAR_M = 0.3
    MAX_CURVATURE = 0.35
    MAX_LAT_ACCEL = 3.5
    MAX_LON_ACCEL = 3.5

    V_BP_KMH = 6.0               # V_bp: speed above the blocking vehicle during a bypass (reward); also the path-planning speed
    F_SPEED_CAP_KMH = 6.0        # brake above this in F when OVERRIDE_ACTIONS (demo reproduction)
    F_CAP_BRAKE = 0.3

    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.mode = 'G'
        self.trigger_count = 0
        self.path = None
        self.landing = None
        self.enter_step = 0
        self.g_blocked = False
        self.block_xy = None
        self.d_route = None
        self.plan = None
        self.last_exit = ''
        self.veh = None
        self.grid = 200
        self.half = self.grid * BEV_RES / 2

    # ------------------------------------------------------------ hooks used by CarlaRouteEnv.step()
    def filter_action(self, action):
        # keep the steering rate limit anchored on the last applied steer, not on the route PID
        env = self.env
        if action is None or self.mode == 'G':
            return action
        if getattr(env, 'control', None) is not None:
            env.local_planner._vehicle_controller.past_steering = float(env.control.steer)
        if self.OVERRIDE_ACTIONS and env.speed > self.F_SPEED_CAP_KMH:
            return np.array([float(action[0]), -self.F_CAP_BRAKE], dtype=np.float32)
        return action

    def update(self, g_ways, danger_mask, veh_mask=None):
        """danger_mask: obstacle mask M_O of Algorithm 1; veh_mask: vehicle channel A_V (rear-zone check)."""
        if not self.ENABLED:
            return g_ways
        try:
            self._update(g_ways, danger_mask, veh_mask)
        except PlanError:
            self._to_g('plan_error')
        if self.mode == 'F':
            return self._path_ways()
        return g_ways

    def waypoints_state(self, max_points, n_valid, max_dist=30.0):
        tr = self.env.vehicle.get_transform()
        loc, fwd, right = tr.location, tr.get_forward_vector(), tr.get_right_vector()
        out = []
        pts = self._samples(max_points)[:n_valid]
        for x, y in pts:
            dx, dy = x - loc.x, y - loc.y
            out.extend([np.clip((dx * fwd.x + dy * fwd.y) / max_dist, -1.0, 1.0),
                        np.clip((dx * right.x + dy * right.y) / max_dist, -1.0, 1.0)])
        return out + [0.0] * (2 * max_points - len(out))

    def tracking_errors(self, distance, angle):
        """Lane-keeping errors for the reward: w.r.t. the bypass path while it is active, else the route values."""
        if self.mode != 'F' or self.path is None:
            return distance, angle
        s_proj, cte, s = self._project()
        i = min(int(np.searchsorted(s, s_proj, side='right')) - 1, len(s) - 2)
        tangent = self.path[i + 1] - self.path[i]
        v = self.env.vehicle.get_velocity()
        vel = np.array([v.x, v.y])
        if np.linalg.norm(tangent) < 1e-6 or np.linalg.norm(vel) < 1e-6:
            return cte, 0.0
        return cte, math.atan2(tangent[0] * vel[1] - tangent[1] * vel[0], float(np.dot(tangent, vel)))

    # ------------------------------------------------------------ per step
    def _update(self, g_ways, danger_mask, veh_mask):
        env = self.env
        self.grid = danger_mask.shape[0]
        self.half = self.grid * BEV_RES / 2
        occ = self._occupancy_flu(danger_mask)
        self.veh = occ if veh_mask is None else self._occupancy_flu(veh_mask)
        self._algorithm1(g_ways, danger_mask)
        try:
            self.d_route = self._route_frame()[3]
        except PlanError:
            self.d_route = None

        fv = env.final_vector
        red = fv is not None and (fv[0] == 1 or fv[1] == 1)
        cond = self.g_blocked and not red and env.speed < self.TRIGGER_SPEED_KMH
        self.trigger_count = self.trigger_count + 1 if cond else 0
        triggered = self.trigger_count >= self.TRIGGER_STEPS

        if self.mode == 'F':
            self._update_f()
        elif triggered:
            self._try_enter_f(occ)

    def _algorithm1(self, g_ways, danger_mask):
        """BYPASSNEEDED of Algorithm 1: k < |Psi_T|, i.e. Psi_T is cut by the obstacle mask; also where it is cut."""
        env = self.env
        q = env.local_planner._waypoints_queue
        self.g_blocked, self.block_xy = False, None
        if self.TRIGGER_POINTS is None:
            ways = g_ways                    # [ego, q[0], ..., q[n-1]] from get_way_v2()
            n = len(ways) - 1
        else:
            n = min(self.TRIGGER_POINTS, len(q))
            if n >= 1:
                pts = [(q[i][0].transform.location.x, q[i][0].transform.location.y) for i in range(n)]
                ways = env.renderer.world_to_pix_crop_batched(*self._batched(pts))
                ways = torch.cat([torch.tensor([[[250.0, 250.0]]], device=ways.device), ways], dim=0)
        if n < 1 or n > len(q):
            return
        vlen = env._check_path_collision_and_cut(ways.clone() - (250.0 - self.grid / 2), danger_mask)
        if vlen < len(ways):
            self.g_blocked = True
            loc = q[max(vlen, 1) - 1][0].transform.location
            self.block_xy = (loc.x, loc.y)

    def _try_enter_f(self, occ):
        if self.block_xy is None:
            return False
        try:
            cand = self._build_candidate(self.block_xy, occ)
        except PlanError:
            return False
        legal = self._legal_length()
        if cand['length'] > self.LEGAL_LOOKAHEAD_M or legal < cand['length']:
            return False
        if not self._entry_safe(cand, occ):
            return False
        lxy, lyaw = cand['landing']
        lw = flu_to_world(np.array([[lxy[0], lxy[1]], [lxy[0] + math.cos(lyaw), lxy[1] + math.sin(lyaw)]]),
                          cand['M'])[:, :2]
        self.landing = (lw[0, 0], lw[0, 1], math.atan2(lw[1, 1] - lw[0, 1], lw[1, 0] - lw[0, 0]))
        self.path = cand['xy_world']
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.plan = dict(cand['plan'], return_arc=float(seg[:cand['i_return']].sum()))
        self.mode = 'F'
        self.enter_step = self.env.step_count
        return True

    def _update_f(self):
        """The bypass ends when the ego reaches the end of the bypass path (back on the route past the landing point)."""
        tr = self.env.vehicle.get_transform()
        s_proj, _, s_tab = self._project()
        lx, ly, lyaw = self.landing
        along = (tr.location.x - lx) * math.cos(lyaw) + (tr.location.y - ly) * math.sin(lyaw)
        if along >= -1.0 and self.d_route is not None and abs(self.d_route) < 0.5:
            self._to_g('returned')
        elif s_tab[-1] - s_proj < 2.0:
            self._to_g('path_end')

    def _to_g(self, reason=''):
        self.last_exit = reason
        self.mode = 'G'
        self.path = None
        self.landing = None

    # ------------------------------------------------------------ path
    def _route_frame(self):
        """Route reference line in the ego FLU frame -> (ref, M, s0, d0, lane width); d > 0 is left of the route."""
        env = self.env
        M = np.array(env.vehicle.get_transform().get_matrix())
        i0 = max(0, env.current_waypoint_index - 5)
        locs = [env.route_waypoints[i][0].transform.location
                for i in range(i0, min(len(env.route_waypoints), i0 + 90))]
        ref = ReferenceLine(world_to_flu(np.array([[p.x, p.y, p.z] for p in locs]), M)[:, :2])
        s0, d0 = ref.project([0.0, 0.0])
        W = float(env.route_waypoints[env.current_waypoint_index][0].lane_width)
        return ref, M, s0, d0, W

    def _build_candidate(self, block_xy, occ):
        """Bypass path from the ego to the rejoin point: candidate = plan_bypass(METHOD, context, rejoin_goal)."""
        env = self.env
        ref, M, s0, d0, W = self._route_frame()
        blk = world_to_flu(np.array([[block_xy[0], block_xy[1], env.vehicle.get_location().z]]), M)[0, :2]
        ego_len, ego_wid, _ = self._ego_box()
        ctx = BypassContext(ref_xy=ref.xy, occ=occ, x_min=-self.half, y_min=-self.half, res=BEV_RES, lane_width=W,
                            side=1 if self.SIDE == 'left' else -1, v0=env.speed / 3.6, ego_len=ego_len, ego_wid=ego_wid,
                            clearance=self.CLEARANCE_M, v_target=self.V_BP_KMH / 3.6, a_lat_max=self.MAX_LAT_ACCEL,
                            kappa_max=self.MAX_CURVATURE, budget_s=self.PLAN_BUDGET_S, seed=int(env.step_count))
        s_blk = float(ctx.project(blk[None])[0][0])
        goal = rejoin_goal_from_occupancy(ctx, s_blk, assumed_len=self.OBSTACLE_LEN_M)
        xy, info = plan_bypass(self.METHOD, ctx, goal)
        if xy is None:
            raise PlanError(f"{self.METHOD}: no bypass path")
        xy = resample(xy, 0.5)
        s_land, _ = ctx.project(xy[-1:])
        _, yaw_land, _ = ctx.ref_at(s_land[0])
        tail = ctx.to_xy(np.arange(s_land[0] + 0.5, s_land[0] + self.TAIL_M, 0.5), 0.0)
        xy_flu = np.vstack([xy, tail])
        cs0 = ctx.ego_sd()[0]
        return {'M': M, 'length': float(s_land[0] - cs0), 'W': W, 'd0': d0, 'i_return': len(xy) - 1,
                'landing': (xy[-1], float(yaw_land)),
                'plan': {'method': self.METHOD, 's0': cs0, 's_block': s_blk, 's_clear': goal['s_clear'],
                         's_land': float(s_land[0]), 'ego_len': ego_len, 'lane_width': W, 'plan_ms': info['time_s'] * 1000},
                'xy_flu': xy_flu, 'xy_world': flu_to_world(xy_flu, M)[:, :2]}

    def _legal_length(self):
        """Route length ahead over which a lane change to SIDE is legal."""
        env = self.env
        acc, prev = 0.0, None
        for i in range(env.current_waypoint_index, len(env.route_waypoints)):
            wp = env.route_waypoints[i][0]
            if prev is not None:
                acc += prev.transform.location.distance(wp.transform.location)
            if acc > self.LEGAL_LOOKAHEAD_M:
                return self.LEGAL_LOOKAHEAD_M
            if self.SIDE == 'left':
                nb, marking, allowed = wp.get_left_lane(), wp.left_lane_marking, carla.LaneChange.Left
            else:
                nb, marking, allowed = wp.get_right_lane(), wp.right_lane_marking, carla.LaneChange.Right
            if not (marking.lane_change in (allowed, carla.LaneChange.Both)
                    and nb is not None and wp.lane_id * nb.lane_id > 0 and nb.lane_type == carla.LaneType.Driving):
                return acc
            prev = wp
        return acc

    def _entry_safe(self, cand, occ):
        """Rear zone of the target lane (vehicle channel), landing footprint, kinematic limits and swept ego footprint."""
        env = self.env
        ego_len, ego_wid, off = self._ego_box()
        if not self._rear_clear(self.SIDE, cand['W'], cand['d0'], self.veh, ego_len, ego_wid):
            return False
        lxy, lyaw = cand['landing']
        rc = self._cells(self._footprint(ego_len, ego_wid, np.zeros(2)), lxy, lyaw)
        if ((rc >= 0).all(1) & (rc < self.grid).all(1)).all() and occ[rc[:, 0], rc[:, 1]].any():
            return False

        xy = cand['xy_flu']
        vis = (xy[:, 0] <= self.half - 0.2 - (ego_len / 2 + self.CLEARANCE_M) - abs(off[0])) & \
              (np.abs(xy[:, 1]) <= self.half - 0.2 - (ego_wid / 2 + self.CLEARANCE_M))
        xy = xy[:int(np.argmin(vis)) if not vis.all() else len(xy)]
        if len(xy) < 3:
            return False
        sig = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        heading = np.unwrap(np.arctan2(np.gradient(xy[:, 1], sig), np.gradient(xy[:, 0], sig)))
        kappa = np.gradient(heading, sig)
        v = env.vehicle.get_velocity()
        v0 = float(np.linalg.norm(world_vector_to_flu([v.x, v.y, v.z], cand['M'])[:2]))
        vt = self.V_BP_KMH / 3.6
        T = 2 * sig[-1] / max(v0 + vt, 0.1)
        sc = quintic(0.0, v0, 0.0, sig[-1], vt, 0.0, T)
        t = np.linspace(0, T, int(np.ceil(T / 0.1)) + 1)
        st, sd, sdd = np.clip(peval(sc, t), 0, sig[-1]), peval(sc, t, 1), peval(sc, t, 2)
        k_t = np.interp(st, sig, kappa)
        if (np.abs(kappa).max() > self.MAX_CURVATURE or np.abs(sdd).max() > self.MAX_LON_ACCEL
                or (sd ** 2 * np.abs(k_t)).max() > self.MAX_LAT_ACCEL):
            return False
        dense = np.arange(0.0, sig[-1], 0.1)
        dense_xy = np.stack([np.interp(dense, sig, xy[:, 0]), np.interp(dense, sig, xy[:, 1])], 1)
        return self._sweep(dense_xy, occ, heading=np.interp(dense, sig, heading)) is None

    def rear_box(self, side, lane_width, d0, ego_len, ego_wid):
        """Ego-FLU box of the neighbour lane on `side` ('left' / 'right'), from beside the ego to REAR_CHECK_M behind it."""
        y_c = (lane_width if side == 'left' else -lane_width) - d0
        hw = ego_wid / 2 + self.CLEARANCE_M
        return -(ego_len / 2 + self.REAR_CHECK_M), ego_len / 2, y_c - hw, y_c + hw

    def _rear_clear(self, side, lane_width, d0, occ, ego_len, ego_wid):
        """Omega_LR for a left bypass, Omega_RR for a right bypass: no vehicle cell inside the zone."""
        x0, x1, y0, y1 = self.rear_box(side, lane_width, d0, ego_len, ego_wid)
        r0, r1 = (int(np.floor((v + self.half) / BEV_RES)) for v in (x0, x1))
        c0, c1 = (int(np.floor((v + self.half) / BEV_RES)) for v in (y0, y1))
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, self.grid - 1), min(c1, self.grid - 1)
        return not occ[r0:r1 + 1, c0:c1 + 1].any()

    # ------------------------------------------------------------ frozen path following
    def _project(self):
        p = self.path
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        loc = self.env.vehicle.get_location()
        e = np.array([loc.x, loc.y])
        a, ab = p[:-1], np.diff(p, axis=0)
        t = np.clip(((e - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0.0, 1.0)
        d = np.linalg.norm(a + ab * t[:, None] - e, axis=1)
        i = int(np.argmin(d))
        return float(s[i] + t[i] * np.linalg.norm(ab[i])), float(d[i]), s

    def _samples(self, n):
        s_proj, _, s = self._project()
        q = s_proj + np.arange(1, n + 1) * self.SAMPLE_M
        q = q[q <= s[-1]]
        return np.stack([np.interp(q, s, self.path[:, 0]), np.interp(q, s, self.path[:, 1])], axis=1)

    def _path_ways(self):
        env = self.env
        pts = self._samples(self.ROUTE_POINTS)
        origin = torch.tensor([[[250.0, 250.0]]], device='cuda', dtype=torch.float32)
        if len(pts) == 0:
            ways = origin
        else:
            ways = torch.cat([origin, env.renderer.world_to_pix_crop_batched(*self._batched(pts))], dim=0)
        env.ways_bev = ways
        return ways

    def _batched(self, pts):
        tr = self.env.vehicle.get_transform()
        n = len(pts)
        ways = torch.tensor(np.asarray(pts), device='cuda', dtype=torch.float32).unsqueeze(1)
        pos = torch.tensor([[tr.location.x, tr.location.y]] * n, device='cuda', dtype=torch.float32).unsqueeze(1)
        yaw = torch.tensor([[math.radians(tr.rotation.yaw)]] * n, device='cuda', dtype=torch.float32).unsqueeze(1)
        return ways, pos, yaw

    # ------------------------------------------------------------ occupancy
    def _ego_box(self):
        bb = self.env.vehicle.bounding_box
        return 2 * bb.extent.x, 2 * bb.extent.y, np.array([bb.location.x, -bb.location.y])

    def _occupancy_flu(self, danger_mask):
        """BEV danger mask (row 0 = farthest ahead, column = right) -> FLU grid (row = +x, column = +y), ego cleared."""
        ego_len, ego_wid, _ = self._ego_box()
        occ = danger_mask > 0
        c = self.grid // 2
        hl = int(math.ceil((ego_len / 2 + self.EGO_CLEAR_M) / BEV_RES))
        hw = int(math.ceil((ego_wid / 2 + self.EGO_CLEAR_M) / BEV_RES))
        occ[c - hl:c + hl + 1, c - hw:c + hw + 1] = False
        return occ[::-1, ::-1].copy()

    def _footprint(self, length, width, offset):
        m = self.CLEARANCE_M
        lx = np.linspace(-length / 2 - m, length / 2 + m, 25)
        ly = np.linspace(-width / 2 - m, width / 2 + m, 13)
        return np.stack(np.meshgrid(lx, ly, indexing='ij'), -1).reshape(-1, 2) + offset

    def _cells(self, fp, xy, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.floor((fp @ np.array([[c, s], [-s, c]]) + xy + self.half) / BEV_RES).astype(int)

    def _sweep(self, xy, occ, ignore_m=0.0, heading=None):
        """Arc length of the first occupied ego footprint along an FLU polyline, or None. Stops at the BEV edge."""
        if len(xy) < 2:
            return None
        ego_len, ego_wid, off = self._ego_box()
        fp = self._footprint(ego_len, ego_wid, off)
        sig = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        if heading is None:
            d = np.gradient(xy, axis=0)
            heading = np.arctan2(d[:, 1], d[:, 0])
        for i in range(len(xy)):
            rc = self._cells(fp, xy[i], heading[i])
            if not ((rc >= 0).all() and (rc < self.grid).all()):
                return None
            if sig[i] > ignore_m and occ[rc[:, 0], rc[:, 1]].any():
                return float(sig[i])
        return None
