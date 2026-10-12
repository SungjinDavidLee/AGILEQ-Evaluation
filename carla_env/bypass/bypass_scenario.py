"""
Bypass scenario: stationary vehicles on the ego route (training and evaluation), and bypass counters for logging.

Off by default (BypassScenario.ENABLED = False); only active in TOWNS. The counters are kept either way.
"""
import random

import numpy as np
import carla


def legal_left(w):
    L = w.get_left_lane()
    return (L is not None and L.lane_type == carla.LaneType.Driving and L.lane_id * w.lane_id > 0
            and w.left_lane_marking.lane_change in (carla.LaneChange.Left, carla.LaneChange.Both))


class BypassScenario:
    ENABLED = False
    TOWNS = ('Town03', 'Town04', 'Town05')
    SPAWN_PROB = 0.5               # per eligible route stretch
    MAX_PER_EPISODE = 2
    AHEAD_M = (30, 40)             # spawn distance ahead of the ego along the route
    LEGAL_BACK_M = 20              # left change legal from LEGAL_BACK_M behind to LEGAL_AHEAD_M ahead of the spot
    LEGAL_AHEAD_M = 30
    JUNCTION_M = 20                # no junction this close to the spot
    DECISION_GAP_M = 100           # one spawn decision per stretch of this length
    CLEAR_M = 10.0                 # no other vehicle this close to the spot
    REMOVE_BEHIND_M = 20           # removed once the ego is this far past it
    VANISH_S = 20.0                # removed once the ego has waited this long behind it in G (None: never)
    VANISH_NEAR_M = 20.0           # waiting counts only within this distance
    BASE_TYPES = ('car', 'van')
    ROUTE_BIAS = 0.3               # training: share of episodes whose route has a spot within ROUTE_BIAS_M of the start
    ROUTE_BIAS_M = 100
    ROUTE_TRIES = 20
    EPISODE_B_PROB = 0.35          # A/B episodes (training): B = short route with one stationary vehicle, A = plain route
    B_ROUTE_AFTER_M = 120          # B: the route ends this far after its first spot

    def __init__(self, env):
        self.env = env
        self.totals = dict(spawned=0, triggers=0, entries=0, success=0, fail=0, vanished=0, f_steps=0, steps=0)
        self.spawned_cars = []
        self.eligible = np.zeros(0, bool)
        self._town_ok = None
        self.ep_type = None
        self.reset()

    def active(self):
        if not self.ENABLED:
            return False
        if self._town_ok is None:
            self._town_ok = self.env.world.get_map().name.split('/')[-1] in self.TOWNS
        return self._town_ok

    def reset(self):
        """Call after the new route is set and the old NPCs are destroyed."""
        self.spawned_cars = []
        self.next_idx = 0
        self.prev_mode = 'G'
        self.ep = dict(spawned=0, triggers=0, entries=0, success=0, fail=0, vanished=0, f_steps=0)
        self.wait = {}
        self.stop_gap = None
        ready = getattr(self.env, 'route_waypoints', None) and self.active()
        self.eligible = self._eligible(self.env.route_waypoints) if ready else np.zeros(0, bool)

    def begin_episode(self):
        """Training, at route selection: draw the episode type ('A' or 'B'; None outside A/B mode)."""
        self.ep_type = None
        if self.active() and self.EPISODE_B_PROB is not None:
            self.ep_type = 'B' if random.random() < self.EPISODE_B_PROB else 'A'
        return self.ep_type

    def prefers_route(self):
        """Training, once per episode: whether to redraw the route until it has an early spot."""
        if self.ep_type is not None:
            return self.ep_type == 'B'
        return self.active() and random.random() < self.ROUTE_BIAS

    def shape_route(self, route_waypoints):
        """B episodes: end the route B_ROUTE_AFTER_M after the first early spot."""
        el = self._eligible(route_waypoints)[self.AHEAD_M[0]:self.ROUTE_BIAS_M + 1]
        return route_waypoints[:self.AHEAD_M[0] + int(np.argmax(el)) + self.B_ROUTE_AFTER_M]

    def has_early_spot(self, route_waypoints):
        return bool(self._eligible(route_waypoints)[self.AHEAD_M[0]:self.ROUTE_BIAS_M + 1].any())

    def _eligible(self, route_waypoints):
        ws = [w for w, _ in route_waypoints]
        n = len(ws)
        leg = np.array([legal_left(w) for w in ws], bool)
        jn = np.array([w.is_junction for w in ws], bool)
        el = np.zeros(n, bool)
        for i in range(self.LEGAL_BACK_M, n - self.LEGAL_AHEAD_M):
            el[i] = leg[i - self.LEGAL_BACK_M:i + self.LEGAL_AHEAD_M + 1].all() and \
                not jn[max(0, i - self.JUNCTION_M):i + self.JUNCTION_M + 1].any()
        return el

    def update(self, done):
        """Call once per env step, after the bypass module and the termination flags are updated."""
        mode = self.env.bypass.mode
        self.totals['steps'] += 1
        if mode == 'F':
            self._count('f_steps')
        if self.env.bypass.trigger_count == self.env.bypass.TRIGGER_STEPS:
            self._count('triggers')
        if mode == 'F' and self.prev_mode != 'F':
            self._count('entries')
        if self.prev_mode == 'F' and mode != 'F':
            self._count('success' if self.env.bypass.last_exit in ('returned', 'path_end') else 'fail')
        elif mode == 'F' and done:
            self._count('fail')
        self.prev_mode = mode
        if self.active() and not done:
            self._measure_gap()
            self._vanish_waited()
            self._remove_passed()
            self._try_spawn()

    def _count(self, key):
        self.ep[key] += 1
        self.totals[key] += 1

    def _try_spawn(self):
        if self.ep_type == 'A':
            return
        max_n = 1 if self.ep_type == 'B' else self.MAX_PER_EPISODE
        if self.spawned_cars or self.ep['spawned'] >= max_n:
            return
        c = self.env.current_waypoint_index + random.randint(*self.AHEAD_M)
        if c < self.next_idx or c >= len(self.eligible) or not self.eligible[c]:
            return
        self.next_idx = c + self.DECISION_GAP_M
        if self.ep_type == 'B' or random.random() < self.SPAWN_PROB:
            self._spawn(c)

    def _spawn(self, c):
        env = self.env
        tf = env.route_waypoints[c][0].transform
        loc = tf.location
        for v in env.world.get_actors().filter('vehicle.*'):
            if v.id != env.vehicle.id and v.get_location().distance(loc) < self.CLEAR_M:
                return
        bps = [bp for bp in env.vehicle_bp_lib
               if bp.has_attribute('base_type') and bp.get_attribute('base_type').as_str().lower() in self.BASE_TYPES]
        bp = random.choice(bps or list(env.vehicle_bp_lib))
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name', 'stationary')
        actor = env.world.try_spawn_actor(bp, carla.Transform(carla.Location(loc.x, loc.y, loc.z + 0.3), tf.rotation))
        if actor is None:
            return
        actor.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        env.npc_vehicle_list.append(actor.id)
        self.spawned_cars.append((actor.id, c))
        self._count('spawned')

    def _measure_gap(self):
        """Smallest bumper gap to a spawned vehicle ahead while the ego is in G below the trigger speed."""
        env = self.env
        if env.speed >= env.bypass.TRIGGER_SPEED_KMH or env.bypass.mode == 'F':
            return
        ego = env.vehicle
        for aid, c in self.spawned_cars:
            actor = env.world.get_actor(aid)
            if actor is None or env.current_waypoint_index >= c:
                continue
            gap = actor.get_location().distance(ego.get_location()) - actor.bounding_box.extent.x - ego.bounding_box.extent.x
            self.stop_gap = gap if self.stop_gap is None else min(self.stop_gap, gap)

    def _vanish_waited(self):
        """Remove a spawned vehicle once the ego has waited behind it for VANISH_S without starting a bypass."""
        env = self.env
        if self.VANISH_S is None or env.bypass.mode == 'F' or env.speed >= env.bypass.TRIGGER_SPEED_KMH:
            return
        for aid, c in list(self.spawned_cars):
            actor = env.world.get_actor(aid)
            if actor is None or env.current_waypoint_index >= c or \
                    actor.get_location().distance(env.vehicle.get_location()) > self.VANISH_NEAR_M:
                continue
            self.wait[aid] = self.wait.get(aid, 0) + 1
            if self.wait[aid] >= self.VANISH_S * env.fps:
                self._destroy(aid, c)
                self._count('vanished')

    def _destroy(self, aid, c):
        actor = self.env.world.get_actor(aid)
        if actor is not None:
            actor.destroy()
        if aid in self.env.npc_vehicle_list:
            self.env.npc_vehicle_list.remove(aid)
        self.spawned_cars.remove((aid, c))

    def _remove_passed(self):
        i = self.env.current_waypoint_index
        for aid, c in list(self.spawned_cars):
            if i > c + self.REMOVE_BEHIND_M:
                self._destroy(aid, c)

    def info(self):
        t = self.totals
        done = t['success'] + t['fail']
        out = {} if self.stop_gap is None else {'bypass_stop_gap': self.stop_gap}
        return {**out, 'bypass_f': self.env.bypass.mode == 'F', 'bypass_episode_b': int(self.ep_type == 'B'),
                'bypass_spawned': self.ep['spawned'], 'bypass_triggers': self.ep['triggers'],
                'bypass_entries': self.ep['entries'],
                'bypass_success': self.ep['success'], 'bypass_fail': self.ep['fail'], 'bypass_vanished': self.ep['vanished'],
                'bypass_success_rate_total': t['success'] / done if done else 0.0,
                'bypass_f_step_ratio_total': t['f_steps'] / max(t['steps'], 1),
                'bypass_spawned_total': t['spawned'], 'bypass_success_total': t['success']}
