"""
Smooth lane changes in planned routes.

The global route planner encodes a lane change as a jump between two consecutive 1 m waypoints (one lane width
sideways, often a few metres ahead as well). The ego cannot follow that, so the off-track check ends the episode right
at the jump. smooth_lane_jumps() turns each jump into a gradual lane change over up to LENGTH_M, placed before the jump
if the target lane already runs alongside there, otherwise just after it (along the source lane, which usually
continues past the jump). Routes that cannot be smoothed are reported so the caller can draw another route.
"""
import math

import carla

JUMP_M = 2.5        # consecutive waypoints farther apart than this are a lane jump
LENGTH_M = 20       # length of the smoothed lane change (waypoints, 1 m apart)
MIN_BLEND_M = 6     # shortest acceptable lane change
MIN_START_M = 8     # a jump closer to the route start: start the route in the target lane instead


class BlendedWaypoint:
    """Waypoint stand-in on the lane-change path: own transform, other attributes from the nearest real waypoint."""

    def __init__(self, location, rotation, base):
        self.transform = carla.Transform(location, rotation)
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)


def _dist(a, b):
    la, lb = a.transform.location, b.transform.location
    return math.hypot(la.x - lb.x, la.y - lb.y)


def _neighbour_towards(wp, target_loc):
    """The neighbouring lane of wp on the side of target_loc (same direction, driving), or None."""
    f = wp.transform.get_forward_vector()
    loc = wp.transform.location
    side = -(target_loc.x - loc.x) * f.y + (target_loc.y - loc.y) * f.x      # > 0: target on the right
    nb = wp.get_right_lane() if side > 0 else wp.get_left_lane()
    if nb is None or nb.lane_type != carla.LaneType.Driving or nb.lane_id * wp.lane_id < 0:
        return None
    return nb


def _smooth(w):
    return w * w * (3 - 2 * w)


def _blend(src_pts, dst_pts, opts):
    """Points moving from src_pts (weight 0) to dst_pts (weight 1)."""
    k = len(src_pts)
    out = []
    for n, (a_wp, b_wp, opt) in enumerate(zip(src_pts, dst_pts, opts)):
        w = _smooth((n + 1) / (k + 1))
        a, b = a_wp.transform.location, b_wp.transform.location
        loc = carla.Location(a.x + w * (b.x - a.x), a.y + w * (b.y - a.y), a.z + w * (b.z - a.z))
        base = b_wp if w >= 0.5 else a_wp
        out.append((BlendedWaypoint(loc, base.transform.rotation, base), opt))
    return out


def _follow(start_wp, until_loc, opt):
    """Waypoints along start_wp's lane (1 m apart) until within JUMP_M of until_loc."""
    out, cur = [], start_wp
    for _ in range(15):
        if cur.transform.location.distance(until_loc) <= JUMP_M:
            break
        nxt = cur.next(1.0)
        if not nxt:
            break
        cur = nxt[0]
        out.append((cur, opt))
    return out


def _before(route, i):
    """Lane change on the waypoints before the jump; then follow the target lane up to route[i + 1]."""
    target = route[i + 1][0].transform.location
    nbs = []
    for j in range(i, max(-1, i - LENGTH_M), -1):
        nb = _neighbour_towards(route[j][0], target)
        if nb is None:
            break
        nbs.append(nb)
    if len(nbs) < MIN_BLEND_M:
        return None
    nbs.reverse()
    first = i + 1 - len(nbs)
    src = [route[j][0] for j in range(first, i + 1)]
    new = _blend(src, nbs, [route[j][1] for j in range(first, i + 1)])
    tail = _follow(nbs[-1], target, route[i + 1][1])
    return first, i + 1, new + tail


def _after(route, i):
    """Follow the source lane past the jump, then change lanes on the waypoints after it."""
    source = route[i][0].transform.location
    nbs = []
    for j in range(i + 1, min(len(route), i + 1 + LENGTH_M)):
        nb = _neighbour_towards(route[j][0], source)
        if nb is None:
            break
        nbs.append(nb)
    if len(nbs) < MIN_BLEND_M:
        return None
    k = len(nbs)
    fill = _follow(route[i][0], nbs[0].transform.location, route[i][1])
    dst = [route[j][0] for j in range(i + 1, i + 1 + k)]
    new = _blend(nbs, dst, [route[j][1] for j in range(i + 1, i + 1 + k)])
    return i + 1, i + 1 + k, fill + new


def smooth_lane_jumps(route):
    """route: list of (waypoint, RoadOption). Returns (route, ok); ok is False if a jump could not be smoothed."""
    route = list(route)
    i = 0
    while i < len(route) - 1:
        if _dist(route[i][0], route[i + 1][0]) <= JUMP_M:
            i += 1
            continue
        if i + 1 < MIN_START_M:                        # jump right at the start: start in the target lane
            route = route[i + 1:]
            i = 0
            continue
        fix = _before(route, i) or _after(route, i)
        if fix is None:
            return route, False
        a, b, pts = fix
        route[a:b] = pts
        i = a + len(pts) - 1
    ok = len(route) > 1 and all(_dist(route[j][0], route[j + 1][0]) <= JUMP_M for j in range(len(route) - 1))
    return route, ok
