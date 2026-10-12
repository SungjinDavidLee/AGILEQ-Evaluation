"""
Bypass path generation: candidate = plan_bypass(method, context, rejoin_goal, previous_plan)

method in {frenet, lattice, astar, rrtstar, qp, nmpc, mppi}. All methods work in the ego FLU frame (x forward, y left),
take the same context (reference line, occupancy, drivable area, ego state, limits) and return a path that starts at the
ego and ends on the route centre line inside the rejoin window. Pure numpy / scipy, no simulator dependency.
"""
import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import lsq_linear, minimize

try:
    from .geometry import quintic, peval
except ImportError:
    from geometry import quintic, peval

METHODS = ('frenet', 'lattice', 'astar', 'rrtstar', 'qp', 'nmpc', 'mppi')
METHOD_NAMES = {'frenet': 'Frenet', 'lattice': 'State Lattice', 'astar': 'A*', 'rrtstar': 'RRT*',
                'qp': 'Corridor QP', 'nmpc': 'NMPC', 'mppi': 'MPPI'}


@dataclass
class BypassContext:
    ref_xy: np.ndarray            # route centre line in ego FLU [N, 2]
    occ: np.ndarray               # obstacle grid, row -> +x, col -> +y
    x_min: float
    y_min: float
    res: float
    lane_width: float
    side: int = 1                 # +1 left bypass, -1 right bypass
    drivable: np.ndarray = None   # allowed area grid (None: everything)
    v0: float = 0.0
    ego_len: float = 4.8
    ego_wid: float = 2.1
    clearance: float = 0.2
    v_target: float = 15 / 3.6
    a_lon: float = 1.5
    a_lat_max: float = 3.5
    kappa_max: float = 0.35
    budget_s: float = 2.0
    seed: int = 0
    _cache: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- reference line
    def ref(self):
        if 'ref' not in self._cache:
            xy = np.asarray(self.ref_xy, float)
            keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
            xy = xy[keep]
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            yaw = np.unwrap(np.arctan2(np.gradient(xy[:, 1], s), np.gradient(xy[:, 0], s)))
            self._cache['ref'] = (xy, s, yaw, np.gradient(yaw, s))
        return self._cache['ref']

    def ref_at(self, s):
        xy, ss, yaw, k = self.ref()
        s = np.clip(np.asarray(s, float), ss[0], ss[-1])
        p = np.stack([np.interp(s, ss, xy[:, 0]), np.interp(s, ss, xy[:, 1])], -1)
        return p, np.interp(s, ss, yaw), np.interp(s, ss, k)

    def to_xy(self, s, d):
        p, yaw, _ = self.ref_at(s)
        d = np.asarray(d, float)
        return p + np.stack([-np.sin(yaw), np.cos(yaw)], -1) * d[..., None]

    def project(self, pts):
        """Nearest-segment projection of points [M, 2] -> (s, d)."""
        xy, ss, _, _ = self.ref()
        pts = np.atleast_2d(pts)
        a, ab = xy[:-1], np.diff(xy, axis=0)
        L2 = np.maximum((ab * ab).sum(1), 1e-12)
        t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(-1) / L2, 0, 1)
        q = a[None] + ab[None] * t[..., None]
        dist = np.linalg.norm(pts[:, None, :] - q, axis=-1)
        i = dist.argmin(1)
        r = np.arange(len(pts))
        s = ss[i] + t[r, i] * np.sqrt(L2[i])
        cross = ab[i, 0] * (pts[:, 1] - a[i, 1]) - ab[i, 1] * (pts[:, 0] - a[i, 0])
        return s, np.sign(cross) * dist[r, i]

    def ego_sd(self):
        if 'ego_sd' not in self._cache:
            s, d = self.project(np.zeros((1, 2)))
            _, yaw, _ = self.ref_at(s[0])
            self._cache['ego_sd'] = (float(s[0]), float(d[0]), float(math.atan2(math.sin(-yaw), math.cos(-yaw))))
        return self._cache['ego_sd']

    # ---------------------------------------------------------------- collision fields
    def fields(self):
        if 'fields' not in self._cache:
            clear = distance_transform_edt(~self.occ) * self.res
            drv = None if self.drivable is None else distance_transform_edt(self.drivable) * self.res
            self._cache['fields'] = (clear, drv)
        return self._cache['fields']

    def lookup(self, grid, pts, outside):
        r = np.floor((pts[..., 0] - self.x_min) / self.res).astype(int)
        c = np.floor((pts[..., 1] - self.y_min) / self.res).astype(int)
        ok = (r >= 0) & (c >= 0) & (r < grid.shape[0]) & (c < grid.shape[1])
        out = np.full(r.shape, outside, float)
        out[ok] = grid[r[ok], c[ok]]
        return out

    def circles(self):
        n = 5
        off = np.linspace(-self.ego_len / 2 + self.ego_len / (2 * n), self.ego_len / 2 - self.ego_len / (2 * n), n)
        return off, math.hypot(self.ego_len / (2 * n), self.ego_wid / 2)

    def penetration(self, xy, heading):
        """Per pose: obstacle penetration (m) of the ego circle cover and drivable-area violation (m); 0 means free."""
        off, r = self.circles()
        xy, heading = np.atleast_2d(xy), np.atleast_1d(heading)
        cen = xy[:, None, :] + off[None, :, None] * np.stack([np.cos(heading), np.sin(heading)], -1)[:, None, :]
        clear, drv = self.fields()
        obs = np.maximum(0.0, r + self.clearance - self.lookup(clear, cen, 1e3)).max(1)
        if drv is None:
            return obs, np.zeros_like(obs)
        road = np.maximum(0.0, self.ego_wid / 2 - self.lookup(drv, cen, 0.0)).max(1)
        return obs, road

    def free(self, xy, heading):
        obs, road = self.penetration(xy, heading)
        return (obs <= 0) & (road <= 0)


# -------------------------------------------------------------------- path helpers
def resample(xy, ds=0.5):
    xy = np.asarray(xy, float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-6]
    xy = xy[keep]
    if len(xy) < 2:
        return xy
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    q = np.r_[np.arange(0, s[-1] - 0.5 * ds, ds), s[-1]]
    return np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], 1)


def headings(xy):
    d = np.gradient(xy, axis=0)
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def curvature(xy):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    return np.gradient(headings(xy), s), s


def speed_profile(ctx, s):
    return np.minimum(ctx.v_target, np.sqrt(ctx.v0 ** 2 + 2 * ctx.a_lon * np.maximum(s, 0)))


def kinematic_ok(ctx, xy):
    p = resample(xy, 0.5)
    if len(p) < 4:
        return False
    k, s = curvature(p)
    v = speed_profile(ctx, s)
    return np.abs(k).max() <= ctx.kappa_max and (v ** 2 * np.abs(k)).max() <= ctx.a_lat_max


def path_free(ctx, xy, ds=0.5):
    p = resample(xy, ds)
    return bool(ctx.free(p, headings(p)).all()) if len(p) >= 2 else False


def sd_path(ctx, s, d):
    return ctx.to_xy(s, d)


def nominal_profile(ctx, goal, L_out, L_ret, s_ret):
    """Out (quintic to target_d), hold, back (quintic to 0) in (s, d)."""
    s0, d0, psi0 = ctx.ego_sd()
    W = goal['target_d']
    dp0 = float(np.clip(math.tan(psi0), -0.5, 0.5))
    s1 = s0 + L_out
    s3 = s_ret + L_ret
    if s1 > s_ret + 1e-6 or s3 > goal['s_land_max'] + 1e-6 or s3 < goal['s_land_min'] - 1e-6:
        return None
    ss = np.r_[np.arange(s0, s3 - 0.25, 0.5), s3]
    dd = np.full_like(ss, W)
    m = ss <= s1
    dd[m] = peval(quintic(d0, dp0, 0.0, W, 0.0, 0.0, L_out), ss[m] - s0)
    m = ss >= s_ret
    dd[m] = peval(quintic(W, 0.0, 0.0, 0.0, 0.0, 0.0, L_ret), ss[m] - s_ret)
    return ss, dd


def _quintic_edge(da, slope_a, db, L, n):
    t = np.linspace(0, L, n)
    return t, peval(quintic(da, slope_a, 0.0, db, 0.0, 0.0, L), t)


# -------------------------------------------------------------------- methods
def plan_frenet(ctx, goal, prev=None):
    """Enumerate out / hold / back quintic profiles and keep the cheapest valid one."""
    cands = []
    lengths = range(8, 27, 2)
    for L_out in lengths:
        for L_ret in lengths:
            for s_ret in np.arange(goal['s_clear'], goal['s_land_max'] - 8 + 1e-6, 1.0):
                if nominal_profile(ctx, goal, L_out, L_ret, s_ret) is not None:
                    cands.append((s_ret + L_ret + 30.0 / L_out + 30.0 / L_ret, L_out, L_ret, s_ret))
    cands.sort()
    for n, (_, L_out, L_ret, s_ret) in enumerate(cands, 1):
        xy = sd_path(ctx, *nominal_profile(ctx, goal, L_out, L_ret, s_ret))
        if kinematic_ok(ctx, xy) and path_free(ctx, xy):
            return xy, {'candidates': len(cands), 'evaluated': n}
    return None, {'candidates': len(cands), 'evaluated': len(cands)}


def _lattice(ctx, goal, ds=3.0, levels=5, max_span=4):
    s0, d0, psi0 = ctx.ego_sd()
    W = goal['target_d']
    dp0 = float(np.clip(math.tan(psi0), -0.5, 0.5))
    lv = np.linspace(0, W, levels)
    st = s0 + ds * np.arange(1, int((goal['s_land_max'] - s0) / ds) + 1)
    nodes = [(s0, d0)] + [(s, d) for s in st for d in lv]
    cost = {0: 0.0}
    parent = {}
    seg = {}
    order = [0] + list(range(1, len(nodes)))
    idx = {(round(s, 3), round(d, 3)): i for i, (s, d) in enumerate(nodes)}
    n_edges = 0
    for i in order:
        if i not in cost:
            continue
        sa, da = nodes[i]
        slope = dp0 if i == 0 else 0.0
        k_a = 0 if i == 0 else int(round((sa - s0) / ds))
        for m in range(1, max_span + 1):
            k_b = k_a + m
            if k_b > len(st):
                break
            sb = s0 + ds * k_b
            for db in lv:
                j = idx[(round(sb, 3), round(db, 3))]
                L = sb - sa
                t, d = _quintic_edge(da, slope, db, L, max(4, int(L / 0.5) + 1))
                n_edges += 1
                s = sa + t
                xy = sd_path(ctx, s, d)
                dd2 = np.gradient(np.gradient(d, t), t)
                if np.abs(dd2).max() > ctx.kappa_max:
                    continue
                if not ctx.free(xy, headings(xy)).all():
                    continue
                c = cost[i] + L + 4.0 * float(np.sum(dd2 ** 2) * (t[1] - t[0])) + 0.6 * float(np.mean(np.abs(d)) / max(abs(W), 1e-6) * L)
                if c < cost.get(j, np.inf):
                    cost[j], parent[j], seg[j] = c, i, (s, d)
    goals = [j for j, (s, d) in enumerate(nodes) if j in cost and abs(d) < 1e-9 and goal['s_land_min'] <= s <= goal['s_land_max']
             and j != 0 and _passed_detour(j, parent, nodes, W)]
    if not goals:
        return None, {'edges': n_edges}
    j = min(goals, key=lambda g: cost[g])
    ss, dd = [], []
    while j in parent:
        s, d = seg[j]
        ss.insert(0, s), dd.insert(0, d)
        j = parent[j]
    s = np.concatenate([x[:-1] for x in ss[:-1]] + [ss[-1]])
    d = np.concatenate([x[:-1] for x in dd[:-1]] + [dd[-1]])
    return (s, d), {'edges': n_edges}


def _passed_detour(j, parent, nodes, W):
    """The chain must actually reach the target side (otherwise 'stay in lane' would be a goal)."""
    while j in parent:
        if abs(nodes[j][1]) >= 0.5 * abs(W):
            return True
        j = parent[j]
    return False


def plan_lattice(ctx, goal, prev=None):
    out, info = _lattice(ctx, goal)
    if out is None:
        return None, info
    return sd_path(ctx, *out), info


def plan_astar(ctx, goal, prev=None):
    """Dense (s, d) grid A*, then quintic connection through the turning keyframes."""
    s0, d0, psi0 = ctx.ego_sd()
    W = goal['target_d']
    hs, hd = 0.5, 0.1
    d_lo, d_hi = min(0.0, W) - 0.6, max(0.0, W) + 0.6
    nd = int(round((d_hi - d_lo) / hd)) + 1
    ns = int((goal['s_land_max'] - s0) / hs) + 1
    dvals = d_lo + hd * np.arange(nd)
    i0 = int(np.argmin(np.abs(dvals - d0)))
    izero = int(np.argmin(np.abs(dvals)))
    ireach = np.where(np.abs(dvals - W) <= 0.35)[0]
    free_cache = {}

    def node_free(k, i, di):
        key = (k, i, di)
        if key not in free_cache:
            s = s0 + k * hs
            xy = ctx.to_xy(s, dvals[i])
            _, yaw, _ = ctx.ref_at(s)
            free_cache[key] = bool(ctx.free(xy[None], np.array([yaw + math.atan2(di * hd, hs)]))[0])
        return free_cache[key]

    # state: (k, i, reached_side)
    start = (0, i0, 0)
    g = {start: 0.0}
    par = {}
    pq = [(goal['s_land_min'] - s0, 0.0, start)]
    found, expanded = None, 0
    t_start = time.perf_counter()
    while pq:
        if time.perf_counter() - t_start > ctx.budget_s:
            break
        f, gc, cur = heapq.heappop(pq)
        if gc > g.get(cur, np.inf) + 1e-9:
            continue
        k, i, r = cur
        expanded += 1
        s = s0 + k * hs
        if r and i == izero and s >= goal['s_land_min']:
            found = cur
            break
        if k + 1 >= ns:
            continue
        for di in (-3, -2, -1, 0, 1, 2, 3):
            j = i + di
            if j < 0 or j >= nd or not node_free(k + 1, j, di):
                continue
            r2 = 1 if (r or j in ireach) else 0
            c = (gc + math.hypot(hs, di * hd) + 0.6 * abs(dvals[j]) / max(abs(W), 1e-6) * hs
                 + 2.0 * (di * hd / hs) ** 2 * hs)
            nxt = (k + 1, j, r2)
            if c < g.get(nxt, np.inf):
                g[nxt], par[nxt] = c, cur
                h = max(0.0, goal['s_land_min'] - (s + hs)) + (0 if r2 else abs(W))
                heapq.heappush(pq, (c + h, c, nxt))
    if found is None:
        return None, {'expanded': expanded}
    chain = [found]
    while chain[-1] in par:
        chain.append(par[chain[-1]])
    chain = chain[::-1]
    s = s0 + hs * np.array([c[0] for c in chain])
    d = dvals[[c[1] for c in chain]]
    d[0] = d0
    # keyframes where the grid path changes slope, connected by quintic segments (zero slope at keyframes)
    slope = np.sign(np.diff(d).round(6))
    keys = [0] + [m for m in range(1, len(d) - 1) if slope[m] != slope[m - 1]] + [len(d) - 1]
    keys = _merge_keys(s, keys, 6.0)
    ss, dd = [], []
    for a, b in zip(keys[:-1], keys[1:]):
        L = s[b] - s[a]
        t, seg = _quintic_edge(d[a], math.tan(psi0) if a == 0 else 0.0, d[b], L, max(4, int(L / 0.5) + 1))
        ss.append(s[a] + t[:-1]), dd.append(seg[:-1])
    ss.append([s[-1]]), dd.append([d[-1]])
    return sd_path(ctx, np.concatenate(ss), np.concatenate(dd)), {'expanded': expanded}


def _merge_keys(s, keys, min_gap):
    out = [keys[0]]
    for k in keys[1:-1]:
        if s[k] - s[out[-1]] >= min_gap:
            out.append(k)
    if s[keys[-1]] - s[out[-1]] < min_gap and len(out) > 1:
        out[-1] = keys[-1]
    else:
        out.append(keys[-1])
    return out


def plan_rrtstar(ctx, goal, prev=None, n_iter=1500, step=1.5, radius=4.0, max_turn=math.radians(25)):
    """RRT* in x-y with a turn limit between consecutive edges, then spline smoothing."""
    rng = np.random.default_rng(ctx.seed)
    s0, d0, psi0 = ctx.ego_sd()
    W = goal['target_d']
    P = np.zeros((n_iter + 2, 2))
    H = np.zeros(n_iter + 2)
    C = np.zeros(n_iter + 2)
    parent = -np.ones(n_iter + 2, int)
    P[1] = [step, 0.0]
    C[1], parent[1], H[0], H[1] = step, 0, 0.0, 0.0
    n = 2
    s_rng = (s0, goal['s_land_max'] + 2)
    d_rng = (min(0.0, W) - 1.0, max(0.0, W) + 1.0)
    best, t_start, it = None, time.perf_counter(), 0

    def edge_ok(a, b):
        L = np.linalg.norm(b - a)
        m = max(2, int(L / 0.5) + 1)
        pts = a + (b - a) * np.linspace(0, 1, m)[:, None]
        return bool(ctx.free(pts, np.full(m, math.atan2(b[1] - a[1], b[0] - a[0]))).all())

    def turn(i, h):
        return abs(math.atan2(math.sin(h - H[i]), math.cos(h - H[i])))

    for it in range(n_iter):
        if time.perf_counter() - t_start > ctx.budget_s:
            break
        if rng.random() < 0.1:
            q = ctx.to_xy(rng.uniform(goal['s_land_min'], goal['s_land_max']), 0.0)
        else:
            q = ctx.to_xy(rng.uniform(*s_rng), rng.uniform(*d_rng))
        dist = np.linalg.norm(P[:n] - q, axis=1)
        i = int(dist.argmin())
        v = q - P[i]
        L = np.linalg.norm(v)
        if L < 1e-6:
            continue
        new = P[i] + v / L * min(step, L)
        near = np.where(np.linalg.norm(P[:n] - new, axis=1) <= radius)[0]
        cand = []
        for j in near:
            h = math.atan2(new[1] - P[j, 1], new[0] - P[j, 0])
            if turn(j, h) <= max_turn:
                cand.append((C[j] + np.linalg.norm(new - P[j]), j, h))
        cand.sort()
        pick = next(((c, j, h) for c, j, h in cand if edge_ok(P[j], new)), None)
        if pick is None:
            continue
        c, j, h = pick
        P[n], C[n], H[n], parent[n] = new, c, h, j
        for k in near:                                     # rewire
            if k == j or k <= 1:
                continue
            hk = math.atan2(P[k, 1] - new[1], P[k, 0] - new[0])
            ck = C[n] + np.linalg.norm(P[k] - new)
            if ck < C[k] and turn(n, hk) <= max_turn and abs(math.atan2(math.sin(hk - H[k]), math.cos(hk - H[k]))) <= max_turn \
                    and edge_ok(new, P[k]):
                parent[k], C[k], H[k] = n, ck, hk
        sn, dn = ctx.project(new[None])
        _, yaw, _ = ctx.ref_at(sn[0])
        if goal['s_land_min'] <= sn[0] <= goal['s_land_max'] and abs(dn[0]) <= 0.3 \
                and abs(math.atan2(math.sin(h - yaw), math.cos(h - yaw))) <= math.radians(10):
            chain = [n]
            while parent[chain[-1]] >= 0:
                chain.append(parent[chain[-1]])
            reached = np.abs(ctx.project(P[chain])[1]).max() >= 0.5 * abs(W)
            if reached and (best is None or C[n] < best[0]):
                best = (C[n], chain[::-1])
        n += 1
    if best is None:
        return None, {'iterations': it, 'nodes': n}
    return _smooth_sd(ctx, P[best[1]]), {'iterations': it, 'nodes': n}


def _smooth_sd(ctx, pts):
    """Smoothing spline of d(s) through the tree path, padded so that it leaves the ego with its heading and
    lands on the lane centre with zero slope."""
    from scipy.interpolate import UnivariateSpline
    s0, d0, psi0 = ctx.ego_sd()
    s, d = ctx.project(resample(pts, 1.0))
    order = np.argsort(s)
    s, d = s[order], d[order]
    keep = np.r_[True, np.diff(s) > 0.2]
    s, d = s[keep], d[keep]
    pre = s0 - np.arange(5, 0, -1) * 1.0
    post = s[-1] + np.arange(1, 6) * 1.0
    S = np.r_[pre, s, post]
    D = np.r_[d0 - math.tan(psi0) * (s0 - pre), d, np.zeros(5)]
    w = np.r_[np.full(5, 50.0), np.ones(len(s)), np.full(5, 50.0)]
    w[5] = w[-6] = 50.0
    try:
        f = UnivariateSpline(S, D, w=w, k=5, s=0.05 * len(S))
    except Exception:
        return pts
    q = np.arange(s0, s[-1] + 4.0, 0.5)                    # run into the zero-slope padding
    dd = f(q)
    dd[0] = d0
    return sd_path(ctx, q, dd)


def plan_qp(ctx, goal, prev=None, ds=1.0):
    """Lateral box-QP inside the free corridor around the State Lattice path (initial path time included)."""
    out, info = _lattice(ctx, goal)
    if out is None:
        return None, dict(info, init='lattice_failed')
    s_b, d_b = out
    s0, d0, psi0 = ctx.ego_sd()
    s = np.r_[np.arange(s_b[0], s_b[-1] - 0.5 * ds, ds), s_b[-1]]
    db = np.interp(s, s_b, d_b)
    n = len(s)
    lo, hi = np.empty(n), np.empty(n)
    for i in range(n):
        _, yaw, _ = ctx.ref_at(s[i])
        lo[i] = hi[i] = db[i]
        for sign, arr in ((-1, lo), (1, hi)):
            d = db[i]
            while abs(d - db[i]) < 2.0:
                d2 = d + sign * 0.1
                if not ctx.free(ctx.to_xy(s[i], d2)[None], np.array([yaw]))[0]:
                    break
                d = d2
            arr[i] = d
    lo[0] = hi[0] = d0
    lo[-1] = hi[-1] = 0.0
    if n > 2:
        lo[1] = hi[1] = d0 + math.tan(psi0) * ds
        lo[-2] = hi[-2] = 0.0
    rows, b = [], []
    for i in range(1, n - 1):                              # curvature ~ second difference
        r = np.zeros(n); r[i - 1], r[i], r[i + 1] = 1, -2, 1
        rows.append(4.0 * r / ds ** 2), b.append(0.0)
    for i in range(1, n - 2):                              # its rate
        r = np.zeros(n); r[i - 1], r[i], r[i + 1], r[i + 2] = -1, 3, -3, 1
        rows.append(2.0 * r / ds ** 3), b.append(0.0)
    for i in range(n):                                     # stay close to the initial path
        r = np.zeros(n); r[i] = 0.3
        rows.append(r), b.append(0.3 * db[i])
    res = lsq_linear(np.array(rows), np.array(b), bounds=(lo - 1e-9, hi + 1e-9), lsmr_tol='auto', max_iter=500)
    return sd_path(ctx, s, res.x), dict(info, qp_status=int(res.status))


def _shoot(ctx, x0, kap, ds):
    n = len(kap)
    psi = np.r_[0.0, np.cumsum(kap * ds)][:n + 1]
    x = np.r_[0.0, np.cumsum(np.cos(psi[:-1]) * ds)] + x0[0]
    y = np.r_[0.0, np.cumsum(np.sin(psi[:-1]) * ds)] + x0[1]
    return np.stack([x, y], -1), psi


def _rollout_cost(ctx, kap, ref_xy, ds, end_s, end_yaw):
    """kap: [K, N] curvature sequences (spatial steps ds) from the ego pose -> cost [K]."""
    K, N = kap.shape
    psi = np.concatenate([np.zeros((K, 1)), np.cumsum(kap * ds, 1)], 1)
    x = np.concatenate([np.zeros((K, 1)), np.cumsum(np.cos(psi[:, :-1]) * ds, 1)], 1)
    y = np.concatenate([np.zeros((K, 1)), np.cumsum(np.sin(psi[:, :-1]) * ds, 1)], 1)
    xy = np.stack([x, y], -1)
    obs, road = ctx.penetration(xy.reshape(-1, 2), psi.reshape(-1))
    obs, road = obs.reshape(K, N + 1), road.reshape(K, N + 1)
    track = ((xy - ref_xy[None]) ** 2).sum(-1)
    dk = np.diff(kap, axis=1)
    _, d_end = ctx.project(xy[:, -1])
    yaw_err = np.arctan2(np.sin(psi[:, -1] - end_yaw), np.cos(psi[:, -1] - end_yaw))
    return (0.2 * track.sum(1) + 20 * (kap ** 2).sum(1) + 200 * (dk ** 2).sum(1)
            + 2000 * (obs ** 2).sum(1) + 2000 * (road ** 2).sum(1) + 3000 * (d_end ** 2 + 4 * yaw_err ** 2))


def _init_controls(ctx, goal, ds):
    out, info = _lattice(ctx, goal)
    if out is None:
        prof = nominal_profile(ctx, goal, 12, 12, goal['s_clear'] + 2)
        if prof is None:
            return None, dict(info, init='none')
        out = prof
        info = dict(info, init='nominal')
    s_b, d_b = out
    tail = np.arange(s_b[-1] + ds, s_b[-1] + 6.0 + 1e-6, ds)          # route-aligned tail to settle on the lane centre
    xy = resample(sd_path(ctx, np.r_[s_b, tail], np.r_[d_b, np.zeros_like(tail)]), ds)
    k, _ = curvature(xy)
    kap = np.clip(k[:-1], -ctx.kappa_max, ctx.kappa_max)
    xs, _ = _shoot(ctx, xy[0], kap, ds)
    _, end_yaw, _ = ctx.ref_at(ctx.project(xy[-1:])[0][0])
    return (kap, xy, float(end_yaw)), info


def plan_nmpc(ctx, goal, prev=None, ds=1.0):
    """Direct shooting on a spatial kinematic model, warm-started from the State Lattice path (L-BFGS-B)."""
    init, info = _init_controls(ctx, goal, ds)
    if init is None:
        return None, info
    kap0, ref_xy, end_yaw = init

    def f(k):
        return float(_rollout_cost(ctx, k[None], ref_xy, ds, None, end_yaw)[0])

    r = minimize(f, kap0, method='L-BFGS-B', bounds=[(-ctx.kappa_max, ctx.kappa_max)] * len(kap0),
                 options={'maxiter': 60, 'maxfun': 4000})
    xy, _ = _shoot(ctx, np.zeros(2), r.x, ds)
    return xy, dict(info, nit=int(r.nit), improved=bool(r.fun < f(kap0) - 1e-6))


def plan_mppi(ctx, goal, prev=None, ds=1.0, K=256, iters=8, sigma=0.03):
    """Sampled curvature sequences around the State Lattice warm start, exponentially weighted updates."""
    init, info = _init_controls(ctx, goal, ds)
    if init is None:
        return None, info
    kap, ref_xy, end_yaw = init
    rng = np.random.default_rng(ctx.seed)
    for _ in range(iters):
        noise = rng.normal(0, sigma, (K, len(kap)))
        noise[0] = 0
        samples = np.clip(kap[None] + noise, -ctx.kappa_max, ctx.kappa_max)
        c = _rollout_cost(ctx, samples, ref_xy, ds, None, end_yaw)
        lam = max(1e-6, 0.1 * (np.median(c) - c.min()))
        w = np.exp(-(c - c.min()) / lam)
        kap = (w[:, None] * samples).sum(0) / w.sum()
    xy, _ = _shoot(ctx, np.zeros(2), kap, ds)
    return xy, info


PLANNERS = {'frenet': plan_frenet, 'lattice': plan_lattice, 'astar': plan_astar, 'rrtstar': plan_rrtstar,
            'qp': plan_qp, 'nmpc': plan_nmpc, 'mppi': plan_mppi}


def plan_bypass(method, context, rejoin_goal, previous_plan=None):
    """-> (xy path in ego FLU or None, info). rejoin_goal: target_d, s_clear, s_land_min, s_land_max."""
    t = time.perf_counter()
    path, info = PLANNERS[method](context, rejoin_goal, previous_plan)
    info = dict(info, method=method, time_s=time.perf_counter() - t)
    return path, info


def rejoin_goal_from_occupancy(ctx, s_block, max_len=25.0, assumed_len=7.0, land_window=25.0):
    """Rejoin window behind the blocking obstacle: scan the ego lane forward from the blocked point until the lane
    is free for one ego length; fall back to assumed_len when the obstacle end is not visible."""
    W = ctx.lane_width * ctx.side
    s, end, free_run = s_block - 2.0, None, 0.0
    while s < s_block + max_len:
        xy = ctx.to_xy(s, 0.0)
        r = np.floor((xy[0] - ctx.x_min) / ctx.res)
        c = np.floor((xy[1] - ctx.y_min) / ctx.res)
        if not (0 <= r < ctx.occ.shape[0] and 0 <= c < ctx.occ.shape[1]):
            end = None
            break
        lat = ctx.to_xy(np.full(5, s), np.linspace(-0.8, 0.8, 5))
        occ = ctx.lookup(ctx.fields()[0], lat, 1e3).min() < ctx.res
        if occ:
            end, free_run = s, 0.0
        elif end is not None:
            free_run += 0.5
            if free_run >= ctx.ego_len:
                break
        s += 0.5
    obs_end = (end if end is not None else s_block + assumed_len)
    s_clear = obs_end + ctx.ego_len / 2 + 1.0
    return {'target_d': W, 's_block': s_block, 's_clear': s_clear, 's_land_min': s_clear + 4.0,
            's_land_max': s_clear + land_window}
