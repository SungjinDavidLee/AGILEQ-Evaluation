import numpy as np

FLIP = np.diag([1., -1., 1., 1.])


class PlanError(Exception):
    pass


def quintic(p0, v0, a0, p1, v1, a1, T):
    if T <= 0:
        raise PlanError("Polynomial duration must be positive")
    c = np.array([p0, v0, a0 / 2, 0., 0., 0.])
    m = np.array([[T ** 3, T ** 4, T ** 5], [3 * T * T, 4 * T ** 3, 5 * T ** 4], [6 * T, 12 * T * T, 20 * T ** 3]])
    rhs = np.array([p1 - p0 - v0 * T - a0 * T * T / 2, v1 - v0 - a0 * T, a1 - a0])
    c[3:] = np.linalg.solve(m, rhs)
    return c


def peval(coeff, t, derivative=0):
    c = np.asarray(coeff)
    for _ in range(derivative):
        c = c[1:] * np.arange(1, len(c))
    return np.polynomial.polynomial.polyval(t, c)


def _transform(points, matrix):
    p = np.asarray(points, dtype=float)
    if p.shape[1] == 2:
        p = np.c_[p, np.zeros(len(p))]
    return (np.c_[p, np.ones(len(p))] @ matrix.T)[:, :3]


# CARLA world (left-handed) <-> ego FLU (x forward, y left, z up)
def world_to_flu(points, world_from_ego):
    return _transform(points, FLIP @ np.linalg.inv(world_from_ego))


def flu_to_world(points, world_from_ego):
    return _transform(points, world_from_ego @ FLIP)


def world_vector_to_flu(v, world_from_ego):
    return FLIP[:3, :3] @ world_from_ego[:3, :3].T @ np.asarray(v, dtype=float)


class ReferenceLine:
    def __init__(self, xy):
        a = np.asarray(xy, dtype=float)
        keep = np.r_[True, np.linalg.norm(np.diff(a, axis=0), axis=1) > 1e-6]
        self.xy = a[keep]
        if len(self.xy) < 3:
            raise PlanError("Degenerate reference line")
        self.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))]
        g = np.gradient(self.xy, self.s, axis=0)
        self.yaw = np.unwrap(np.arctan2(g[:, 1], g[:, 0]))
        self.k = np.gradient(self.yaw, self.s)

    def at(self, s):
        s = np.asarray(s, dtype=float)
        if np.any(s < self.s[0] - 1e-6) or np.any(s > self.s[-1] + 1e-6):
            raise PlanError("Reference line exhausted")
        xy = np.stack([np.interp(s, self.s, self.xy[:, i]) for i in range(2)], axis=-1)
        return xy, np.interp(s, self.s, self.yaw), np.interp(s, self.s, self.k)

    def project(self, p):
        p = np.asarray(p, dtype=float)

        def residual(s):
            q, yaw, _ = self.at(s)
            return float(np.dot(p - q, [np.cos(yaw), np.sin(yaw)]))

        roots = []
        for a, b in zip(self.s[:-1], self.s[1:]):
            fa, fb = residual(a), residual(b)
            if abs(fa) < 1e-9:
                roots.append(a)
            if fa * fb < 0:
                for _ in range(35):
                    mid = (a + b) / 2
                    fm = residual(mid)
                    if fa * fm <= 0:
                        b = mid
                    else:
                        a, fa = mid, fm
                roots.append((a + b) / 2)
        s = min(roots or [self.s[0], self.s[-1]], key=lambda z: np.linalg.norm(self.at(z)[0] - p))
        q, yaw, _ = self.at(s)
        return float(s), float(np.dot(p - q, [-np.sin(yaw), np.cos(yaw)]))
