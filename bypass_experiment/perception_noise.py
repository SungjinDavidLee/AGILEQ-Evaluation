"""
Ground-truth BEV channels rendered with synthetic perception-like noise.
  PerceptionLikeRenderer: vehicles (preset 'D')
  StaticChannelNoise:     drivable area (edge wobble)
  LaneLikeNoise:          lane markings and stop lines (world-anchored dashes, per-dash thickness, slight bend)
Layout: 200x200, 0.2 m/px, row 0 = 20 m ahead, column -> right.
"""
import math
import numpy as np
import cv2

PRESETS = {
    # solid rounded rectangle, 0.85-1.0 x length, 0.8-1.0 x width, +-3 deg yaw, slow in time
    'D': dict(shape='rrect', amp=0.03, s_len=(0.85, 1.0), s_wid=(0.80, 1.0), depth_mu=0.0, depth_sd=0.15,
              sig_row=3.0, sig_col=3.0, rho=0.97, gain=8.0, bias=0.0, window_m=1.0, corner_frac=0.45, yaw_sd_deg=3.0),
}

_rows, _cols = np.mgrid[0:200, 0:200].astype(np.float32)
F_M = (100.0 - (_rows + 0.5)) * 0.2          # forward (m)
R_M = ((_cols + 0.5) - 100.0) * 0.2          # right (m)


class PerceptionLikeRenderer:
    def __init__(self, preset='D', rng=None, **over):
        self.p = dict(PRESETS[preset], **over)
        self.preset = preset
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.state = {}                        # vehicle id -> {'field', 's_len', 's_wid', 'depth'}

    def _noise_field(self):
        n = self.rng.standard_normal((200, 200)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigmaX=self.p['sig_col'], sigmaY=self.p['sig_row'])
        return n / (n.std() + 1e-6)

    def _hole_field(self):
        n = self.rng.standard_normal((200, 200)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigmaX=self.p['hole_sig'], sigmaY=self.p['hole_sig'])
        return n / (n.std() + 1e-6)

    def _ar(self, x, lo, hi, rho):
        """slow random walk inside [lo, hi] (AR(1) around the centre)."""
        c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
        z = (x - c) / h if h > 0 else 0.0
        z = rho * z + math.sqrt(1 - rho ** 2) * float(self.rng.standard_normal()) * 0.6
        return c + h * float(np.clip(z, -1.0, 1.0))

    def _vehicle_state(self, vid):
        p, st = self.p, self.state.get(vid)
        if st is None:
            st = {'field': self._noise_field(), 'hole': self._hole_field() if p.get('shape') == 'rect' else None,
                  's_len': float(self.rng.uniform(*p['s_len'])), 's_wid': float(self.rng.uniform(*p['s_wid'])),
                  'depth': float(self.rng.normal(p['depth_mu'], p['depth_sd']))}
        else:
            st['field'] = p['rho'] * st['field'] + math.sqrt(1 - p['rho'] ** 2) * self._noise_field()
            st['field'] /= (st['field'].std() + 1e-6)
            if st.get('hole') is not None:
                st['hole'] = p['rho'] * st['hole'] + math.sqrt(1 - p['rho'] ** 2) * self._hole_field()
                st['hole'] /= (st['hole'].std() + 1e-6)
            st['s_len'] = self._ar(st['s_len'], *p['s_len'], p['rho'])
            st['s_wid'] = self._ar(st['s_wid'], *p['s_wid'], p['rho'])
            st['depth'] = p['depth_mu'] + p['rho'] * (st['depth'] - p['depth_mu']) + \
                math.sqrt(1 - p['rho'] ** 2) * p['depth_sd'] * float(self.rng.standard_normal())
        self.state[vid] = st
        return st

    def render_logits(self, data_rad, vehicles, ids=None):
        """data_rad: {'x','y','yaw'(rad)} ego; vehicles: [[cx, cy, yaw_rad, length, width], ...] world (CARLA).
        Returns float32 200x200 logits (env layout); sigmoid(logit) > 0.5 is the binarised mask."""
        p = self.p
        out = np.full((200, 200), -20.0, np.float32)
        ex, ey, eyaw = data_rad['x'], data_rad['y'], data_rad['yaw']
        c, s = math.cos(eyaw), math.sin(eyaw)
        for k, v in enumerate(vehicles):
            vid = ids[k] if ids is not None else k
            st = self._vehicle_state(vid)
            dx, dy = v[0] - ex, v[1] - ey
            f0 = c * dx + s * dy + st['depth']                 # depth error along the ego forward axis
            r0 = -s * dx + c * dy
            if abs(f0) > 26 or abs(r0) > 26:
                continue
            if p.get('yaw_sd_deg'):
                st['dyaw'] = p['rho'] * st.get('dyaw', 0.0) + math.sqrt(1 - p['rho'] ** 2) * math.radians(p['yaw_sd_deg']) * float(self.rng.standard_normal())
            rel = v[2] - eyaw + st.get('dyaw', 0.0)
            a, b = 0.5 * v[3] * st['s_len'], 0.5 * v[4] * st['s_wid']
            du, dv = F_M - f0, R_M - r0
            cu, su = math.cos(rel), math.sin(rel)
            u = cu * du + su * dv                               # along the vehicle heading
            w = -su * du + cu * dv                              # across
            if p.get('shape') == 'rrect':
                r_c = p.get('corner_frac', 0.4) * b                       # corner radius
                qx, qy = np.abs(u) - (a - r_c), np.abs(w) - (b - r_c)
                dist = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - r_c
                logit = p['gain'] * (-dist + p['amp'] * st['field']) + p['bias']
            elif p.get('shape') == 'rect':
                g = p.get('grow_m', 0.0)
                sdf = np.minimum(a + g - np.abs(u), b + g - np.abs(w))  # box grown by grow_m, + inside (m)
                logit = p['gain'] * (sdf + p['amp'] * st['field']) + p['bias']
                # holes: top hole_frac of a smooth field, only in the interior (keeps the outline readable)
                thr = np.quantile(st['hole'], 1.0 - p['hole_frac'])
                hole = (st['hole'] > thr) & (sdf > p['hole_margin_m'])
                logit = np.where(hole, -8.0, logit)
            else:
                q = np.sqrt((u / a) ** 2 + (w / b) ** 2)
                sdf = (1.0 - q) * min(a, b)
                logit = p['gain'] * (sdf + p['amp'] * st['field']) + p['bias']
            win = (np.abs(du) < a + p['window_m']) & (np.abs(dv) < b + p['window_m'])
            out = np.where(win, np.maximum(out, logit), out)
        return np.clip(out, -20.0, 20.0)


class StaticChannelNoise:
    """GT static BEV channel (road / lane / stop line, 200x200 bool, env layout) -> noisy logits.
    Signed pixel distance to the mask boundary + a smooth, slowly varying field (edge wobble) + a few holes.
    Fields live in the ego BEV frame and follow AR(1) in time (rho), so edges do not flicker frame to frame."""

    def __init__(self, rng, amp_px=0.5, sig_px=2.0, hole_frac=0.05, hole_sig_px=2.0, rho=0.98, gain=4.0):
        self.rng, self.amp, self.sig, self.hf, self.hs, self.rho, self.gain = rng, amp_px, sig_px, hole_frac, hole_sig_px, rho, gain
        self.field = self._f(self.sig)
        self.hole = self._f(self.hs)

    def _f(self, sig):
        n = self.rng.standard_normal((200, 200)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigmaX=sig, sigmaY=sig)
        return n / (n.std() + 1e-6)

    def step(self, mask):
        r = self.rho
        self.field = r * self.field + math.sqrt(1 - r * r) * self._f(self.sig)
        self.field /= (self.field.std() + 1e-6)
        self.hole = r * self.hole + math.sqrt(1 - r * r) * self._f(self.hs)
        self.hole /= (self.hole.std() + 1e-6)
        m = mask.astype(np.uint8)
        if not m.any():
            return np.full((200, 200), -20.0, np.float32)
        din = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
        sdf = np.where(m > 0, din, -dout) - 0.5                     # px, + inside
        logit = self.gain * (sdf + self.amp * self.field)
        if self.hf > 0:
            thr = np.quantile(self.hole, 1.0 - self.hf)
            logit = np.where((self.hole > thr) & (m > 0), -8.0, logit)
        return np.clip(logit, -20.0, 20.0).astype(np.float32)


def _value_noise(X, Y, spacing, seed):
    """World-anchored smooth value noise in [0, 1): lattice hash + smoothstep bilinear interpolation."""
    gx, gy = X / spacing, Y / spacing
    ix, iy = np.floor(gx), np.floor(gy)
    fx, fy = gx - ix, gy - iy
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    def h(i, j):
        v = np.sin(i * 127.1 + j * 311.7 + seed * 74.7) * 43758.5453
        return v - np.floor(v)
    a, b, c, d = h(ix, iy), h(ix + 1, iy), h(ix, iy + 1), h(ix + 1, iy + 1)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


class LaneLikeNoise:
    """GT lane / stop-line mask (200x200 bool, env layout, ~2 px lines) -> perception-like broken markings:
    dashes of varying length (world-anchored gaps), per-dash thickness thin(1 px) / normal / thick, a slight bend,
    and a slow change of the gap edges over time. World anchoring keeps the gaps glued to the road."""

    def __init__(self, rng, gap_frac=0.35, gap_m=(1.2, 3.0), thin_frac=0.40, thick_frac=0.20, thick_m=3.0,
                 bend_px=1.0, bend_m=5.0, drift=0.04, rho=0.97, seed=0):
        self.rng, self.gf, self.gm, self.tf, self.kf, self.km = rng, gap_frac, gap_m, thin_frac, thick_frac, thick_m
        self.bp, self.bm, self.drift, self.rho, self.seed = bend_px, bend_m, drift, rho, seed
        self.q = 0.0
        r, c = np.mgrid[0:200, 0:200].astype(np.float32)
        self.F = (100.0 - (r + 0.5)) * 0.2
        self.R = ((c + 0.5) - 100.0) * 0.2

    def step(self, mask, ego):
        """ego = (x, y, yaw_rad) in CARLA world."""
        if not mask.any():
            return np.full((200, 200), -20.0, np.float32)
        ex, ey, yw = ego
        cs, sn = math.cos(yw), math.sin(yw)
        X = ex + self.F * cs - self.R * sn
        Y = ey + self.F * sn + self.R * cs
        s0 = self.seed
        # slight bend: displace the mask by a smooth world-anchored field
        dx = (_value_noise(X, Y, self.bm, s0 + 1) - 0.5) * 2 * self.bp
        dy = (_value_noise(X, Y, self.bm, s0 + 2) - 0.5) * 2 * self.bp
        gy, gx = np.mgrid[0:200, 0:200].astype(np.float32)
        m = cv2.remap(mask.astype(np.uint8), gx + dx.astype(np.float32), gy + dy.astype(np.float32),
                      cv2.INTER_NEAREST, borderValue=0) > 0
        # dashes: gaps where a two-octave world noise is below a (slowly drifting) quantile
        self.q = self.rho * self.q + math.sqrt(1 - self.rho ** 2) * self.drift * float(self.rng.standard_normal())
        g = 0.6 * _value_noise(X, Y, self.gm[0], s0 + 3) + 0.4 * _value_noise(X, Y, self.gm[1], s0 + 4)
        thr = np.quantile(g[m], np.clip(self.gf + self.q, 0.0, 0.9)) if m.any() else 1.0   # exact gap fraction
        keep = g > thr
        m = m & keep
        # per-dash thickness
        t = _value_noise(X, Y, self.km, s0 + 5)
        thick = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        thin = m & ~np.roll(m, 1, axis=1)                          # one-pixel edge of the 2 px line
        out = np.where(t > 1.0 - self.kf, thick & (cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0), m)
        out = np.where(t < self.tf, thin, out)
        return np.where(out, 20.0, -20.0).astype(np.float32)
