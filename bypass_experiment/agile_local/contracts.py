from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


class ContractError(ValueError):
    """Reject ambiguous, stale, or incompatible integration inputs."""


def finite_array(value, shape=None):
    a = np.asarray(value, dtype=float)
    if not np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ContractError(f"Expected finite {shape}, got {a.shape}")
    return a


@dataclass(frozen=True)
class Stamp:
    episode: str
    frame: int
    time_s: float

    def __post_init__(self):
        if not self.episode or self.frame < 0 or not np.isfinite(self.time_s):
            raise ContractError("Invalid episode/frame/simulation timestamp")


@dataclass
class Ego:
    stamp: Stamp
    world_from_carla_ego: np.ndarray  # 4x4 CARLA transform captured at stamp
    velocity_flu: np.ndarray         # [forward, left] m/s, at same origin/time
    acceleration_flu: np.ndarray    # [forward, left] m/s², gravity handled upstream
    yaw_rate_rps: float = 0.0
    length_m: float = 4.2
    width_m: float = 1.8
    bbox_center_flu: tuple = (0.0, 0.0)  # actor origin is not always box center

    def __post_init__(self):
        self.world_from_carla_ego = finite_array(self.world_from_carla_ego, (4, 4))
        self.velocity_flu = finite_array(self.velocity_flu, (2,))
        self.acceleration_flu = finite_array(self.acceleration_flu, (2,))
        if not np.isfinite([self.length_m,self.width_m,self.yaw_rate_rps]).all() or self.length_m <= 0 or self.width_m <= 0:
            raise ContractError("Invalid vehicle dimensions")
        self.bbox_center_flu=tuple(finite_array(self.bbox_center_flu,(2,)))


@dataclass
class BEV:
    stamp: Stamp
    free: np.ndarray       # bool [H,W]: decoded known drivable space
    occupied: np.ndarray   # bool [H,W]: decoded obstacle occupancy
    forbidden: np.ndarray  # bool [H,W]: solid lines/restricted space, not lane centers
    x_min: float
    y_min: float
    dx: float
    dy: float
    frame_name: str = "ego_flu"
    # row r -> x_min+(r+.5)*dx ; column c -> y_min+(c+.5)*dy

    def __post_init__(self):
        arrays = [np.asarray(a) for a in (self.free, self.occupied, self.forbidden)]
        if any(a.dtype != np.bool_ for a in arrays):
            raise ContractError("BEV expects decoded boolean masks, not logits/features")
        if arrays[0].ndim != 2 or any(a.shape != arrays[0].shape for a in arrays):
            raise ContractError("BEV masks must have identical [H,W] shape")
        if not all(np.isfinite(v) for v in (self.x_min,self.y_min,self.dx,self.dy)) or min(self.dx,self.dy) <= 0:
            raise ContractError("Invalid grid geometry")
        if self.frame_name != "ego_flu":
            raise ContractError("Reproject BEV to ego_flu before planning")
        self.free,self.occupied,self.forbidden = [a.copy() for a in arrays]


@dataclass
class Prediction:
    actor_id: str
    times_s: np.ndarray     # relative to stamp, starts at 0
    xy: np.ndarray          # [T,2], fixed ego FLU frame at stamp
    yaw: np.ndarray         # [T], radians in fixed FLU frame
    length_m: float
    width_m: float
    margin_m: float = 0.25

    def __post_init__(self):
        self.times_s = finite_array(self.times_s)
        self.xy = finite_array(self.xy)
        self.yaw = finite_array(self.yaw)
        n = len(self.times_s)
        if n < 2 or self.times_s[0] != 0 or np.any(np.diff(self.times_s)<=0):
            raise ContractError("Prediction needs increasing times starting at zero")
        if self.xy.shape != (n,2) or self.yaw.shape != (n,):
            raise ContractError("Prediction dimensions do not match time")
        if not np.isfinite([self.length_m,self.width_m,self.margin_m]).all() or min(self.length_m,self.width_m) <= 0 or self.margin_m < 0:
            raise ContractError("Invalid prediction footprint")


@dataclass
class Traffic:
    stamp: Stamp
    state: str             # RED/YELLOW/GREEN/UNKNOWN/NONE
    applicable: bool       # already associated with ego movement/lane
    confidence: float
    stop_line_xy: np.ndarray | None = None   # 2 endpoints in ego_flu
    approach_unit: np.ndarray | None = None  # forward normal crossing the line
    source: str = "TLCFormer+association"

    def __post_init__(self):
        if not isinstance(self.applicable,(bool,np.bool_)):
            raise ContractError('Signal association must be an explicit boolean')
        if self.state not in {"RED","YELLOW","GREEN","UNKNOWN","NONE"}:
            raise ContractError("Unknown traffic label")
        if not 0 <= self.confidence <= 1:
            raise ContractError("Invalid traffic confidence")
        if self.applicable and self.state == "NONE":
            raise ContractError("Applicable signal cannot be NONE")
        if self.stop_line_xy is not None:
            self.stop_line_xy = finite_array(self.stop_line_xy,(2,2))
            self.approach_unit = finite_array(self.approach_unit,(2,))
            if not np.isclose(np.linalg.norm(self.approach_unit),1,atol=1e-4):
                raise ContractError("Stop-line approach normal must be a unit vector")

    @property
    def stop_required(self):
        return self.applicable and (self.state != "GREEN" or self.confidence < .8)


@dataclass
class Observation:
    ego: Ego
    bev: BEV
    traffic: Traffic
    predictions: list[Prediction]
    prediction_stamp: Stamp
    route_xy: np.ndarray  # dense forward route plus a short section behind ego
    target_speed_mps: float
    lateral_targets_m: tuple = (0.0,)  # explicit behavior/lane authorization

    def validate(self):
        s = self.ego.stamp
        for x in (self.bev.stamp,self.traffic.stamp,self.prediction_stamp):
            if x.episode != s.episode or x.frame != s.frame or abs(x.time_s-s.time_s)>1e-4:
                raise ContractError("Inputs must share episode, source frame, simulation time")
        self.route_xy = finite_array(self.route_xy)
        if self.route_xy.ndim!=2 or self.route_xy.shape[1]!=2 or len(self.route_xy)<3:
            raise ContractError("Dense route requires at least 3 [x,y] points")
        if not np.isfinite(self.target_speed_mps) or self.target_speed_mps < 0:
            raise ContractError("Invalid target speed")
        if not self.lateral_targets_m or not np.isfinite(self.lateral_targets_m).all():
            raise ContractError("Behavior must provide explicit lateral targets")
        if self.traffic.stop_required and self.traffic.stop_line_xy is None:
            raise ContractError("Applicable stop state without a localized stop line")


@dataclass
class PerceptionEnvelope:
    """Attach at inference INPUT capture, not when an asynchronous output arrives."""
    stamp: Stamp
    world_from_carla_ego: np.ndarray
    payload: object

    def __post_init__(self):
        self.world_from_carla_ego=finite_array(self.world_from_carla_ego,(4,4)).copy()

    def require_current(self,ego):
        if self.stamp.episode!=ego.stamp.episode or self.stamp.frame!=ego.stamp.frame or abs(self.stamp.time_s-ego.stamp.time_s)>1e-4:
            raise ContractError('Perception output belongs to a different source frame')
        if not np.allclose(self.world_from_carla_ego,ego.world_from_carla_ego,atol=1e-5,rtol=0):
            raise ContractError('Perception source ego pose differs from the snapshot')


@dataclass
class Plan:
    stamp: Stamp
    t: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    speed: np.ndarray
    acceleration: np.ndarray
    curvature: np.ndarray
    status: str             # FEASIBLE / STOP / EMERGENCY
    reason: str
    cost: float = float("inf")
    diagnostics: dict = field(default_factory=dict)

    @classmethod
    def emergency(cls, stamp, reason):
        # Empty path is deliberate: no fictitious zero-distance braking trajectory.
        return cls(stamp,np.empty(0),np.empty((0,2)),np.empty(0),np.empty(0),
                   np.empty(0),np.empty(0),"EMERGENCY",reason)
