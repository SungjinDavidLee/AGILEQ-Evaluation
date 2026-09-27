import carla
import numpy as np
import weakref
import math
import copy

def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[:truncate - 1] + u"\u2026") if len(name) > truncate else name


def get_displacement_vector(car_pos, waypoint_pos, theta):


    relative_pos = waypoint_pos - car_pos

    theta = theta

    R = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    T = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    waypoint_car = R @ relative_pos
    waypoint_car = T @ waypoint_car

    waypoint_car[np.abs(waypoint_car) < 10e-10] = 0

    return waypoint_car


def angle_diff(v0, v1):

    v0_xy = v0[:2]
    v1_xy = v1[:2]
    v0_xy_norm = np.linalg.norm(v0_xy)
    v1_xy_norm = np.linalg.norm(v1_xy)
    if v0_xy_norm == 0 or v1_xy_norm == 0:
        return 0

    v0_xy_u = v0_xy / v0_xy_norm
    v1_xy_u = v1_xy / v1_xy_norm
    dot_product = np.dot(v0_xy_u, v1_xy_u)
    angle = np.arccos(dot_product)


    cross_product = np.cross(v0_xy_u, v1_xy_u)
    if cross_product < 0:
        angle = -angle
    if abs(angle) >= 2.3:
        return 0
    return round(angle, 2)


def distance_to_line(A, B, p):
    p[2] = 0
    num = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom


def vector(v):

    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])


def smooth_action(old_value, new_value, smooth_factor):
    return old_value * smooth_factor + new_value * (1.0 - smooth_factor)


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):


    point = np.array([loc.x, loc.y, loc.z, 1])

    point_camera = np.dot(w2c, point)


    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]


    point_img = np.dot(K, point_camera)

    if point_img[2] <= 0:
        return None


    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2].astype(int)


class LidarSensor:
    def __init__(self, parent_actor,conf):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()

        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')

        bp.set_attribute('channels', '64')
        bp.set_attribute('range', '100')
        bp.set_attribute('points_per_second', '1200000')
        bp.set_attribute('rotation_frequency', '60')

        self.sensor = world.spawn_actor(bp, carla.Transform(
        carla.Location(x=conf['x'], y=conf['y'], z=conf['z']),
        carla.Rotation(pitch=conf['pitch'], roll=conf['roll'], yaw=conf['yaw'])
        ), attach_to=self._parent)


        weak_self = weakref.ref(self)
        self.sensor.listen(lambda point_cloud: LidarSensor._on_point_cloud(weak_self, point_cloud))
        self.point_cloud=None

    @staticmethod
    def _on_point_cloud(weak_self, lidar_data):
        self = weak_self()
        if not self:
            return
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))

        self.point_cloud = points[:, :3].copy() * 2.0

    def _destroy(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
        self.sensor = None


class GnssSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        blueprint = self._parent.get_world().get_blueprint_library().find('sensor.other.gnss')
        self.sensor = self._parent.get_world().spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                                           attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

    def _destroy(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
        self.sensor = None


class CollisionSensor:
    def __init__(self, parent_actor, call_back=None):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.call_back = call_back
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = {}
        for frame, intensity in self.history:
            history[frame] = history.get(frame, 0) + intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        if self.call_back:
            self.call_back(event)

        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

    def _destroy(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()
        self.sensor = None


class LaneInvasionSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bp = self._parent.get_world().get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = self._parent.get_world().spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        pass

    def _destroy(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()
        self.sensor = None


class CameraSensor_RGB:
    def __init__(self, parent_actor,conf):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()

        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(conf['width']))
        bp.set_attribute('image_size_y', str(conf['height']))
        bp.set_attribute('fov', str(conf['fov']))

        self.sensor = world.spawn_actor(bp, carla.Transform(
        carla.Location(x=conf['x'], y=conf['y'], z=conf['z']),
        carla.Rotation(pitch=conf['pitch'], roll=conf['roll'], yaw=conf['yaw'])
        ), attach_to=self._parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensor_RGB._on_image(weak_self, image))
        self.img=None


    @staticmethod
    def _on_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        try:
            array = np.reshape(np.frombuffer(image.raw_data, dtype=np.uint8), (image.height, image.width, 4))

            self.img = array[:, :, :3][:, :, ::-1].copy()
        except Exception as e:
            print(f"[CameraSensor_RGB] _on_image error: {e}")

    def _destroy(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
        self.sensor = None


class World():
    def __init__(self, client, town):
        self.world = client.load_world(town)
        self.map = self.get_map()
        self.actor_list = []

    def tick(self):
        for actor in list(self.actor_list):
            actor.tick()

        self.world.tick()


    def destroy(self):
        print("--- [DEBUG] Starting World.destroy() ---")
        print("Destroying all spawned actors")
        for actor in list(self.actor_list):
            actor.destroy()


    def get_carla_world(self):
        return self.world

    def __getattr__(self, name):

        return getattr(self.world, name)
