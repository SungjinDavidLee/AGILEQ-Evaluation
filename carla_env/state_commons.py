import torch
import numpy as np
import gym
import random
import carla

MAP_MAX_SPEED = 40.0
MAX_JERK = 30.0
DEV_SCALE = 30.0

def create_encode_state_fn(measurements_to_include):

    measure_flags = ["steer" in measurements_to_include,
                     "throttle" in measurements_to_include,
                     "brake" in measurements_to_include,
                     "speed" in measurements_to_include,
                     "traffic_light_state" in measurements_to_include,
                     "waypoints" in measurements_to_include,
                     "bev_image" in measurements_to_include,
                     "route_completion" in measurements_to_include,
                     "limit_speed" in measurements_to_include,
                     "jerk" in measurements_to_include,
                     "previous_state" in measurements_to_include,
                     "speed_devation" in measurements_to_include,
                     "stop_line" in measurements_to_include,
                     "right_turn_indicator" in measurements_to_include,
                     ]

    include_jerk = measure_flags[9]
    include_previous_state = measure_flags[10]

    def create_observation_space():
        observation_space = {}
        low, high = [], []


        if "steer" in measurements_to_include: low.append(-1), high.append(1)
        if "throttle" in measurements_to_include: low.append(0), high.append(1)
        if "brake" in measurements_to_include: low.append(0), high.append(1)
        if "speed" in measurements_to_include: low.append(0), high.append(1)
        if "jerk" in measurements_to_include: low.append(-1.0), high.append(1.0)
        if "speed_deviation" in measurements_to_include: low.append(-1.0), high.append(1.0)
        if "right_turn_indicator" in measurements_to_include:
            low.extend([0, 0])
            high.extend([1, 1])


        if "traffic_light_state" in measurements_to_include:
            low.extend([0, 0, 0, 0])
            high.extend([1, 1, 1, 1])

        if "limit_speed" in measurements_to_include: low.append(0), high.append(1)
        if "stop_line" in measurements_to_include: low.append(0), high.append(1)


        if "waypoints" in measurements_to_include:

            for _ in range(7 * 2):
                low.append(-1.0)
                high.append(1.0)


        if "previous_state" in measurements_to_include:

            low.append(-1.0), high.append(1.0)
            low.extend([0.0, 0.0])
            high.extend([1.0, 1.0])
            low.append(0.0), high.append(1.0)

            if "jerk" in measurements_to_include:
                low.append(-1.0), high.append(1.0)

            if "speed_deviation" in measurements_to_include:
                low.append(-1.0), high.append(1.0)


            low.extend([0, 0, 0, 0])
            high.extend([1, 1, 1, 1])

            if "limit_speed" in measurements_to_include: low.append(0.0), high.append(1.0)
            if "stop_line" in measurements_to_include: low.append(0.0), high.append(1.0)


        if "route_completion" in measurements_to_include: low.append(0), high.append(1)

        observation_space['vehicle_measures'] = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)


        if "bev_image" in measurements_to_include:
            observation_space['bev_image'] = gym.spaces.Box(low=0, high=1, shape=(7, 196, 196), dtype=np.float32)

        return gym.spaces.Dict(observation_space)


    def encode_state(env):
        encoded_state = {}
        vehicle_measures = []

        road_limit_kmh = env.get_dynamic_target_speed()
        if road_limit_kmh is None or road_limit_kmh == 0:
            road_limit_kmh = 30.0

        tl_state = carla.TrafficLightState.Unknown
        if env.traffic_light_handler:
            tl_state, _, _ = env.traffic_light_handler.get_light_state(env.vehicle)

        target_speed_kmh = road_limit_kmh


        if tl_state == carla.TrafficLightState.Yellow:
            target_speed_kmh = road_limit_kmh * 0.5
        elif tl_state == carla.TrafficLightState.Red:
            target_speed_kmh = 0.0


        if "steer" in measurements_to_include: vehicle_measures.append(env.control.steer)
        if "throttle" in measurements_to_include: vehicle_measures.append(env.control.throttle)
        if "brake" in measurements_to_include: vehicle_measures.append(env.control.brake)

        if "speed" in measurements_to_include:
            norm_speed = np.clip(env.speed / MAP_MAX_SPEED , 0.0, 1.0)
            vehicle_measures.append(norm_speed)

        if "jerk" in measurements_to_include:
            clipped_jerk = np.clip(env.current_jerk, -MAX_JERK, MAX_JERK)
            norm_jerk = clipped_jerk / MAX_JERK
            vehicle_measures.append(norm_jerk)

        if "speed_deviation" in measurements_to_include:


            speed_deviation = (env.speed - target_speed_kmh) / DEV_SCALE
            speed_deviation = np.clip(speed_deviation, -1.0, 1.0)
            vehicle_measures.append(speed_deviation)

        if "right_turn_indicator" in measurements_to_include:
            vehicle_measures.extend([1, 0] if env.right_turn_indicator else [0, 1])


        if "traffic_light_state" in measurements_to_include:
            tl_vector = (
                        env.final_vector
                        if env.final_vector is not None
                        else [0, 0, 0, 1]
                        )
            vehicle_measures.extend(tl_vector)

        if "limit_speed" in measurements_to_include:
            norm_limit = np.clip(road_limit_kmh / MAP_MAX_SPEED, 0.0, 1.0)
            vehicle_measures.append(norm_limit)

        if "stop_line" in measurements_to_include:
            stop_dist = env.dist_to_stop if hasattr(env, 'dist_to_stop') else 1.0
            vehicle_measures.append(np.clip(stop_dist, 0.0, 1.0))


        if "waypoints" in measurements_to_include:

            wp_vector = env.predicted_waypoints_norm if hasattr(env, 'predicted_waypoints_norm') and env.predicted_waypoints_norm is not None else [0.0] * 14
            vehicle_measures.extend(wp_vector)


        if "previous_state" in measurements_to_include:

            prev_road_limit = getattr(env, 'prev_road_limit', MAP_MAX_SPEED)
            if prev_road_limit == 0: prev_road_limit = 30.0

            prev_target_kmh = prev_road_limit


            if env.prev_traffic_light_state == carla.TrafficLightState.Red:
                prev_target_kmh = 0.0


            if env.prev_traffic_light_state == carla.TrafficLightState.Yellow:
                prev_target_kmh *= 0.5
            elif env.prev_traffic_light_state == carla.TrafficLightState.Red:
                prev_target_kmh = 0.0


            vehicle_measures.append(env.prev_steer)
            vehicle_measures.append(env.prev_throttle)
            vehicle_measures.append(env.prev_brake)
            norm_prev_speed = np.clip(env.prev_speed / MAP_MAX_SPEED, 0.0, 1.0)
            vehicle_measures.append(norm_prev_speed)

            if "jerk" in measurements_to_include:
                clipped_prev_jerk = np.clip(env.prev_jerk, -MAX_JERK, MAX_JERK)
                norm_prev_jerk = clipped_prev_jerk / MAX_JERK
                vehicle_measures.append(norm_prev_jerk)


            if "speed_deviation" in measurements_to_include:
                prev_deviation = (env.prev_speed - prev_target_kmh) / DEV_SCALE
                vehicle_measures.append(np.clip(prev_deviation, -1.0, 1.0))


            if "traffic_light_state" in measurements_to_include:
                tl_vector = env.final_vector if hasattr(env, 'final_vector') else [0, 0, 0, 1]
                vehicle_measures.extend(tl_vector)


            if "limit_speed" in measurements_to_include:
                vehicle_measures.append(np.clip(prev_road_limit / MAP_MAX_SPEED, 0.0, 1.0))

            if "stop_line" in measurements_to_include:
                vehicle_measures.append(np.clip(env.prev_dist_to_stop, 0.0, 1.0))

        if "route_completion" in measurements_to_include:
            completion_ratio = np.clip(env.distance_traveled_wp / env.route_length, 0.0, 1.0) if env.route_length > 0 else 0.0
            vehicle_measures.append(completion_ratio)

        vehicle_measures = np.nan_to_num(vehicle_measures, nan=0.0, posinf=1.0, neginf=-1.0).tolist()
        encoded_state['vehicle_measures'] = np.array(vehicle_measures, dtype=np.float32)


        if "bev_image" in measurements_to_include:
            encoded_state['bev_image'] = env.bev_image.cpu().squeeze(0).numpy()[:, 2:198, 2:198]


        return encoded_state

    return create_observation_space(), encode_state
