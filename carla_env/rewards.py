import numpy as np
from config import reward_params
from carla_env.wrappers import vector

import carla


min_speed = reward_params["reward_params"]["min_speed"]
max_speed = reward_params["reward_params"]["max_speed"]
target_speed = reward_params["reward_params"]["target_speed"]
max_distance = reward_params["reward_params"]["max_distance"]
max_std_center_lane = reward_params["reward_params"]["max_std_center_lane"]
max_angle_center_lane = reward_params["reward_params"]["max_angle_center_lane"]
penalty_reward = reward_params["reward_params"]["penalty_reward"]
early_stop = reward_params["reward_params"]["early_stop"]
max_jerk = reward_params["reward_params"]["max_jerk"]
reward_functions = {}


W_SPEED = 0.6
W_LANE = 0.35
W_JERK = 0.05

SAFE_DISTANCE = 20.1
SAFE_TTC = 3.0
SPEED_DIFF_MAX = 10.0
STOP_DISTANCE = 20.1

low_speed_timer = 0
obstacle_safe_timer = 0

low_speed_timer = 0
obstacle_safe_timer = 0

def create_reward_fn(reward_fn):
    def func(env):
        terminal_reason = "Running..."
        if early_stop:

            global low_speed_timer, obstacle_safe_timer
            low_speed_timer += 1.0 / env.fps
            env.episode_timer += 1.0 / env.fps

            red_light_violation, distance_stop_line = env.run_red_light.tick(env.vehicle)
            traffic = env.traffic_light_handler.get_light_state(env.vehicle)
            npc_vehicle, npc_distance, npc_speed_ms = env._is_vehicle_hazard()
            closest_walker, distance_walker = env._get_closest_hazardous_walker_v2()
            green_target_speed = env.get_dynamic_target_speed() if env.get_dynamic_target_speed() is not None else target_speed
            red_light_stop = traffic[0] == carla.TrafficLightState.Red and not env.right_turn_indicator

            is_justified_to_stop = (red_light_stop or
                                    npc_vehicle or
                                    closest_walker or
                                    traffic[0] == carla.TrafficLightState.Yellow)

            if is_justified_to_stop:
                low_speed_timer = 0.0
            else:
                if env.speed < 1.0 and low_speed_timer > 10.0:
                    env.episode_timer += 1.0 / env.fps


            if env.speed < 1.0 and is_justified_to_stop:
                obstacle_safe_timer += 1.0 / env.fps
            else:
                obstacle_safe_timer = 0.0

            if obstacle_safe_timer >= 80.0:
                env.terminal_state = True
                env.safe_timeout = True
                terminal_reason = "Safe Timeout (Obstacle)"

            if env.collision_true:
                env.terminal_state = True
                terminal_reason = "collision_vehicle"
                env._record_infraction('collision_vehicle')

            if env.hit_true:
                env.terminal_state =True
                terminal_reason = "hitting_pedestrian"
                env._record_infraction('hitting_pedestrian')

            if env.collision_static:
                env.terminal_state = True
                terminal_reason = "collision_static"
                env._record_infraction('collision_static')

            if env.episode_timer > env.max_episode_time:
                env.terminal_state = True
                terminal_reason = "scenario_timeout"
                env._record_infraction('scenario_timeout')


        reward = 0
        reward += reward_fn(env,
                            npc_vehicle,
                            npc_distance,
                            traffic,
                            distance_stop_line,
                            closest_walker,
                            distance_walker,
                            green_target_speed,
                            npc_speed_ms)

        if env.terminal_state:
            if env.safe_timeout:
                pass
            elif env.collision_true:
                reward += (penalty_reward * 1.5)
            elif env.hit_true:
                reward += (penalty_reward * 2)
            else:
                reward += penalty_reward


            low_speed_timer = 0.0
            obstacle_safe_timer = 0.0
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)


        if terminal_reason in env.terminal_reason_counter:
            env.terminal_reason_counter[terminal_reason] += 1
        elif terminal_reason == "Safe Timeout (Obstacle)":
            env.terminal_reason_counter["Safe time obstacle"] += 1
        else:
            env.terminal_reason_counter["Other"] += 1

        if env.success_state:
            print(f"{env.episode_idx}| Success")
            env.terminal_state = True
            low_speed_timer = 0.0
            obstacle_safe_timer = 0.0
            if hasattr(env, "terminal_reason_counter"):
                env.terminal_reason_counter["Success"] += 1

        return reward, is_justified_to_stop
    return func


def reward_fn5(env,
               npc_vehicle,
               npc_distance,
               traffic,
               distance_stop_line,
               closest_walker,
               distance_walker,
               npc_speed_ms):

    speed_ms = env.speed / 3.6
    stop_penalty = 0.0
    red_light_stop = traffic[0] == carla.TrafficLightState.Red and not env.right_turn_indicator

    is_obstacle_present = (closest_walker or
                        npc_vehicle or
                        red_light_stop or
                        traffic[0] == carla.TrafficLightState.Yellow)


    if is_obstacle_present:
        constraint_speed_rewards = []


        if closest_walker and distance_walker is not None:

            ego_speed_ms = max(speed_ms, 1e-3)
            ttc = distance_walker / ego_speed_ms

            walker_reward = np.clip(ttc / SAFE_TTC, 0.0, 1.0)
            constraint_speed_rewards.append(walker_reward)


        if npc_vehicle and npc_distance is not None:


            norm_distance = np.clip(npc_distance / SAFE_DISTANCE, 0.0, 1.0)


            if npc_speed_ms is not None:
                speed_diff = abs(speed_ms - npc_speed_ms)
                norm_speed = np.clip(1.0 - speed_diff / SPEED_DIFF_MAX, 0.0, 1.0)
            else:
                norm_speed = 1.0

            w_distance = 0.4
            w_speed = 0.6

            vehicle_reward = (w_distance * norm_distance) + (w_speed * norm_speed)
            constraint_speed_rewards.append(vehicle_reward)


        if red_light_stop or traffic[0] == carla.TrafficLightState.Yellow:
            norm_distance = np.clip(1.0 - distance_stop_line / STOP_DISTANCE, 0.0, 1.0)
            norm_speed = np.clip(1.0 / (1.0 + speed_ms), 0.0, 1.0)

            w_distance = 0.5
            w_speed = 0.5

            red_reward = (w_distance * norm_distance) + (w_speed * norm_speed)
            constraint_speed_rewards.append(red_reward)

        if traffic[0] == carla.TrafficLightState.Yellow:
            yellow_reward = 1 - (env.speed / min_speed)
            constraint_speed_rewards.append(yellow_reward)

        speed_reward = min(constraint_speed_rewards)


    else:
        if env.speed < min_speed:
            speed_reward = env.speed / min_speed
        elif env.speed > target_speed:
            speed_reward = 1.0 - (env.speed - target_speed) / (max_speed - target_speed)
        else:
            speed_reward = 1.0


    jerk_penalty_factor = (env.current_jerk / max_jerk) **2
    jerk_penalty = min(jerk_penalty_factor, 1.0)
    jerk_reward = 1.0 - jerk_penalty


    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)


    angle_factor = max(1.0 - abs(env.angle / np.deg2rad(max_angle_center_lane)), 0.0)

    std = np.std(env.distance_from_center_history)
    distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)

    heading_angle_factor = max(1.0 - abs(env.heading_angle / np.deg2rad(max_angle_center_lane)), 0.0)

    lane_reward = centering_factor * angle_factor * distance_std_factor * heading_angle_factor


    reward = ((W_SPEED * speed_reward) +
              (W_LANE * lane_reward) +
              (W_JERK * jerk_reward))

    reward += stop_penalty


    return reward

reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)
