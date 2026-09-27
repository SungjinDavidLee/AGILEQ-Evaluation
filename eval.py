import os
import argparse
import pandas as pd
import numpy as np
import time
from config import CONFIGS

parser = argparse.ArgumentParser(description="Eval a CARLA agent")
parser.add_argument("--host", default="localhost", type=str, help="IP of the host server (default: 127.0.0.1)")
parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
parser.add_argument("--model", type=str, default="", required=True, help="Path to a model evaluate")
parser.add_argument("--no_render", action="store_false", help="If True, render the environment")
parser.add_argument("--fps", type=int, default=15, help="FPS to render the environment")
parser.add_argument("--no_record_video", action="store_false", help="If True, record video of the evaluation")
parser.add_argument("--config", type=str, required=True, default="TQC", help="Config to use (default: 1)")
parser.add_argument("--town", type=str, default="town01", help="town")

parser.add_argument("--iteration", type=int, default=1, help="eval iteration")
parser.add_argument("--start", type=str, default=None,
                    help="Start point: spawn_point index (e.g. 175) or coordinates (e.g. 15,2.55,0.3 / 15,2.55,0.3,90)")
parser.add_argument("--dest", type=str, default=None,
                    help="Destination: spawn_point index (e.g. 175) or coordinates (e.g. 15,2.55,0.3)")

args = vars(parser.parse_args())


def parse_endpoint(text):

    if text is None:
        return None
    parts = [p for p in text.replace("(", "").replace(")", "").split(",") if p.strip() != ""]
    if len(parts) == 1:
        return int(float(parts[0]))
    return tuple(float(p) for p in parts)


eval_route = None
if args["start"] is not None or args["dest"] is not None:
    if args["start"] is None or args["dest"] is None:
        parser.error("--start and --dest must be given together.")
    eval_route = (parse_endpoint(args["start"]), parse_endpoint(args["dest"]))
    print("Custom eval route:", eval_route)
CONFIG = CONFIGS[args["config"]]

from sb3_contrib import TQC, CrossQ
from stable_baselines3 import PPO, DDPG, SAC, TD3
from crossq_pp.crossq_pp import CrossQpp

from utils import VideoRecorder, parse_wrapper_class
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards import reward_functions
from carla_env.wrappers import vector, get_displacement_vector
from carla_env.envs.carla_FixedRoute_env import CarlaRouteEnv
from eval_plots import plot_eval, summary_eval


def run_eval(env, model, model_path=None, record_video=False):
    total_reward = []

    model_name = os.path.basename(model_path)
    log_path = os.path.join(os.path.dirname(model_path), f"eval_{args['town']}")
    os.makedirs(log_path, exist_ok=True)
    video_path = os.path.join(log_path, model_name.replace(".zip", "_eval.avi"))
    csv_path = os.path.join(log_path, model_name.replace(".zip", "_eval.csv"))
    model_id = f"{model_path.split('/')[-2]}-{model_name.split('_')[-2]}"

    state = env.reset()
    rendered_frame = env.lender_image

    columns = [
        "model_id", "episode", "step", "throttle", "steer", "brake",
        "vehicle_location_x", "vehicle_location_y", "vehicle_location_z",
        "vehicle_velocity_x", "vehicle_velocity_y", "vehicle_velocity_z",
        "vehicle_rotation_pitch", "vehicle_rotation_yaw", "vehicle_rotation_roll",
        "target_waypoint_x", "target_waypoint_y", "target_waypoint_z",
        "reward", "distance", "speed", "center_dev", "angle_next_waypoint",
        "waypoint_x", "waypoint_y",
        "route_x", "route_y",
        "route_completion", "infraction_penalty", "mean_driving_speed",
        "timestamp"
    ]
    df = pd.DataFrame(columns=columns)

    if record_video:
        print("Recording video to {} ({}x{}x{}@{}fps)".format(video_path, *rendered_frame.shape, int(env.fps)))
        video_recorder = VideoRecorder(video_path, frame_size=rendered_frame.shape, fps=env.fps)
        video_recorder.add_frame(rendered_frame)
    else:
        video_recorder = None

    episode_idx = 0
    print(f"Starting evaluation for {model_id}...")


    try:
        while episode_idx < 21:
            print(f"--- Episode {episode_idx + 1} ---")
            saved_route = False
            initial_vehicle_location = vector(env.vehicle.get_location())
            initial_heading = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
            current_time = 0.0

            done = False
            while not done:
                action, _ = model.predict(state, deterministic=True)
                state, reward, done, info = env.step(action)
                total_reward.append(reward)
                current_time += 1.0 / env.fps

                if not saved_route and hasattr(env, 'route_waypoints'):
                    for way in env.route_waypoints:
                        route_relative = get_displacement_vector(initial_vehicle_location, vector(way[0].transform.location), initial_heading)
                        new_row = pd.DataFrame([['route', episode_idx, route_relative[0], route_relative[1]]], columns=["model_id", "episode", "route_x", "route_y"])
                        df = pd.concat([df, new_row], ignore_index=True)
                    saved_route = True

                vehicle_relative = get_displacement_vector(initial_vehicle_location, vector(env.vehicle.get_location()), initial_heading)
                waypoint_relative = get_displacement_vector(initial_vehicle_location, vector(env.current_waypoint.transform.location), initial_heading)

                control = env.vehicle.get_control()
                vehicle_transform = env.vehicle.get_transform()
                vehicle_location = vehicle_transform.location
                vehicle_velocity = env.vehicle.get_velocity()
                vehicle_rotation = vehicle_transform.rotation

                target_wp_location = env.next_waypoint.transform.location if hasattr(env, 'next_waypoint') and env.next_waypoint else None
                new_row_data = {
                    "model_id": model_id, "episode": episode_idx, "step": env.step_count,
                    "timestamp": current_time,
                    "throttle": control.throttle, "steer": control.steer, "brake": control.brake,
                    "vehicle_location_x": vehicle_location.x,
                    "vehicle_location_y": vehicle_location.y,
                    "vehicle_location_z": vehicle_location.z,
                    "vehicle_velocity_x": vehicle_velocity.x,
                    "vehicle_velocity_y": vehicle_velocity.y,
                    "vehicle_velocity_z": vehicle_velocity.z,
                    "vehicle_rotation_pitch": vehicle_rotation.pitch,
                    "vehicle_rotation_yaw": vehicle_rotation.yaw,
                    "vehicle_rotation_roll": vehicle_rotation.roll,
                    "target_waypoint_x": target_wp_location.x if target_wp_location else np.nan,
                    "target_waypoint_y": target_wp_location.y if target_wp_location else np.nan,
                    "target_waypoint_z": target_wp_location.z if target_wp_location else np.nan,
                    "reward": reward, "distance": info.get('total_distance', 0), "speed": env.speed,
                    "center_dev": env.distance_from_center, "angle_next_waypoint": np.rad2deg(env.angle),
                    "waypoint_x": waypoint_relative[0], "waypoint_y": waypoint_relative[1],
                    "route_completion": info.get('route_completion', 0.0),
                    "infraction_penalty": info.get('infraction_penalty', 1.0),
                    "mean_driving_speed": info.get('mean_driving_speed', 0.0)
                }
                df = pd.concat([df, pd.DataFrame([new_row_data]).drop(columns=['route_x', 'route_y'], errors='ignore')], ignore_index=True)

                if record_video:
                    rendered_frame = env.lender_image
                    video_recorder.add_frame(rendered_frame)


            df.to_csv(csv_path, index=False)

            episode_idx += 1
            if episode_idx < 21:
                state = env.reset()

    except Exception as e:
        print(f"\n[Warning] An error occurred during run_eval: {e}")
        print("Saving the data collected so far.")
        import traceback
        traceback.print_exc()

    finally:

        if record_video:
            video_recorder.release()


        df.to_csv(csv_path, index=False)
        print(f"--- CSV saved: {csv_path} ---")

        try:
            summary_eval(csv_path, town=args["town"])
        except Exception as summary_err:
            print(f"Failed to create summary report: {summary_err}")

        env.save_infraction_history(model_name=CONFIG["algorithm"], town_name=args["town"], iteration=args["iteration"])
        print("Mean_reward  : ", np.mean(total_reward))


if __name__ == "__main__":
    model_path = args["model"]

    algorithm_dict = {"PPO": PPO, "DDPG": DDPG, "SAC": SAC,
                    "TD3": TD3, "TQC": TQC, "crossq": CrossQ,
                    "crossq_pp" : CrossQpp}


    if CONFIG["algorithm"] not in algorithm_dict:
        raise ValueError("Invalid algorithm name")

    AlgorithmRL = algorithm_dict[CONFIG["algorithm"]]

    if CONFIG["algorithm"] not in algorithm_dict:
        raise ValueError("Invalid algorithm name")

    observation_space, encode_state_fn = create_encode_state_fn(CONFIG["state"])

    env = CarlaRouteEnv(host=args["host"], port=args["port"],
                        reward_fn=reward_functions[CONFIG["reward_fn"]],
                        observation_space=observation_space,
                        encode_state_fn=encode_state_fn,
                        fps=args["fps"], action_smoothing=CONFIG["action_smoothing"],
                        action_space_type='continuous', activate_spectator=False, eval=True,
                        activate_render=args["no_render"], eval_town=args["town"],
                        eval_route=eval_route)


    for wrapper_class_str in CONFIG["wrappers"]:

        wrap_class, wrap_params = parse_wrapper_class(wrapper_class_str)
        env = wrap_class(env, *wrap_params)


    model = AlgorithmRL.load(model_path, env=env, device='cuda')

    try:
        run_eval(env, model, model_path, record_video=args['no_record_video'])
        env.save_terminal_stats(model=CONFIG["algorithm"], town=args["town"], train=False)

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing the CARLA environment.")
        env.close()
