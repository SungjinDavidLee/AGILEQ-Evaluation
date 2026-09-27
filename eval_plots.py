import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

def eucldist(x1, y1, x2, y2):

    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def plot_eval(eval_csv_paths, output_name=None):
    episode_numbers = pd.read_csv(eval_csv_paths[0])['episode'].unique()

    cols = ['Steer', 'Throttle', 'Speed (km/h)', 'Reward', 'Center Deviation (m)', 'Distance (m)',
            'Angle next waypoint (grad)', 'Trayectory']


    fig, axs = plt.subplots(len(episode_numbers), len(cols), figsize=(4 * len(cols), 3 * len(episode_numbers)))

    if len(eval_csv_paths) == 1:
        eval_plot_path = eval_csv_paths[0].replace(".csv", ".png")
    else:
        os.makedirs('tensorboard/eval_plots', exist_ok=True)
        eval_plot_path = f'./tensorboard/eval_plots/{output_name}'

    models = ['Waypoints']

    for e, path in enumerate(eval_csv_paths):
        df = pd.read_csv(path)
        model_id = df.loc[df['model_id'] != 'route', 'model_id'].unique()[0]
        models.append(model_id)
        for i, episode_number in enumerate(episode_numbers):

            episode_df = df[(df['episode'] == episode_number) & (df['model_id'] != 'route')]
            route_df = df[(df['episode'] == episode_number) & (df['model_id'] == 'route')]


            axs[i, 0].plot(episode_df['step'], episode_df['steer'], label=model_id)
            axs[i, 0].set_xlabel('Step')
            axs[i, 0].set_ylim(-1, 1)


            axs[i][1].plot(episode_df['step'], episode_df['throttle'], label=model_id)
            axs[i][1].set_xlabel('Step')
            axs[i, 1].set_ylim(0, 1)

            axs[i][2].plot(episode_df['step'], episode_df['speed'], label=model_id)
            axs[i][2].set_xlabel('Step')
            axs[i, 2].set_ylim(0, 40)


            axs[i][3].plot(episode_df['step'], episode_df['reward'], label=model_id)
            axs[i][3].set_xlabel('Step')
            axs[i, 3].set_ylim(-0.2, 1)

            axs[i][4].plot(episode_df['step'], episode_df['center_dev'], label=model_id)
            axs[i][4].set_xlabel('Step')
            axs[i, 4].set_ylim(0, 3)

            axs[i][5].plot(episode_df['step'], episode_df['distance'], label=model_id)
            axs[i][5].set_xlabel('Step')

            axs[i][6].plot(episode_df['step'], episode_df['angle_next_waypoint'], label=model_id)
            axs[i][6].set_xlabel('Step')

            if e == 0:
                axs[i][7].plot(route_df['route_x'].head(1), route_df['route_y'].head(1), 'go',
                               label='Start')
                axs[i][7].plot(route_df['route_x'].tail(1), route_df['route_y'].tail(1), 'ro',
                               label='End')
                axs[i][7].plot(route_df['route_x'], route_df['route_y'], label='Waypoints', color="green")

                axs[i, 7].set_xlim(left=min(-5, min(route_df['route_x'] - 3)))
                axs[i, 7].set_xlim(right=max(5, max(route_df['route_x'] + 3)))
            axs[i][7].plot(episode_df['vehicle_location_x'], episode_df['vehicle_location_y'], label=model_id)


    pad = 5
    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, row in zip(axs[:, 0], episode_numbers):
        ax.annotate(f"Episode {row}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    handles, labels = axs[0][7].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0, 0.1 + 0.02 * len(labels), 1, 1))

    plt.savefig(eval_plot_path)


def summary_eval(eval_csv_path, town):
    df_raw = pd.read_csv(eval_csv_path)

    df_route = df_raw[df_raw['model_id'] == 'route']
    df = df_raw[df_raw['model_id'] != 'route'].copy()

    summary_list = []

    for episode in df['episode'].unique():
        episode_df = df[df['episode'] == episode].copy()

        if episode_df.empty:
            continue


        final_route_completion = episode_df['route_completion'].iloc[-1]

        final_infraction_penalty = episode_df['infraction_penalty'].iloc[-1]


        driving_score = final_route_completion * final_infraction_penalty * 100

        mean_driving_speed = episode_df['mean_driving_speed'].iloc[-1]

        total_distance = episode_df['distance'].iloc[-1]
        total_reward = episode_df['reward'].sum()
        original_speed_mean = episode_df['speed'].mean()
        center_dev_mean = episode_df['center_dev'].mean()


        heading_angle_std = calculate_heading_angle_std(episode_df)
        acceleration_std, rms_jerk = calculate_longitudinal_comfort(episode_df)

        if not episode_df.empty:

            last_vehicle_pos = episode_df[['vehicle_location_x', 'vehicle_location_y']].iloc[-1]
        else:
            last_vehicle_pos = pd.Series({'vehicle_location_x' : np.nan, 'vehicle_location_y': np.nan})

        route_for_episode = df_route[df_route['episode'] == episode]

        if not route_for_episode.empty:
            last_waypoint_pos = route_for_episode[['route_x', 'route_y']].iloc[-1]
            success = eucldist(last_vehicle_pos['vehicle_location_x'], last_vehicle_pos['vehicle_location_y'],
                           last_waypoint_pos['route_x'], last_waypoint_pos['route_y']) < 5 if not last_vehicle_pos.isnull().any() else False
        else:
            success = False

        summary_list.append({
            'episode': str(episode),
            'heading_angle_std': heading_angle_std,
            'acceleration_std' : acceleration_std,
            'rms_jerk' : rms_jerk,
            'driving_score': driving_score,
            'route_completion': final_route_completion,
            'infraction_penalty': final_infraction_penalty,
            'mean_driving_speed': mean_driving_speed,
            'original_mean_speed': original_speed_mean,
            'center_dev_mean': center_dev_mean,
            'total_distance': total_distance,
            'total_reward': total_reward,
            'success': success
        })

    df_summary = pd.DataFrame(summary_list)


    if not df_summary.empty:
        numeric_cols = ['heading_angle_std', 'acceleration_std', 'rms_jerk',
                        'driving_score', 'route_completion', 'infraction_penalty',
                        'mean_driving_speed', 'original_mean_speed', 'center_dev_mean',
                        'success']
        sum_cols = ['total_distance', 'total_reward']


        numeric_means = df_summary[numeric_cols].apply(pd.to_numeric, errors='coerce').mean()
        sum_totals = df_summary[sum_cols].apply(pd.to_numeric, errors='coerce').sum()

        total_row = {
            'episode': 'total (avg)',
            **numeric_means.to_dict(),
            **sum_totals.to_dict()
        }
        df_summary = pd.concat([df_summary, pd.DataFrame([total_row])], ignore_index=True)

    output_path = eval_csv_path.replace("eval.csv", f"eval_summary_{town}.csv")
    df_summary.to_csv(output_path, index=False)
    print(f"Saving summary to {output_path}")


def calculate_heading_angle_std(episode_df):

    heading_angles_diff = []
    for index, row in episode_df.iterrows():
        if pd.isna(row['target_waypoint_x']) or pd.isna(row['vehicle_location_x']) or pd.isna(row['vehicle_rotation_yaw']):
            heading_angles_diff.append(0.0)
            continue

        vehicle_loc = np.array([row['vehicle_location_x'], row['vehicle_location_y']])
        vehicle_yaw_rad = np.deg2rad(row['vehicle_rotation_yaw'])
        vehicle_fwd_np = np.array([np.cos(vehicle_yaw_rad), np.sin(vehicle_yaw_rad)])

        target_loc = np.array([row['target_waypoint_x'], row['target_waypoint_y']])
        vec_to_target_np = target_loc - vehicle_loc

        norm_fwd = np.linalg.norm(vehicle_fwd_np)
        norm_target = np.linalg.norm(vec_to_target_np)

        if norm_fwd > 1e-4 and norm_target > 1e-4:
            unit_fwd = vehicle_fwd_np / norm_fwd
            unit_target = vec_to_target_np / norm_target
            dot_product = np.dot(unit_fwd, unit_target)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_diff_rad = np.arccos(dot_product)
            cross_z = unit_fwd[0] * unit_target[1] - unit_fwd[1] * unit_target[0]
            if cross_z < 0:
                angle_diff_rad = -angle_diff_rad
            heading_angles_diff.append(np.degrees(angle_diff_rad))
        else:
            heading_angles_diff.append(0.0)

    return np.nanstd(heading_angles_diff) if heading_angles_diff else 0.0

def calculate_longitudinal_comfort(episode_df):

    speeds_mps = []
    timestamps = episode_df['timestamp'].values
    velocities_x = episode_df['vehicle_velocity_x'].values
    velocities_y = episode_df['vehicle_velocity_y'].values
    velocities_z = episode_df['vehicle_velocity_z'].values


    for i in range(len(timestamps)):
        speed = math.sqrt(velocities_x[i]**2 + velocities_y[i]**2 + velocities_z[i]**2)
        speeds_mps.append(speed)

    accelerations = []
    jerks = []

    if len(speeds_mps) > 1:
        for i in range(1, len(speeds_mps)):
            time_diff = timestamps[i] - timestamps[i-1]
            if time_diff > 1e-6:
                accel = (speeds_mps[i] - speeds_mps[i-1]) / time_diff
                accelerations.append(accel)
            else:
                accelerations.append(0.0)

    if len(accelerations) > 1:
        for i in range(1, len(accelerations)):

            if i + 1 < len(timestamps):
                 time_diff = timestamps[i+1] - timestamps[i]
                 if time_diff > 1e-6:
                    jerk = (accelerations[i] - accelerations[i-1]) / time_diff
                    jerks.append(jerk)
                 else:
                     jerks.append(0.0)


    acceleration_std = np.nanstd(accelerations) if accelerations else 0.0
    rms_jerk = np.sqrt(np.nanmean(np.square(jerks))) if jerks else 0.0

    return acceleration_std, rms_jerk
