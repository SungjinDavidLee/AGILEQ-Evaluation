# Bypass experiment

This folder reproduces the Frenet bypass demo with the policy AGILE-Q-C2 v1 (`ieee_results/AGILE-Q-C2/v1/last_model.zip`). That policy was not trained for bypassing.

To make the demo work, this folder uses its own copy of the bypass logic. It differs from the main module `carla_env/bypass/plan_bypass.py` as follows:
- It uses a ground-truth BEV and a ground-truth trigger (listed below).
- It overrides actions: a speed cap while bypassing, and an emergency brake.

The main module stays switched off while this demo runs. During a bypass, CrossQ follows waypoints sampled from a frozen Frenet bypass path.

Download and place the model files as described in the [model-weights setup](../README.md#model-weights): the perception `.pth` files go directly next to `eval.py` in the repository root. The demo's default policy path is `ieee_results/AGILE-Q-C2/v1/last_model.zip` relative to that root. Pass `--model` to `run_bypass_demo.py` to use another policy `.zip`.

## What differs from the paper setting
State these points in any figure or caption that uses this demo.
- **BEV input is ground truth with synthetic noise, not BEVFormer predictions.** All channels are rendered from the map and simulator actors (`perception_noise.py`):
  - vehicles: solid rounded rectangles at 0.85–1.0× length and 0.8–1.0× width, ±3° yaw
  - lanes and stop lines: broken, world-anchored dashes with varying thickness
  - drivable area: edge wobble
- TLCFormer predictions are used unchanged.
- **The trigger uses simulator vehicle information** (`_is_vehicle_hazard`). The in-bypass safety check also uses the ground-truth box of the NPC being passed.
- **The scenario and the policy input are fixed:**
  - One stationary NPC stands 14.5 m ahead in Town04 (road 22, rightmost of two same-direction lanes).
  - The episode is limited to 60 s, and to 20 s while stopped.
  - The policy is AGILE-Q-C2 v1, which was trained without `right_turn_indicator`. `c2_state` in the configs sets that 34-dim observation.
- **Actions are overridden while bypassing:** speed is capped at 6 km/h, and an aborted bypass brakes to a stop.

## Run
```bash
export CARLA_ROOT=/path/to/CARLA_0.9.15
export PYTHON=/path/to/env/bin/python   # environment from environment_agileq_eval.yml
./run.sh iter5_vis configs/iter5_vis.json   # this folder's logic, with red dots for the global route (video only)
./run.sh iter5     configs/iter5.json
./run.sh main      configs/main_module.json # main mask-only module
```
With `configs/main_module.json`, the main module drives the bypass:
- the trigger is the Algorithm-1 block
- the safety checks use the BEV mask only
- `ENABLED` and `OVERRIDE_ACTIONS` (6 km/h speed cap) are both set to `True`
- the settings of our successful run override the paper defaults of the module:
  - 15 route points for the blocked check (`trigger_points`; the main module uses the guidance waypoints Psi_T)
  - 1.5 m waypoint spacing on the bypass path (`sample_m`; paper 1 m)
- the module has no emergency stop (paper version), so results can differ from the run made with the earlier module version

This folder then only provides the scenario, the ground-truth BEV with noise, the episode limits and the logging.
Outputs:
- `logs/<run>/steps.csv`: per-step modes, triggers and waypoints
- `logs/<run>/episodes.json`: episode summary
- `videos/<run>/episode_1.mp4`

## Settings (iter5)
| Item | Value |
|---|---|
| Trigger | 45 consecutive steps (3 s): vehicle ahead, TLC not red, speed < 5 km/h |
| Entry gates | Left lane change legal over the whole maneuver (≤ 40 m); the swept ego footprint (+0.2 m) is free of the τ = 0.5 danger mask and of the NPC box; the landing footprint is free |
| Path | Quintic shift to the left-lane center over 8 m, hold until NPC front + ego/2 + 7 m, return over 8 m, 10 m tail; frozen once adopted |
| Following | T_wp and A_W are sampled every 1.5 m along the frozen path |
| While bypassing | Speed cap 6 km/h (brake 0.3); the steering rate limit is anchored on the last applied steer; hits within 2 m ahead are ignored; abort after 5 consecutive hazardous steps → emergency stop |

The result is not deterministic. With `iter5_vis`, 1 of 4 runs completed the bypass and return; the others stalled during the return leg. Treat the video as a qualitative example, not a success rate.
