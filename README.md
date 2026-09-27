# AGILEQ-Evaluation

[한국어](README_ko.md) · [Training](https://github.com/SungjinDavidLee/AGILEQ-Training/tree/main/AGILEQ-train)

CARLA evaluation code from `AGILE-Q-eval`. It evaluates reinforcement-learning driving policies with BEVFormer perception from surround-view cameras and traffic-light perception from front-camera sequences. Code is located at the repository root.

## Contents

| Path | Purpose |
| --- | --- |
| `eval.py` | Load a policy, evaluate driving episodes, and save results. |
| `eval_plots.py` | Summarize evaluation CSV files and provide plotting helpers. |
| `run_experiments.py` | Evaluate selected checkpoints across towns. |
| `config.py`, `crossq_pp/` | Algorithm settings and CrossQ++ implementation. |
| `carla_env/envs/carla_FixedRoute_env.py` | Environment selected by `eval.py`. |
| `carla_env/envs/carla_RandomRoute_env_traffic.py` | Additional random-route environment implementation. |
| `carla_env/envs/perception_models/` | BEVFormer, traffic-light model, InternImage configuration, and extension sources. |
| `BEV/` | Map rendering utilities and map assets. |
| `environment_agileq_eval.yml` | Exported Conda environment. |

## Setup

Run commands from the repository root. Use Linux, CARLA 0.9.15 with the required towns, an NVIDIA GPU, and a CUDA toolkit/compiler compatible with PyTorch. The supplied environment uses Python 3.9 and PyTorch 2.7 with CUDA 12.8 packages; inference selects CUDA directly.

```bash
conda env create -n agileq_eval -f environment_agileq_eval.yml
conda activate agileq_eval
export CARLA_ROOT=/absolute/path/to/CARLA_0.9.15
```

The YAML contains platform-specific builds and CUDA wheel versions. Adjust package sources or versions for your machine if those builds are unavailable. `sb3_contrib` must expose `CrossQ` and `BatchRenorm1d`.

Build both extension packages inside the activated environment:

```bash
(cd carla_env/envs/perception_models/models/ops_dcnv3 && bash make.sh)
(cd carla_env/envs/perception_models/ops && bash make.sh)
```

The sources for DCNv3 and multi-scale deformable attention are included; compiled binaries are not. The InternImage configuration used by `get_config_t()` is bundled as `carla_env/envs/perception_models/internimage_t_1k_224.yaml`.

## Required checkpoints

| File | Purpose |
| --- | --- |
| `/absolute/path/to/run/model_100000_steps.zip` | Trained driving policy, supplied through `--model`. |
| `./best_bev_model.pth` | BEVFormer weights loaded by the evaluation environment. |
| `./best_traffic_model.pth` | Traffic-light model weights loaded by the evaluation environment. |

Weights are not included. Put the perception checkpoints in the repository root or update their loading paths in the environment. Checkpoint architectures and the policy observation space must match this code. EfficientNet initialization may also download pretrained backbone weights if they are not cached.

## Evaluate

```bash
python eval.py \
  --config crossq_pp \
  --model /absolute/path/to/run/model_100000_steps.zip \
  --town town01 \
  --port 2000 \
  --no_render \
  --no_record_video
```

Use the configuration matching the policy: `PPO`, `SAC`, `DDPG`, `TD3`, `TQC`, `crossq`, or `crossq_pp`. The environment starts CARLA using `CARLA_ROOT/CarlaUE4.sh`.

`--no_render` disables the environment display; `--no_record_video` disables AVI recording. To specify a route, supply both `--start` and `--dest`, using spawn-point indices or comma-separated coordinates, for example `--start 175 --dest 42`. The loop in `eval.py` currently runs 21 episodes. `--iteration` labels the infraction-history output; it does not change the episode count.

Town names must match `BEV/` directories exactly: `town01`, `town02`, `town03`, `town04`, `town05`, `town07`, and `Town10HD_Opt`. Corresponding CARLA maps must be installed.

For batch evaluation, edit `run_experiments.py`: set `CARLA_ROOT`, the checkpoint path, towns, and configurations before running it. The script terminates CARLA server processes between runs. The random-route implementation is included separately; the current entry point imports the fixed-route environment.

## Results

Per-step CSV files and optional AVI videos are saved beside the policy checkpoint under `eval_<town>/`. CSV data includes controls, vehicle pose, route points, reward, speed, center deviation, route completion, and infraction penalty. `summary_eval()` creates summary output, and the environment saves infraction history and terminal statistics.

Python comments and docstrings are removed, with third-party header notices preserved in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). Python syntax and preservation of executable statements were checked during import. CUDA compilation and CARLA evaluation have not been run as part of this repository update.
