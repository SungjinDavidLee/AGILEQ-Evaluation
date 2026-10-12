# AGILE-Q 평가

[English](README.md) | [한국어](README_ko.md)

CARLA 0.9.15에서 학습된 AGILE-Q(CrossQ) 주행 정책을 평가하는 코드입니다. 정책에는 ground truth를 직접 입력하지 않습니다. BEV 입력은 BEVFormer 예측(`best_bev_model.pth`), 신호등 입력은 TLCFormer 예측(`best_traffic_model.pth`)을 사용합니다.

학습 코드: [AGILEQ-Training / AGILEQ-train](https://github.com/SungjinDavidLee/AGILEQ-Training/tree/main/AGILEQ-train).

## 실행 환경

- CARLA 0.9.15
- CUDA를 지원하는 NVIDIA GPU
- Conda

저장소 루트 폴더에서 다음 명령을 실행합니다.

```bash
conda env create -f environment_agileq_eval.yml
conda activate agileq_eval
export CARLA_ROOT=/path/to/CARLA_0.9.15
```

환경 코드가 `$CARLA_ROOT`의 CARLA를 직접 실행합니다.

> 아래 명령은 제공된 코드 기준의 사용 예시입니다. 이번 README 수정 과정에서 CARLA 실행과 평가는 검증하지 않았습니다.

## 모델 가중치 다운로드 및 배치

[Google Drive에서 모델 압축파일을 다운로드](https://drive.google.com/file/d/1nC7NmAk6vBsQcedpB-zmNv05dSJVbUGO/view?usp=sharing)한 뒤 압축을 풉니다.

`.pth` 파일은 저장소 루트의 **`eval.py` 바로 옆에** 넣습니다. 압축을 풀면서 상위 폴더가 하나 더 생겼다면 그 안의 파일을 꺼내 아래 위치에 배치합니다. 평가 코드는 실행 위치를 기준으로 파일을 읽으므로 평가 명령도 저장소 루트에서 실행합니다.

| 파일 | 압축 해제 후 배치 위치 | 용도 |
|---|---|---|
| `best_bev_model.pth` | `AGILEQ-Evaluation/best_bev_model.pth` | `eval.py`와 우회 데모의 BEVFormer |
| `best_traffic_model.pth` | `AGILEQ-Evaluation/best_traffic_model.pth` | `eval.py`와 우회 데모의 TLCFormer |

기본 `eval.py`에는 위 두 파일이 필요합니다. 인지 모델 가중치는 Google Drive에서 제공하므로 이를 받기 위해 `git lfs pull`을 실행할 필요는 없습니다.

압축파일에 `ieee_results/` 폴더도 들어 있다면 **하위 폴더 구조를 유지한 채** 저장소 루트로 복사합니다. 인지 모델의 `.pth`와 `--model`에 지정하는 주행 정책 `.zip`은 서로 다른 파일입니다. 압축파일에 `.pth`만 있다면 학습된 주행 정책 `.zip`을 별도로 준비해야 합니다.

## 주행 정책 가중치

평가 예제는 `ieee_results/AGILE-Q-C2/` 아래에 학습된 AGILE-Q-C2 정책이 있다고 가정합니다. 다운로드한 정책을 해당 위치에 넣거나, `--model`에 다른 정책 `.zip` 경로를 지정합니다.

- `v1`: 적색 신호 우회전 상황을 추가하기 전의 AGILE-Q-C2입니다. `--config crossq`를 사용하며 state에 `right_turn_indicator`와 `bypass_indicator`가 없습니다.
- `v2`: 적색 신호 우회전 상황도 다루는 AGILE-Q-C2입니다(`--config crossq_pp`).
- `v3`: v2에 Town05 우회 상황을 추가한 정책입니다. 기존 문서에서는 v3/v4 중 일반 주행에 더 강한 정책으로 설명합니다.
- `v4`: 아래 경로 처리를 적용해 v3를 우회용으로 fine-tuning한 정책입니다. 기존 문서에서는 v3/v4 중 우회에 더 강한 정책으로 설명합니다.

## 평가 실행

```bash
# 한 town 평가
python eval.py --config crossq_pp --model ieee_results/AGILE-Q-C2/v2/best_model.zip --town town01

# 사용자 지정 경로: spawn-point index 또는 x,y,z[,yaw]
python eval.py --config crossq_pp --model ieee_results/AGILE-Q-C2/v2/best_model.zip --town town04 \
    --start 116.77,228.87,0.3,-25.73 --dest 289.2,163.0,0.3

# 전체 town 평가
python run_experiments.py
```

출력 위치는 다음과 같습니다.

- 모델 파일 옆의 `eval_<town>/`: episode별 CSV, 요약 CSV, 영상
- `results/`: 종료 원인 통계

### 경로 처리(기본 비활성화)

`carla_env/envs/carla_FixedRoute_env.py`의 `CarlaRouteEnv`에는 v4 학습에 사용한 경로 처리에 대응하는 두 class attribute가 있습니다.

- `SMOOTH_LANE_CHANGES = False`: planner는 차선 변경을 1 m 간격 waypoint 사이의 3.5–7 m 도약으로 표현하며, 다중 차선 town에서는 이 때문에 off-track으로 episode가 끝날 수 있습니다. `True`로 설정하면 `carla_env/navigation/route_smoothing.py`로 점진적인 차선 변경 경로를 만들고, ego를 첫 경로 waypoint에 배치합니다.
- `ROUTE_EXTEND_M = 0`: `20`으로 설정하면 경로를 목적지 이후까지 연장해 guidance corridor가 목표보다 먼저 끝나지 않도록 합니다. 성공 판정과 경로 완주율은 원래 목적지를 기준으로 계산합니다.

기본값은 논문의 평가 설정을 유지합니다. v4를 평가하려면 두 값을 각각 `True`, `20`으로 설정합니다.

## State 구성

상태 목록은 `config.py`의 `states["test"]`입니다. 평가하는 정책을 학습할 때 사용한 state와 일치해야 합니다.

| Token | 내용 |
|---|---|
| A: ego | steer, throttle, brake, speed, jerk, 우회전 indicator, 우회 indicator(선택 사항) |
| B: BEV | ego, 주행 가능 영역, 차선, 보행자, 차량, 정지선, 잘린 경로의 7개 채널 |
| C: waypoints | 잘린 경로 waypoint 7개(T_wp) |
| D: traffic light | TLC 상태 4개, 정지선까지의 거리 |
| E: previous step | 이전 step의 ego 및 신호등 상태 |

## 우회(Bypass, 기본 비활성화)

`carla_env/bypass/plan_bypass.py`는 논문의 High-Level Waypoint Guidance(Algorithm 1)에 해당하는 우회 기능을 구현합니다.

| Mode | 동작 |
|---|---|
| G | 원래 경로 waypoint를 장애물 mask로 자릅니다. |
| F | `plan_bypass(METHOD, ...)`로 인접 차선에 진입했다가 복귀하는 경로를 생성하고 고정합니다. |

`METHOD`는 `frenet`, `lattice`, `astar`, `rrtstar`, `qp`, `nmpc`, `mppi` 중 선택하며 기본값은 `frenet`입니다. F 경로의 전방 waypoint(1 m 간격)도 같은 장애물 mask로 잘라, 우회 경로가 막히면 정책이 정지하도록 합니다. Ego가 우회 경로 끝에 도달하면 F가 종료됩니다.

우회는 다음 두 조건이 모두 충족되면 시작합니다.

1. **경로가 막힌 상태:** guidance waypoint Psi_T(최대 8개, 1 m 간격)가 장애물 mask로 잘리고(`k < |Psi_T|`), 신호가 적색이 아니며, ego 속도가 5 km/h 미만인 상태가 3 s 동안 유지되어야 합니다. 장애물 mask에는 차량과 보행자가 포함되며, 적색 신호에서 우회전 중이 아니라면 정지선도 포함됩니다.
2. **합법적이고 안전한 차선 변경:** 지도에서 차선 변경을 허용하며, 목표 차선 후방 영역의 차량 채널에 차량이 없어야 합니다. 후방 확인 영역은 ego 뒤 `REAR_CHECK_M = 15 m`입니다. 왼쪽 우회는 Omega_LR, 오른쪽 우회는 Omega_RR을 사용하며 `SIDE`로 구분합니다. 현재는 왼쪽 우회 경로만 구현되어 있습니다.

`carla_env/bypass/plan_bypass.py`의 `PlanBypass` class attribute로 동작을 설정합니다.

- `ENABLED = False`: `True`로 설정하면 우회를 활성화합니다.
- `OVERRIDE_ACTIONS = False`: `True`이면 F mode에서 `F_SPEED_CAP_KMH`(6 km/h)를 초과할 때 제동합니다. 우회용으로 학습하지 않은 정책에 적용하기 위한 기능이며, 우회 데모(`bypass_experiment/`)에서 사용했습니다.

우회 행동을 학습하려면 AGILE-Q 학습 코드에서 우회를 활성화하고 Town05 같은 다중 차선 town을 사용합니다.

### 평가용 정지 차량 시나리오(기본 비활성화)

`carla_env/bypass/bypass_scenario.py`는 `BypassScenario.ENABLED = False`가 기본값이며 Town03, Town04, Town05에서만 사용합니다. 평가 경로의 전방 30–40 m 중 왼쪽 차선 변경이 허용된 지점에 정지 차량을 배치합니다. 배치 규칙은 학습과 같으며 episode당 최대 3대입니다. Ego가 우회를 시작하지 않고 차량 뒤에서 `VANISH_S = 20 s` 동안 대기하면 해당 차량을 제거합니다.

Episode CSV에는 우회 mode와 episode별 횟수 및 간격(`bypass_spawned`, `bypass_triggers`, `bypass_entries`, `bypass_success`, `bypass_fail`, `bypass_vanished`, `bypass_stop_gap`)도 기록합니다. Ego가 정지 차량을 우회하도록 하려면 `PlanBypass`도 활성화합니다.

`states["test"]`에는 공개 정책 학습에 사용한 `"bypass_indicator"`가 포함됩니다. 우회가 비활성화되어 있으면 indicator는 `[0, 1]`을 유지합니다.

`bypass_experiment/`는 별도로 조정한 설정으로 우회 데모를 재현합니다. 자세한 내용은 [우회 데모 README](bypass_experiment/README.md)를 확인합니다.

### 구현 참고 사항

코드는 계속 시험하고 수정 중입니다. 논문이 모든 구현 세부 사항을 설명하지는 않으므로 일부 동작은 논문과 조금 다릅니다.

- 학습에서는 황색 신호도 우회 trigger를 막습니다. 평가의 TLCFormer는 적색과 황색을 통합합니다.
- 차선 변경의 합법성은 Psi_T만이 아니라 전체 우회 구간(`LEGAL_LOOKAHEAD_M = 50 m`까지)에서 확인합니다.
- F 시작 전에 경로를 따라 ego가 차지하는 swept footprint, 복귀 지점의 footprint, 곡률 및 횡가속도 제한도 확인합니다.
- 복귀 지점은 고정된 D_bp가 아닙니다. 장애물 mask로부터 장애물 끝 + ego 길이의 절반 + 1 m 위치를 구하고, planner가 그 이후 25 m 안에서 복귀 지점을 선택합니다.
- Ego가 복귀 지점을 지나 원래 경로에서 0.5 m 이내에 있거나 우회 경로 끝에 가까워지면 F가 종료됩니다.
- `bypass_indicator`는 `right_turn_indicator` 옆에 있는 ego token의 일부입니다.
- F에서는 조향 변화율 제한의 기준을 경로 PID steer가 아니라 마지막으로 적용한 steer로 둡니다.
- BYPASSFEASIBLE은 우회 방향의 후방 영역만 확인합니다(왼쪽: Omega_LR, 오른쪽: Omega_RR). 두 영역을 모두 확인하지는 않습니다.
- Reward 상수에 차이가 있습니다. `V_maxdiff`는 10 m/s(36 km/h, 논문 30 km/h), `sigma_max`는 0.35(논문 0.4)입니다. `D_maxct`는 논문처럼 2.0 m이지만, 공개 정책은 차선 중심 reward와 off-track 한계를 1.8 m로 설정해 학습했습니다.
- 우회 학습의 `R_vel^N`은 논문 결과 이후 재조정했습니다. `D_maxnpc = max(9 m, 2 s · V_ego)`(논문 20 m), `w_n = w_d = 0.5`(논문 0.4/0.6)를 사용하고, 선행 차량을 `max(10 m, 3 s · V_ego)` 안에서 고려합니다(이전 20 m). 공개 v1과 v2는 논문 값을 사용해 학습했습니다.
- 경로 처리 옵션(차선 변경 평활화, 경로 위에서 시작, 목적지 이후 연장)은 논문에 포함되지 않으며, 평가 코드에서는 기본 비활성화되어 있습니다.

### 알려진 제한

방해 차량의 전면을 지난 후 약 10 m 미만 지점에서 경로가 끝나면 원래 경로로 복귀할 공간이 부족해 우회 경로를 찾지 못합니다. 이 경우 ego는 60 s timeout까지 차량 뒤에서 대기합니다.
