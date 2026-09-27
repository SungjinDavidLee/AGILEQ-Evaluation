# AGILEQ-Evaluation

[English](README.md) · [학습 코드](https://github.com/SungjinDavidLee/AGILEQ-Training/tree/main/AGILEQ-train)

`AGILE-Q-eval`의 CARLA 평가 코드입니다. 주변 카메라 영상의 BEVFormer 인지와 연속 전방 영상의 신호등 인지를 사용하여 강화학습 주행 정책을 평가합니다. 코드는 저장소 최상위에 배치되어 있습니다.

## 구성

| 경로 | 역할 |
| --- | --- |
| `eval.py` | 정책 로드, 주행 에피소드 평가 및 결과 저장 |
| `eval_plots.py` | 평가 CSV 요약 및 시각화 함수 |
| `run_experiments.py` | 여러 Town에서 지정 체크포인트 평가 |
| `config.py`, `crossq_pp/` | 알고리즘 설정 및 CrossQ++ 구현 |
| `carla_env/envs/carla_FixedRoute_env.py` | `eval.py`에서 사용하는 환경 |
| `carla_env/envs/carla_RandomRoute_env_traffic.py` | 별도로 포함된 무작위 경로 환경 |
| `carla_env/envs/perception_models/` | BEVFormer, 신호등 모델, InternImage 설정 및 확장 연산 소스 |
| `BEV/` | 지도 렌더링 유틸리티 및 지도 자료 |
| `environment_agileq_eval.yml` | Conda 환경 내보내기 파일 |

## 환경 설정

명령은 저장소 최상위에서 실행합니다. Linux, 필요한 Town 맵이 설치된 CARLA 0.9.15, NVIDIA GPU, PyTorch와 호환되는 CUDA 툴킷·컴파일러가 필요합니다. 제공 환경은 Python 3.9와 PyTorch 2.7 및 CUDA 12.8 패키지를 사용하며, 추론 코드에서 CUDA를 직접 지정합니다.

```bash
conda env create -n agileq_eval -f environment_agileq_eval.yml
conda activate agileq_eval
export CARLA_ROOT=/absolute/path/to/CARLA_0.9.15
```

YAML에는 특정 플랫폼의 빌드와 CUDA wheel 버전이 기록되어 있습니다. 해당 빌드를 설치할 수 없다면 사용 환경에 맞게 패키지 공급 경로나 버전을 조정해야 합니다. `CrossQ`와 `BatchRenorm1d`를 제공하는 `sb3_contrib` 환경이 필요합니다.

활성화한 환경에서 두 확장 패키지를 빌드합니다.

```bash
(cd carla_env/envs/perception_models/models/ops_dcnv3 && bash make.sh)
(cd carla_env/envs/perception_models/ops && bash make.sh)
```

DCNv3와 multi-scale deformable attention의 소스를 포함하며 컴파일된 바이너리는 포함하지 않습니다. `get_config_t()`가 사용하는 InternImage 설정은 `carla_env/envs/perception_models/internimage_t_1k_224.yaml`에 있습니다.

## 필요한 체크포인트

| 파일 | 역할 |
| --- | --- |
| `/absolute/path/to/run/model_100000_steps.zip` | `--model`로 전달하는 학습된 주행 정책 |
| `./best_bev_model.pth` | 평가 환경에서 불러오는 BEVFormer 가중치 |
| `./best_traffic_model.pth` | 평가 환경에서 불러오는 신호등 모델 가중치 |

가중치는 포함되어 있지 않습니다. 인지 모델 체크포인트를 저장소 최상위에 배치하거나 환경 코드의 로딩 경로를 수정해야 합니다. 체크포인트의 모델 구조와 정책 관측 공간은 이 코드와 일치해야 합니다. EfficientNet 초기화 시 캐시가 없다면 사전학습 백본 가중치를 내려받을 수 있습니다.

## 평가 실행

```bash
python eval.py \
  --config crossq_pp \
  --model /absolute/path/to/run/model_100000_steps.zip \
  --town town01 \
  --port 2000 \
  --no_render \
  --no_record_video
```

정책에 맞는 설정을 `PPO`, `SAC`, `DDPG`, `TD3`, `TQC`, `crossq`, `crossq_pp` 중 선택합니다. 환경은 `CARLA_ROOT/CarlaUE4.sh`로 CARLA를 실행합니다.

`--no_render`는 환경 화면 표시를, `--no_record_video`는 AVI 녹화를 끕니다. 경로를 지정하려면 `--start`와 `--dest`를 함께 전달합니다. 스폰 지점 인덱스 또는 쉼표로 구분한 좌표를 사용할 수 있으며, 예시는 `--start 175 --dest 42`입니다. 현재 `eval.py` 반복문은 21개 에피소드를 실행합니다. `--iteration`은 위반 이력 출력의 식별 값이며 에피소드 수를 변경하지 않습니다.

Town 이름은 `BEV/`의 폴더 이름과 대소문자까지 일치해야 합니다. 포함된 이름은 `town01`, `town02`, `town03`, `town04`, `town05`, `town07`, `Town10HD_Opt`입니다. 대응하는 CARLA 맵도 설치되어 있어야 합니다.

일괄 평가 전에는 `run_experiments.py`의 `CARLA_ROOT`, 체크포인트 경로, Town 목록 및 알고리즘 설정을 수정합니다. 스크립트는 실행 사이에 CARLA 서버 프로세스를 종료합니다. 무작위 경로 환경도 별도로 포함되어 있으나, 현재 진입점은 고정 경로 환경을 불러옵니다.

## 평가 결과

단계별 CSV와 선택적 AVI 영상은 정책 체크포인트 옆의 `eval_<town>/`에 저장됩니다. CSV에는 제어 입력, 차량 자세, 경로 지점, 보상, 속도, 차로 중앙 이탈 거리, 경로 완주율 및 위반 페널티가 기록됩니다. `summary_eval()`이 요약 결과를 생성하며, 환경은 위반 이력과 종료 통계도 저장합니다.

Python 주석과 docstring은 제거했으며 제3자 헤더 고지는 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)에 보존했습니다. 반영 과정에서 Python 문법과 실행 구문의 보존을 확인했습니다. CUDA 컴파일과 실제 CARLA 평가는 실행하지 않았습니다.
