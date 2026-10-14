# kslab — k-safeguard ML 실험실

k-safeguard의 "규칙으로 답이 안 나오는 난독화 케이스에 ML을 붙일 수 있는가"를 **저장소를
건드리지 않고** 먼저 실증하는 샌드박스다. 실험은 두 트랙으로 나뉜다.

| 트랙 | 묻는 것 | 시작점 |
|---|---|---|
| [`restoration/`](./restoration/README.md) | 복원기(난독화 → 원문 후보)를 얼마나 잘 만들 수 있나 | [`restoration/reports/SUMMARY.md`](./restoration/reports/SUMMARY.md) |
| [`guardrail/`](./guardrail/README.md) | 어떤 가드레일로, 어떤 난독화가 실제로 위협인지 재나 | [`guardrail/CANDIDATES.md`](./guardrail/CANDIDATES.md) |

두 트랙이 함께 쓰는 데이터 조사와 사용 결정은 [`DATASETS.md`](./DATASETS.md)에 있다.

## 이 폴더의 규칙

- **모든 파일은 여기 안에만.** venv, 코퍼스 캐시, 모델 캐시, 체크포인트, 결과물 전부.
  `datasets`·`torch`·HF의 기본 캐시는 사용자 홈이라 `env.sh`·`env.ps1`이 이를 `data/`
  하위로 돌린다. 러너는 시작할 때 이걸 확인하고 어긋나면 즉시 멈춘다.
- **k-safeguard 저장소는 읽기 전용.** 브랜치·커밋·PR 없음.
- **505개 dev 시드는 평가 전용.** 학습에 넣지 않는다.
- **후보만 만들고 판단하지 않는다.** block/allow를 반환하는 코드는 이 폴더에 두지 않는다.
- **노트 없이 실행하지 않고, 리뷰 없이 다음 단계로 가지 않는다.** 게이트는 트랙별로 돈다 —
  [`NOTES.md`](./NOTES.md) 참고.
- **모의 실험·사전 실측을 임의로 돌리지 않는다.** 특히 큰 모델을 이 머신에 올리는 일은 먼저
  묻는다. 8B 모델 로드로 WSL이 통째로 재시작된 전례가 있다(2026-10-13).

## 시작하기

```bash
source ./env.sh             # venv 활성화 + HF/torch/MLflow 저장 위치를 data/ 로 고정
python -m unittest discover -s tests

python -m restoration.exp.stage status         # 복원기 트랙 단계 상태
python -m guardrail.exp.stage status           # 가드레일 트랙 단계 상태
python -m restoration.exp.run_e8_abstention --help
```

러너는 반드시 `python -m <트랙>.exp.<러너>`로 실행한다. 트랙(결과·노트·보고서 위치)은 러너
패키지가 정하므로, 파일 경로로 직접 실행하면 트랙이 정해지지 않았다며 멈춘다.

venv를 새로 만들 때는 `requirements.txt` 첫 줄의 안내대로 머신에 맞는 torch 휠 인덱스를 쓴다.
Windows PowerShell이면 `. .\env.ps1`이 같은 일을 한다.

## 구조

```
kslab/            공유 라이브러리 — 자모·난독화 생성기·모델·지표·기록
tests/            공유 라이브러리 테스트
restoration/      복원기 실험 트랙
guardrail/        가드레일 실험 트랙
  exp/            러너 (python -m <트랙>.exp.<러너>)
  notes/          단계별 노트·리뷰 기록 (커밋)
  reports/        보고서 (커밋)
  baselines/      집계 JSON (커밋)
  results/        run 산출물 — 공격 원문 포함 (커밋 제외)
  EXPERIMENTS.md  실험 대장
data/             코퍼스·모델 캐시·데이터 샘플·MLflow 저장소 (커밋 제외, 두 트랙 공유)
NOTES.md          노트·리뷰 게이트 사용법
DATASETS.md       데이터셋 조사와 사용 결정
```

| 경로 | 내용 |
|---|---|
| `kslab/jamo.py` | 자모 분해·조합. 테이블은 저장소 `normalization.py:18-34`와 동일(테스트로 대조) |
| `kslab/phonology.py` | 표준발음법 규칙 테이블(격음화·비음화·유음화·평파열음화·자음군 단순화·경음화) + 모음 혼동 |
| `kslab/obfuscate.py` | 난독화 생성기 17종. 저장소 main의 5종 + liaison·palatalize·jongseong_cram + E9의 음운 규칙 9종 |
| `kslab/compose.py` | 다중(조합) 난독화 — 프로파일, 단계별 시드, 손상량 보정, 자리별 출처 추적 |
| `kslab/pairs.py` | (난독화, 원문) 페어 구성, source-disjoint 분할, 누출 검사 |
| `kslab/features.py` | 후보 위치 규칙과 슬롯 특징 인코딩. 조합 기법은 구성 기법의 합집합으로 해석 |
| `kslab/ceiling.py` | 학습 없이 구하는 구조적 복원 상한(도달불가/표현불가 분해) |
| `kslab/models/` | linear(로지스틱) → mlp → bilstm 사다리 |
| `kslab/metrics.py` | proxy 지표 + seed-cluster bootstrap (저장소 구현 이식) + 자리 단위 4분류 |
| `kslab/tracks.py` | 트랙 구분 — 트랙별 결과·노트·보고서·대장 위치 |
| `kslab/record.py` | run manifest, 경로 격리 검사, baseline 발행 |
| `kslab/tracking.py` | MLflow 기록(`record.Run`이 호출) |
| `kslab/notes.py`, `kslab/stage_cli.py` | 실험 노트와 단계 리뷰 게이트 |

## 실험 기록 (MLflow)

`env.sh`/`env.ps1`을 거치면 러너가 끝날 때마다 MLflow에도 기록된다. 러너를 고칠 필요는
없다 — `kslab/record.py`의 `Run.start()`/`Run.finish()`가 `kslab/tracking.py`를 부른다.
두 트랙이 같은 MLflow 저장소를 쓰고, run마다 `kslab.track` 태그로 구분된다.

```bash
source ./env.sh
mlflow ui --port 5000        # http://127.0.0.1:5000
```

| MLflow | 내용 |
|---|---|
| experiment / run name | 러너의 experiment 이름 / run-id. 스모크 실행은 `_smoke` experiment |
| description | 노트의 "왜 돌리나"·"무엇을 바꿨나" (노트가 없는 스모크는 러너 docstring) |
| params | `config` |
| metrics | `finish(metrics)`의 수치를 `sweeps/tensify/bilstm/...` 식 키로 평탄화 |
| tags | 트랙(`kslab.track`), 판정(`status`), git 커밋·dirty, 실행 명령, 호스트, 패키지 버전 |
| datasets | `add_input()`으로 등록한 입력의 이름·SHA-256·경로 (내용은 싣지 않음) |
| params (`fit.*`) | 모델이 `fit()`에서 스스로 올리는 하이퍼파라미터·학습 표본 수 — 러너에 하드코딩된 값도 잡힌다 |
| metrics (`headline/*`) | 러너가 고른 대표 지표. run끼리 비교할 때는 이것만 보면 된다 (`finish(metrics, headline=...)`) |
| metrics (`train/*/loss`) | epoch별 학습 loss (MLP·BiLSTM) |
| artifacts | `<트랙>/results/<run-id>/`의 `.md`·`.json`(report, metrics, manifest, 시작 시점 노트)와 `code.diff`(미커밋 변경 전체)·`pip-freeze.txt` |
| 노트·리뷰 | 태그 `kslab.stage`·`kslab.compares_to`·`review.status`·`review.feedback` |

- 저장소는 `data/mlflow/`(sqlite + artifacts)이고 커밋되지 않는다. 원본은 여전히
  `<트랙>/results/<run-id>/`이며 MLflow 기록이 실패해도 실험은 멈추지 않는다.
- `predictions.jsonl`처럼 공격 원문이 든 산출물은 MLflow에 올리지 않는다.
- 러너의 stdout은 `results/<run-id>/run.log`에 남는다. 러너가 원문 예시를 찍는 경우가 있어
  MLflow에는 올리지 않는다.
- 대표 지표(`headline`)는 지금 E8 러너에만 있다. 새 러너는 `finish()`에 직접 넘긴다.
- 러너가 `finish()` 전에 죽거나 중단되면 그 run은 `FAILED`로 남는다.
- MLflow 도입 전 복원기 실험(E0~E10)은 `python -m restoration.exp.backfill_mlflow`로
  `restoration/baselines/*.json`에서 수치만 옮겼다(`kslab.backfilled=true`). 다른 머신에서
  새로 받았을 때도 이걸 한 번 돌리면 된다.
- 기록을 끄려면 `KSLAB_MLFLOW=0`.

## 실행 환경

2026-10 기준 WSL2 · Intel Ultra 7 155H 22스레드 · RAM 27GB · **GPU RTX 4050 Laptop 6GB**.
WSL에서는 `nvidia-smi`가 PATH에 없을 수 있다(`/usr/lib/wsl/lib/nvidia-smi`).

- venv: Python 3.14, `torch 2.9.1+cu128`. 단 복원기 모델 코드(`kslab/models/`)에는 아직 GPU
  처리가 없어 CPU로 학습한다.
- 가드레일 측정용으로 `transformers 4.57.6`·`accelerate`·`bitsandbytes`가 venv에 있지만
  `requirements.txt`에는 아직 넣지 않았다 — [`guardrail/README.md`](./guardrail/README.md)의
  환경 충돌 참고.
- E0~E10은 이전 머신(Windows, GPU 없음, 12코어, RAM 15.7GB)에서 돌렸다. 그 시점의 아키텍처
  선택은 CPU 제약을 전제로 했다.
