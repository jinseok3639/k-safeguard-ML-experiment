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
구성은 MLflow 공식 문서의 방식(실험 = 과제, 부모·자식 run으로 묶기, 태그로 거르기, 노트로
설명)을 따른다.

```bash
source ./env.sh
mlflow ui --port 5000 --artifacts-destination "$KSLAB_ROOT/data/mlflow/served-artifacts"
# 브라우저에서 http://127.0.0.1:5000
```

`--artifacts-destination`은 MLflow 연습 스크립트처럼 **서버 주소로** 기록하는 코드가 올리는
artifact의 저장 위치다. 빼면 명령을 실행한 폴더에 `mlartifacts/`가 생긴다. 러너는 서버를 거치지
않고 sqlite에 직접 쓰므로 이 옵션과 무관하다.

### 화면에서 보이는 구조

```
restoration                                   ← 실험(Experiment) = 트랙
  S02 생성기 의존도                             ← 단계 run. Notes = STAGE.md (질문·답·근거)
    S02.2 기법군 빼고 학습                      ← 실험 run. Notes = run 노트
      bilstm · minus_tensify · seed2026         ← 모델 학습 한 번. loss 곡선·하이퍼파라미터
  S00 단계 도입 전 실험 (E0~E10)                ← 옛 실험(집계 수치만)
guardrail                                     ← 가드레일 트랙 (첫 run부터 생긴다)
```

### 처음 볼 때

1. 왼쪽에서 `restoration`을 누른다. 단계 run(`S02 …`)이 보이고, 왼쪽 `+`로 펼치면 그 안의
   실험 run이 나온다.
2. **단계 run을 누르면 Notes 칸 맨 위에 질문 → 답 → 근거 숫자**가 있다. 대부분은 여기서 끝난다.
3. 숫자를 확인하고 싶으면 실험 run → **Artifacts → `report.md`**(정리된 표). 원본 수치 전체는
   `metrics.json`이다.
4. 학습 곡선은 모델 run(`bilstm · …`) → Model metrics → `loss`.

### 무엇이 어디에 있나

| MLflow | 내용 |
|---|---|
| run 이름 | 사람이 보는 이름(`S02.2 기법군 빼고 학습`). 기계용 id는 태그 `kslab.run_id` |
| Notes | 단계 run: `STAGE.md`. 실험 run: run 노트(실행 시점 → `stage sync`로 해석까지 갱신) |
| Metrics | **요약만** — `headline/*`(러너가 고른 대표 지표)와 요약 수치. 세부 평가표(`evaluations`·`models`)는 싣지 않는다. run 하나에 수천 개를 올리면 화면이 멈춘다 |
| Parameters | `config`. 모델 run에는 그 모델의 하이퍼파라미터·학습 표본 수 |
| Tags | 트랙, 단계, 판정(`status`), 리뷰 상태(`review.status`), git 커밋·dirty, 실행 명령, 패키지 버전 |
| Datasets | `add_input()`으로 등록한 입력의 이름·SHA-256·경로 (내용은 싣지 않음) |
| Artifacts | `report.md`, `metrics.json`(전체 수치), `manifest.json`, 시작 시점 노트, `code.diff`, `pip-freeze.txt` |

- 저장소는 `data/mlflow/`(sqlite + artifacts)이고 커밋되지 않는다. **원본은 `<트랙>/results/<run-id>/`와
  `<트랙>/notes/`**이며 MLflow 기록이 실패해도 실험은 멈추지 않는다.
- 기록 구성을 바꿨거나 다른 머신에서 새로 받았으면 `python -m kslab.mlflow_rebuild`로 원본에서
  다시 쓴다. 기존 저장소는 지우지 않고 `data/mlflow/v1-<시각>/`로 옮긴다(실행 중인 `mlflow ui`는 먼저 끈다).
  결과 폴더가 없는 옛 실험(E0~E10)은 `restoration/baselines/*.json`에서 수치만 옮긴다.
- `predictions.jsonl`처럼 공격 원문이 든 산출물과 `run.log`(원문 예시가 찍힐 수 있음)는 MLflow에
  올리지 않는다.
- 러너가 `finish()` 전에 죽거나 중단되면 그 run은 `FAILED`로 남는다.
- 기록을 끄려면 `KSLAB_MLFLOW=0`. 테스트는 임시 저장소를 쓰거나 기록을 끈다.

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
