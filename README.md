# kslab — k-safeguard ML 실험

[k-safeguard](https://github.com/jinseok3639/k-safeguard)에서 "규칙으로 답이 안 나오는 난독화
케이스에 ML을 붙일 수 있는가"를 실증하는 실험 저장소다. 라이브러리에 들어가는 결정의 근거가
여기에 있다. 실험은 두 트랙으로 나뉜다.

| 트랙 | 묻는 것 | 시작점 |
|---|---|---|
| [`restoration/`](./restoration/README.md) | 복원기(난독화 → 원문 후보)를 얼마나 잘 만들 수 있나 | [`restoration/reports/SUMMARY.md`](./restoration/reports/SUMMARY.md) |
| [`guardrail/`](./guardrail/README.md) | 어떤 가드레일로, 어떤 난독화가 실제로 위협인지 재나 | [`guardrail/CANDIDATES.md`](./guardrail/CANDIDATES.md) |

두 트랙이 함께 쓰는 데이터 조사와 사용 결정은 [`DATASETS.md`](./DATASETS.md)에 있다.

## 결과 읽는 법

- **트랙별 실험 대장** `<트랙>/EXPERIMENTS.md` — 맨 위 단계 표에 질문과 답이 한 줄씩, 아래 run
  표에 run별 핵심 수치와 판정이 있다.
- **보고서** `<트랙>/reports/` — 실험별 상세. 복원기 트랙은 `SUMMARY.md`가 전체를 묶는다.
- **집계 수치** `<트랙>/baselines/*.json` — 보고서 숫자의 원본. 테스트가 재현 값을 대조한다.

## 실험 원칙

- **505개 dev 시드는 평가 전용.** 학습에 넣지 않는다.
- **후보만 만들고 판단하지 않는다.** block/allow를 반환하는 코드는 이 저장소에 두지 않는다.
- **k-safeguard 저장소는 읽기 전용으로 참조한다.**
- **실행 전에 질문과 예상을 적고, 결과를 본 뒤 고치지 않는다.** 러너는 실행 시점의 노트 해시를
  manifest에 남긴다.
- **공격 원문이 든 산출물은 공개하지 않는다.** run 산출물(`<트랙>/results/`)과 데이터(`data/`)는
  커밋하지 않고, 집계는 `baselines/`와 `reports/`로만 남긴다.

## 시작하기

```bash
source ./env.sh             # Windows PowerShell: . .\env.ps1
python -m unittest discover -s tests

python -m restoration.exp.stage status
python -m restoration.exp.run_e8_abstention --help
```

`env.sh`·`env.ps1`은 venv를 활성화하고 HF·torch·MLflow 저장 위치를 저장소 안 `data/`로 고정한다.
러너는 시작할 때 이를 확인하고 어긋나면 멈춘다. venv를 새로 만들 때는 `requirements.txt` 첫 줄의
안내대로 머신에 맞는 torch 휠 인덱스를 쓴다(Python 3.14, `torch 2.9.1+cu128`에서 확인).

러너는 반드시 `python -m <트랙>.exp.<러너>`로 실행한다. 결과·보고서 위치는 러너 패키지가 정한다.

## 구조

```
kslab/            공유 라이브러리 — 자모·난독화 생성기·모델·지표·기록 (kslab/README.md)
tests/            공유 라이브러리 테스트
restoration/      복원기 실험 트랙
guardrail/        가드레일 실험 트랙
  exp/            러너 (python -m <트랙>.exp.<러너>)
  reports/        보고서
  baselines/      집계 JSON
  results/        run 산출물 — 공격 원문 포함 (커밋 제외)
  EXPERIMENTS.md  실험 대장
data/             코퍼스·모델 캐시·데이터 샘플·MLflow 저장소 (커밋 제외)
```

| 경로 | 내용 |
|---|---|
| `kslab/jamo.py` | 자모 분해·조합. 테이블은 k-safeguard `normalization.py:18-34`와 동일(테스트로 대조) |
| `kslab/phonology.py` | 표준발음법 규칙 테이블(격음화·비음화·유음화·평파열음화·자음군 단순화·경음화) + 모음 혼동 |
| `kslab/obfuscate.py` | 난독화 생성기 17종. k-safeguard main의 5종 + liaison·palatalize·jongseong_cram + E9의 음운 규칙 9종 |
| `kslab/compose.py` | 다중(조합) 난독화 — 프로파일, 단계별 시드, 손상량 보정, 자리별 출처 추적 |
| `kslab/pairs.py` | (난독화, 원문) 페어 구성, source-disjoint 분할, 누출 검사 |
| `kslab/features.py` | 후보 위치 규칙과 슬롯 특징 인코딩. 조합 기법은 구성 기법의 합집합으로 해석 |
| `kslab/ceiling.py` | 학습 없이 구하는 구조적 복원 상한(도달불가/표현불가 분해) |
| `kslab/models/` | linear(로지스틱) → mlp → bilstm 사다리 |
| `kslab/metrics.py` | proxy 지표 + seed-cluster bootstrap (k-safeguard 구현 이식) + 자리 단위 4분류 |
| `kslab/tracks.py` | 트랙 구분 — 트랙별 결과·보고서·대장 위치 |
| `kslab/record.py` | run manifest, 경로 격리 검사, baseline 발행 |
| `kslab/tracking.py` | MLflow 기록(`record.Run`이 호출) |
| `kslab/notes.py`, `kslab/stage_cli.py` | 실험 노트와 단계 리뷰 게이트 |

## 실험 기록 (MLflow)

`env.sh`/`env.ps1`을 거치면 러너가 끝날 때마다 MLflow에도 기록된다. `kslab/record.py`의
`Run.start()`/`Run.finish()`가 `kslab/tracking.py`를 부른다. 실험 = 트랙, 단계 = 부모 run,
실험 run = 자식 run, 모델 학습 한 번 = 손자 run으로 묶는다.

```bash
mlflow ui --port 5000 --artifacts-destination "$KSLAB_ROOT/data/mlflow/served-artifacts"
# 브라우저에서 http://127.0.0.1:5000
```

| MLflow | 내용 |
|---|---|
| run 이름 | 보이는 이름(`S02.2 기법군 빼고 학습`). 기계용 id는 태그 `kslab.run_id` |
| Metrics | 요약만 — `headline/*`(러너가 고른 대표 지표)와 요약 수치. 전체 수치는 artifact `metrics.json` |
| Parameters | `config`. 모델 run에는 하이퍼파라미터·학습 표본 수 |
| Tags | 트랙, 단계, 판정(`status`), git 커밋·dirty, 실행 명령, 패키지 버전 |
| Datasets | `add_input()`으로 등록한 입력의 이름·SHA-256·경로 (내용은 싣지 않음) |
| Artifacts | `report.md`, `metrics.json`, `manifest.json`, `code.diff`, `pip-freeze.txt` |

- MLflow 저장소(`data/mlflow/`)는 커밋되지 않는다. 원본은 `<트랙>/results/<run-id>/`와 노트이며,
  MLflow 기록이 실패해도 실험은 멈추지 않는다.
- 다른 머신에서 새로 받았거나 기록 구성을 바꿨으면 `python -m kslab.mlflow_rebuild`로 원본에서 다시
  쓴다. 기존 저장소는 지우지 않고 `data/mlflow/v1-<시각>/`로 옮긴다. 결과 폴더가 없는 E0~E10은
  `restoration/baselines/*.json`에서 수치만 옮긴다.
- `predictions.jsonl`·`run.log`처럼 공격 원문이 들 수 있는 산출물은 MLflow에 올리지 않는다.
- 기록을 끄려면 `KSLAB_MLFLOW=0`. 테스트는 임시 저장소를 쓰거나 기록을 끈다.
