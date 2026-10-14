# kslab — 자모 슬롯 복원 모델 실험실

k-safeguard의 "규칙으로 답이 안 나오는 난독화 케이스에 ML을 붙일 수 있는가"를 **저장소를
건드리지 않고** 먼저 실증하는 샌드박스다. 설계 견적 원본은
[`k-safeguard_ml_restoration_plan.md`](./k-safeguard_ml_restoration_plan.md).

## 이 폴더의 규칙

- **모든 파일은 여기 안에만.** venv, 코퍼스 캐시, 체크포인트, 결과물 전부.
  `datasets`·`torch`의 기본 캐시는 사용자 홈이라 `env.ps1`이 이를 `data/` 하위로 돌린다.
  러너는 시작할 때 이걸 확인하고 어긋나면 즉시 멈춘다.
- **k-safeguard 저장소는 읽기 전용.** 브랜치·커밋·PR 없음.
- **505개 dev 시드는 평가 전용.** 학습에 넣지 않는다(E0만 예외 — 아래 참고).
- **가드레일 추론은 이 로컬 환경에서 하지 않는다.** GPU가 없어 E0~E9b는 proxy 지표로만
  판단한다. 실제 NRR·ΔFPR이 필요하면 `notebooks/colab_kanana_guardrail_eval.ipynb`로
  Colab에서 돌리고 결과 zip을 다시 `results/`에 반영한다(E10 선례).
- **후보만 만들고 판단하지 않는다.** block/allow를 반환하는 코드는 이 폴더에 두지 않는다.
- **노트 없이 실행하지 않고, 리뷰 없이 다음 단계로 가지 않는다.** 스모크가 아닌 run은
  `notes/`에 실행 이유가 적혀 있어야 시작하고, 시작된 단계가 승인되기 전에는 다른 단계의
  run이 시작되지 않는다. [`notes/README.md`](./notes/README.md) 참고.

## 시작하기

```powershell
python -m venv .venv
. .\env.ps1                 # venv 활성화 + HF/torch 캐시를 ML\data\ 로 고정
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python -m unittest discover -s tests -v
python -m exp.run_e0_baseline_reproduce --run-id e0-<날짜>
```

bash를 쓴다면 `source ./env.sh`가 같은 일을 한다.

## 실험 기록 (MLflow)

`env.sh`/`env.ps1`을 거치면 러너가 끝날 때마다 MLflow에도 기록된다. 러너를 고칠 필요는
없다 — `kslab/record.py`의 `Run.start()`/`Run.finish()`가 `kslab/tracking.py`를 부른다.

```bash
source ./env.sh
mlflow ui --port 5000        # http://127.0.0.1:5000
```

| MLflow | 내용 |
|---|---|
| experiment / run name | 러너의 experiment 이름 / run-id. 스모크 실행은 `_smoke` experiment |
| description | 러너 모듈 docstring(실험 목적) |
| params | `config` |
| metrics | `finish(metrics)`의 수치를 `sweeps/tensify/bilstm/...` 식 키로 평탄화 |
| tags | 판정(`status`), git 커밋·dirty, 실행 명령, 호스트, 패키지 버전 |
| datasets | `add_input()`으로 등록한 입력의 이름·SHA-256·경로 (내용은 싣지 않음) |
| params (`fit.*`) | 모델이 `fit()`에서 스스로 올리는 하이퍼파라미터·학습 표본 수 — 러너에 하드코딩된 값도 잡힌다 |
| metrics (`headline/*`) | 러너가 고른 대표 지표. run끼리 비교할 때는 이것만 보면 된다 (`finish(metrics, headline=...)`) |
| metrics (`train/*/loss`) | epoch별 학습 loss (MLP·BiLSTM) |
| artifacts | `results/<run-id>/`의 `.md`·`.json`(report, metrics, manifest, 시작 시점 노트)와 `code.diff`(미커밋 변경 전체)·`pip-freeze.txt` |
| 노트·리뷰 | run 설명 = `notes/`의 노트. 태그 `kslab.stage`·`kslab.compares_to`·`review.status`·`review.feedback` |

- 저장소는 `data/mlflow/`(sqlite + artifacts)이고 커밋되지 않는다. 원본은 여전히
  `results/<run-id>/`이며 MLflow 기록이 실패해도 실험은 멈추지 않는다.
- `predictions.jsonl`처럼 공격 원문이 든 산출물은 MLflow에 올리지 않는다.
- 러너의 stdout은 `results/<run-id>/run.log`에 남는다. 러너가 원문 예시를 찍는 경우가 있어
  MLflow에는 올리지 않는다.
- 대표 지표(`headline`)는 지금 E8 러너에만 있다. 새 러너는 `finish()`에 직접 넘긴다.
- 러너가 `finish()` 전에 죽거나 중단되면 그 run은 `FAILED`로 남는다.
- MLflow 도입 전 실험(E0~E10)은 `python -m exp.backfill_mlflow`로 `baselines/*.json`에서
  수치만 옮겼다(`kslab.backfilled=true`). 다른 머신에서 새로 받았을 때도 이걸 한 번 돌리면 된다.
- 기록을 끄려면 `KSLAB_MLFLOW=0`.

## 실행 환경

GPU 없음(Intel UHD) · 12코어 · RAM 15.7GB · C: 여유 21GB. torch는 CPU 휠을 쓴다.
이 제약이 아키텍처 선택을 좌우한다 — 110M급 pretrained 인코더 fine-tune은 이 기계에서
현실적으로 불가능하므로, 가벼운 위치별 분류기로 어디까지 되는지를 먼저 본다.

## 구조

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
| `kslab/record.py` | run manifest, 경로 격리 검사, baseline 발행 |
| `kslab/tracking.py` | MLflow 기록(`record.Run`이 호출) |
| `kslab/notes.py`, `exp/stage.py` | 실험 노트와 단계 리뷰 게이트 |
| `notes/` | 커밋 대상 — 단계별 실험 이유·해석·리뷰 기록 |
| `exp/run_e*.py` | 실험 러너 |
| `baselines/`, `reports/` | 커밋 대상 — 집계 JSON과 보고서 |
| `results/`, `data/` | 커밋 제외 — 원문·코퍼스·체크포인트 |

## 지표와 승격 기준

승격 후보가 되려면 셋을 모두 넘어야 한다.

1. **Clean Mutation Rate ≤ 1%** — 난독화되지 않은 정상 문장을 복원기에 통과시켰을 때
   바뀌는 문장 비율. `EVALUATION_SPEC.md` §11.4 기준 그대로다. **1차 게이트**이고,
   이걸 못 넘으면 복원 정확도가 아무리 좋아도 승격하지 않는다 — 된소리 provider가
   정확히 이 축(ΔFPR-obf +14.29%p)에서 `DO_NOT_PROMOTE`가 됐다.
2. 해당 기법의 음절 exact가 현행 규칙/사전 방식 대비 유의하게 높을 것.
3. CPU p95 지연 ≤ 100ms(MLP) / ≤ 200ms(BiLSTM).

## 진행 상황

**먼저 [`reports/SUMMARY.md`](./reports/SUMMARY.md)를 읽는다** — E0~E10 전체를 묶은
종합 판정과 저장소에 돌려줄 산출물이 거기 있다.

실험별 수치는 [`EXPERIMENTS.md`](./EXPERIMENTS.md), 개별 상세는 [`reports/`](./reports/).
