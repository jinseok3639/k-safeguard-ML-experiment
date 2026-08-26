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
- **가드레일 추론은 여기서 하지 않는다.** NRR·ΔFPR은 GPU와 하네스가 있는 환경의 몫이고,
  여기서는 proxy 지표로만 판단한다.
- **후보만 만들고 판단하지 않는다.** block/allow를 반환하는 코드는 이 폴더에 두지 않는다.

## 시작하기

```powershell
python -m venv .venv
. .\env.ps1                 # venv 활성화 + HF/torch 캐시를 ML\data\ 로 고정
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python -m unittest discover -s tests -v
python -m exp.run_e0_baseline_reproduce --run-id e0-<날짜>
```

bash를 쓴다면 `source ./env.sh`가 같은 일을 한다.

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

**먼저 [`reports/SUMMARY.md`](./reports/SUMMARY.md)를 읽는다** — E0~E7 전체를 묶은
종합 판정과 저장소에 돌려줄 산출물이 거기 있다.

실험별 수치는 [`EXPERIMENTS.md`](./EXPERIMENTS.md), 개별 상세는 [`reports/`](./reports/).
